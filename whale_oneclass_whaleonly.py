"""
Whale Presence Detection — One-Class Autoencoder (Whale-Only)
==============================================================
All three splits use ONLY whale data.

Pipeline:
  Train  : learn to reconstruct whale spectrograms
  Val    : monitor reconstruction error + set detection threshold
           from the whale error distribution (e.g. 95th percentile)
  Test   : report reconstruction error statistics on whale clips

At inference on unknown audio:
  error <= threshold  →  whale
  error >  threshold  →  no whale (anomaly)

Expected dataset structure:
    dataset/
        train/
            whale/   *.mp3
        val/
            whale/   *.mp3
        test/
            whale/   *.mp3
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

class Config:
    # Audio
    SAMPLE_RATE = 22050
    DURATION    = 8.0
    N_MELS      = 128
    N_FFT       = 2048
    HOP_LENGTH  = 256
    F_MIN       = 0
    F_MAX       = 1000

    # Model
    LATENT_DIM   = 64    # bottleneck size — smaller = stricter reconstruction
    BASE_CH      = 32    # base conv channels

    # Training
    BATCH_SIZE   = 16
    EPOCHS       = 50
    LR           = 1e-3
    WEIGHT_DECAY = 1e-5
    PATIENCE     = 10    # early stopping patience

    # Threshold: set at this percentile of val whale errors
    # e.g. 95 means "flag anything with higher error than 95% of known whale clips"
    THRESHOLD_PERCENTILE = 95

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Augmentation
    AUGMENT       = True
    TIME_MASK_MAX = 20
    FREQ_MASK_MAX = 16

    # Paths
    DATA_DIR        = "dataset"
    CHECKPOINT_PATH = "best_autoencoder.pt"
    THRESHOLD_PATH  = "detection_threshold.npy"
    SEED            = 42


def set_seed(seed: int = Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# 2. DATASET  (whale-only)
# ─────────────────────────────────────────────

class WhaleOnlyDataset(Dataset):
    """
    Loads MP3s from a single 'whale' folder.
    Returns spectrogram tensor — no labels needed.
    """

    def __init__(self, whale_dir: str, cfg: Config, augment: bool = False):
        self.cfg     = cfg
        self.augment = augment

        if not os.path.isdir(whale_dir):
            raise FileNotFoundError(f"Folder not found: '{whale_dir}'")

        self.files = sorted([
            os.path.join(whale_dir, f)
            for f in os.listdir(whale_dir)
            if f.lower().endswith(".mp3")
        ])

        if not self.files:
            raise FileNotFoundError(f"No MP3 files found in '{whale_dir}'")

        self._target_len    = int(cfg.SAMPLE_RATE * cfg.DURATION)
        self._target_frames = int(
            np.ceil(cfg.SAMPLE_RATE * cfg.DURATION / cfg.HOP_LENGTH)
        )

    def _load_audio(self, path: str) -> np.ndarray:
        y, _ = librosa.load(
            path, sr=self.cfg.SAMPLE_RATE,
            duration=self.cfg.DURATION, mono=True
        )
        tl = self._target_len
        if len(y) < tl:
            y = np.pad(y, (0, tl - len(y)))
        else:
            y = y[:tl]
        return y

    def _to_melspec(self, y: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        S   = librosa.feature.melspectrogram(
            y=y, sr=cfg.SAMPLE_RATE, n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LENGTH, n_mels=cfg.N_MELS,
            fmin=cfg.F_MIN, fmax=cfg.F_MAX
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

        T = self._target_frames
        if S_db.shape[1] < T:
            S_db = np.pad(S_db, ((0, 0), (0, T - S_db.shape[1])))
        else:
            S_db = S_db[:, :T]
        return S_db.astype(np.float32)

    def _spec_augment(self, spec: np.ndarray) -> np.ndarray:
        cfg  = self.cfg
        n, t = spec.shape
        f0   = random.randint(0, max(0, n - cfg.FREQ_MASK_MAX))
        spec[f0:f0 + random.randint(0, cfg.FREQ_MASK_MAX), :] = 0.0
        t0   = random.randint(0, max(0, t - cfg.TIME_MASK_MAX))
        spec[:, t0:t0 + random.randint(0, cfg.TIME_MASK_MAX)] = 0.0
        return spec

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        y    = self._load_audio(self.files[idx])
        spec = self._to_melspec(y)
        if self.augment and self.cfg.AUGMENT:
            spec = self._spec_augment(spec)
        # shape: (1, N_MELS, T)
        return torch.tensor(spec[np.newaxis], dtype=torch.float32)


# ─────────────────────────────────────────────
# 3. MODEL: CONVOLUTIONAL AUTOENCODER
# ─────────────────────────────────────────────

class CPUAdaptiveAvgPool2d(nn.Module):
    """
    Workaround for MPS limitation:
    AdaptiveAvgPool2d with non-divisible sizes is not supported on MPS.
    This wrapper temporarily moves the tensor to CPU for that one layer.
    Safe to use on CPU/CUDA too — has zero overhead on those devices.
    """
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        device = x.device
        return self.pool(x.cpu()).to(device)


class ConvAutoencoder(nn.Module):
    """
    Encoder  : spectrogram → latent vector
    Decoder  : latent vector → reconstructed spectrogram
    Anomaly score = reconstruction error (MSE per sample)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        ch = cfg.BASE_CH

        self.encoder = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(ch, ch*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch*2, ch*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch*2), nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(ch*2, ch*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch*4, ch*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch*4), nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(ch*4, ch*8, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch*8), nn.LeakyReLU(0.2, inplace=True),
            CPUAdaptiveAvgPool2d((4, 4)),   # MPS-safe adaptive pooling
        )

        self._ch            = ch
        self.bottleneck_in  = nn.Linear(ch*8 * 4 * 4, cfg.LATENT_DIM)
        self.bottleneck_out = nn.Linear(cfg.LATENT_DIM, ch*8 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ch*8, ch*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*4), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch*4, ch*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*2), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch*2, ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch, 1, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.bottleneck_in(h)

    def decode(self, z):
        ch = self._ch
        h  = self.bottleneck_out(z).view(-1, ch*8, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        recon = self.decode(self.encode(x))
        recon = nn.functional.interpolate(
            recon, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return recon


def reconstruction_error(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE, shape (B,)."""
    return ((x - recon) ** 2).mean(dim=[1, 2, 3])


# ─────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience  = patience
        self.best_loss = float("inf")
        self.counter   = 0

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - 1e-5:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_autoencoder(cfg: Config):
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    print(f"Using device: {device}\n")

    train_ds = WhaleOnlyDataset(
        os.path.join(cfg.DATA_DIR, "train", "whale"), cfg, augment=True
    )
    val_ds = WhaleOnlyDataset(
        os.path.join(cfg.DATA_DIR, "val", "whale"), cfg, augment=False
    )

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=False)

    model     = ConvAutoencoder(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR,
                                  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    stopper   = EarlyStopping(patience=cfg.PATIENCE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    history = {"train_loss": [], "val_loss": [],
               "val_err_mean": [], "val_err_std": []}
    best_val_loss = float("inf")

    for epoch in range(1, cfg.EPOCHS + 1):

        # ── Train ────────────────────────────
        model.train()
        train_loss = 0.0
        for X in tqdm(train_loader, desc=f"Epoch {epoch:3d} Train", leave=False):
            X     = X.to(device)
            loss  = reconstruction_error(X, model(X)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X)
        train_loss /= len(train_ds)

        # ── Validate ─────────────────────────
        model.eval()
        val_errors = []
        with torch.no_grad():
            for X in tqdm(val_loader, desc=f"Epoch {epoch:3d} Val  ", leave=False):
                errs = reconstruction_error(X.to(device), model(X.to(device)))
                val_errors.extend(errs.cpu().numpy().tolist())

        val_errors = np.array(val_errors)
        val_loss   = float(val_errors.mean())
        val_std    = float(val_errors.std())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_err_mean"].append(val_loss)
        history["val_err_std"].append(val_std)

        print(f"Epoch {epoch:3d} | "
              f"train_loss={train_loss:.5f} | "
              f"val_err mean={val_loss:.5f}  std={val_std:.5f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.CHECKPOINT_PATH)
            print(f"  ✔ Checkpoint saved ({cfg.CHECKPOINT_PATH})")

        if stopper(val_loss):
            print(f"\nEarly stopping at epoch {epoch}.")
            break

        print()

    _plot_training_history(history)
    return model, history


def _plot_training_history(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train loss",      color="steelblue")
    axes[0].plot(history["val_loss"],   label="val loss (mean)", color="tomato")
    axes[0].set_title("Reconstruction Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    means = np.array(history["val_err_mean"])
    stds  = np.array(history["val_err_std"])
    axes[1].plot(means, color="steelblue", label="val error mean")
    axes[1].fill_between(range(len(means)),
                         means - stds, means + stds,
                         alpha=0.2, color="steelblue", label="±1 std")
    axes[1].set_title("Val Reconstruction Error (whale clips)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("autoencoder_history.png", dpi=150)
    print("Training history saved → autoencoder_history.png")


# ─────────────────────────────────────────────
# 5. THRESHOLD FROM VAL WHALE ERRORS
# ─────────────────────────────────────────────

def calibrate_threshold(cfg: Config) -> float:
    """
    Collect reconstruction errors on val/whale clips.
    Set threshold = Nth percentile of those errors.
    Any future clip with error > threshold is flagged as 'no whale'.
    """
    device = torch.device(cfg.DEVICE)
    model  = ConvAutoencoder(cfg).to(device)
    model.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location=device))
    model.eval()

    val_ds = WhaleOnlyDataset(
        os.path.join(cfg.DATA_DIR, "val", "whale"), cfg, augment=False
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, num_workers=4)

    val_errors = []
    with torch.no_grad():
        for X in tqdm(val_loader, desc="Computing val errors"):
            errs = reconstruction_error(X.to(device), model(X.to(device)))
            val_errors.extend(errs.cpu().numpy().tolist())

    val_errors = np.array(val_errors)
    threshold  = float(np.percentile(val_errors, cfg.THRESHOLD_PERCENTILE))

    print(f"\nVal whale reconstruction errors:")
    print(f"  mean  = {val_errors.mean():.6f}")
    print(f"  std   = {val_errors.std():.6f}")
    print(f"  min   = {val_errors.min():.6f}")
    print(f"  max   = {val_errors.max():.6f}")
    print(f"\nThreshold ({cfg.THRESHOLD_PERCENTILE}th percentile) = {threshold:.6f}")
    print(f"→ Clips with error > {threshold:.6f} will be flagged as NO WHALE\n")

    np.save(cfg.THRESHOLD_PATH, threshold)
    print(f"Threshold saved → {cfg.THRESHOLD_PATH}")

    _plot_val_error_distribution(val_errors, threshold, cfg.THRESHOLD_PERCENTILE)
    return threshold


def _plot_val_error_distribution(errors: np.ndarray, threshold: float, pct: int):
    plt.figure(figsize=(9, 4))
    plt.hist(errors, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
    plt.axvline(threshold, color="red", linestyle="--", linewidth=2,
                label=f"{pct}th percentile threshold = {threshold:.5f}")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Count")
    plt.title("Reconstruction Error Distribution — Val Whale Clips\n"
              "Clips to the RIGHT of the threshold will be flagged as anomalies")
    plt.legend()
    plt.tight_layout()
    plt.savefig("val_error_distribution.png", dpi=150)
    print("Val error distribution saved → val_error_distribution.png")


# ─────────────────────────────────────────────
# 6. TEST EVALUATION (whale-only)
# ─────────────────────────────────────────────

def test(cfg: Config):
    """
    Evaluate on test/whale clips only.
    Reports what fraction are correctly identified as whale
    (i.e. error <= threshold).
    """
    device    = torch.device(cfg.DEVICE)
    threshold = float(np.load(cfg.THRESHOLD_PATH))
    print(f"\nLoaded threshold : {threshold:.6f}")

    model = ConvAutoencoder(cfg).to(device)
    model.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location=device))
    model.eval()

    test_ds = WhaleOnlyDataset(
        os.path.join(cfg.DATA_DIR, "test", "whale"), cfg, augment=False
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE,
                             shuffle=False, num_workers=4)

    test_errors = []
    with torch.no_grad():
        for X in tqdm(test_loader, desc="Testing"):
            errs = reconstruction_error(X.to(device), model(X.to(device)))
            test_errors.extend(errs.cpu().numpy().tolist())

    test_errors  = np.array(test_errors)
    n_correct    = int((test_errors <= threshold).sum())
    n_total      = len(test_errors)
    whale_recall = n_correct / n_total

    print(f"\n── Test Results (whale clips only) ──")
    print(f"  Total clips          : {n_total}")
    print(f"  Correctly detected   : {n_correct}  ({whale_recall*100:.1f}%)")
    print(f"  Missed (false alarm) : {n_total - n_correct}")
    print(f"\n  Error mean  : {test_errors.mean():.6f}")
    print(f"  Error std   : {test_errors.std():.6f}")
    print(f"  Error min   : {test_errors.min():.6f}")
    print(f"  Error max   : {test_errors.max():.6f}")

    _plot_test_error_distribution(test_errors, threshold)


def _plot_test_error_distribution(errors: np.ndarray, threshold: float):
    plt.figure(figsize=(9, 4))
    plt.hist(errors[errors <= threshold], bins=30, color="steelblue",
             alpha=0.8, label="Detected as whale",        edgecolor="white")
    plt.hist(errors[errors >  threshold], bins=30, color="tomato",
             alpha=0.8, label="Missed (flagged anomaly)", edgecolor="white")
    plt.axvline(threshold, color="black", linestyle="--",
                label=f"threshold={threshold:.5f}")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Count")
    plt.title("Test Whale Clips — Reconstruction Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("test_error_distribution.png", dpi=150)
    print("\nTest error distribution saved → test_error_distribution.png")


# ─────────────────────────────────────────────
# 7. SINGLE FILE INFERENCE
# ─────────────────────────────────────────────

@torch.no_grad()
def predict_single(mp3_path: str, cfg: Config) -> dict:
    device    = torch.device(cfg.DEVICE)
    threshold = float(np.load(cfg.THRESHOLD_PATH))

    model = ConvAutoencoder(cfg).to(device)
    model.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location=device))
    model.eval()

    ds = WhaleOnlyDataset.__new__(WhaleOnlyDataset)
    ds.cfg            = cfg
    ds._target_len    = int(cfg.SAMPLE_RATE * cfg.DURATION)
    ds._target_frames = int(np.ceil(cfg.SAMPLE_RATE * cfg.DURATION / cfg.HOP_LENGTH))
    ds.augment        = False

    y    = ds._load_audio(mp3_path)
    spec = ds._to_melspec(y)
    X    = torch.tensor(spec[np.newaxis, np.newaxis], dtype=torch.float32).to(device)

    error = reconstruction_error(X, model(X)).item()

    return {
        "file"                : os.path.basename(mp3_path),
        "reconstruction_error": round(error, 6),
        "threshold"           : round(threshold, 6),
        "prediction"          : "whale" if error <= threshold else "no_whale",
    }


# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()

    print("=" * 55)
    print("  Whale Detection — One-Class Autoencoder")
    print("        (whale data only, all splits)")
    print("=" * 55, "\n")

    # Step 1: Train on train/whale only
    print("── Step 1: Training ──")
    train_autoencoder(cfg)

    # Step 2: Set threshold from val/whale error distribution
    print("\n── Step 2: Calibrating Threshold ──")
    calibrate_threshold(cfg)

    # Step 3: Evaluate on test/whale
    print("\n── Step 3: Test Evaluation ──")
    test(cfg)

    # Step 4: Inference on a new file
    # result = predict_single("dataset/test/whale/some_clip.mp3", cfg)
    # print(result)
