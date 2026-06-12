"""
Whale Pattern Discovery — Unsupervised CNN (Convolutional Autoencoder)
=======================================================================
Goal: visually understand what patterns the CNN finds in whale spectrograms,
      without using any labels.

Pipeline:
  1. Detect events in whale/ clips → extract patches
  2. Train a small Convolutional Autoencoder on those patches
  3. After training, extract what each CNN filter learned to detect
  4. For each filter, find the real whale patches that activate it most
  5. Visualise those patches — they show what the filter "looks for"
  6. Repeat for no_whale/ patches and compare

Expected dataset structure:
    dataset/train/whale/      *.mp3
    dataset/train/no_whale/   *.mp3
"""

import os
import random
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

class Config:
    # Audio
    SAMPLE_RATE  = 22050
    DURATION     = 8.0
    N_MELS       = 64
    N_FFT        = 1024
    HOP_LENGTH   = 512
    F_MIN        = 0
    F_MAX        = 1000

    # Event detection
    EVENT_ENERGY_PERCENTILE   = 95
    EVENT_MIN_DURATION_FRAMES = 15
    EVENT_MERGE_GAP_FRAMES    = 20
    EVENT_PAD_FRAMES          = 8
    FLATNESS_PERCENTILE       = 40

    # Patch resizing — all patches resized to this before CNN
    PATCH_H = 32    # frequency bins
    PATCH_W = 32    # time frames (~0.37s at hop=256)

    # CNN Autoencoder
    BASE_CH    = 16   # base number of filters
    LATENT_DIM = 32   # bottleneck size

    # Training
    BATCH_SIZE = 32
    EPOCHS     = 30
    LR         = 1e-3
    DEVICE     = "cuda" if torch.cuda.is_available() else \
                 "mps"  if torch.backends.mps.is_available() else \
                 "cpu"

    # Visualisation
    N_FILTERS_SHOW = 8    # how many filters to visualise
    N_TOP_PATCHES  = 6    # top activating patches per filter

    # Paths
    DATA_DIR   = "dataset/train"
    OUTPUT_DIR = "pattern_outputs"
    MAX_FILES  = 300
    SEED       = 42


# ─────────────────────────────────────────────
# AUDIO & SPECTROGRAM
# ─────────────────────────────────────────────

def load_spectrogram(path: str, cfg: Config) -> np.ndarray:
    y, _ = librosa.load(path, sr=cfg.SAMPLE_RATE,
                        duration=cfg.DURATION, mono=True)
    tl   = int(cfg.SAMPLE_RATE * cfg.DURATION)
    y    = np.pad(y, (0, max(0, tl - len(y))))[:tl]
    S    = librosa.feature.melspectrogram(
        y=y, sr=cfg.SAMPLE_RATE, n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH, n_mels=cfg.N_MELS,
        fmin=cfg.F_MIN, fmax=cfg.F_MAX
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_db.astype(np.float32)


def load_all_specs(folder: str, cfg: Config,
                   max_files: int = None) -> list:
    files = sorted([f for f in os.listdir(folder)
                    if f.endswith(".mp3")])
    if max_files:
        files = random.sample(files, min(max_files, len(files)))
    specs = []
    for fname in tqdm(files,
                      desc=f"  Loading {os.path.basename(folder)}"):
        try:
            specs.append(load_spectrogram(
                os.path.join(folder, fname), cfg))
        except Exception:
            pass
    return specs


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def mel_bin_to_hz(bin_idx: int, cfg: Config) -> float:
    freqs = librosa.mel_frequencies(
        n_mels=cfg.N_MELS, fmin=cfg.F_MIN, fmax=cfg.F_MAX
    )
    return float(freqs[np.clip(bin_idx, 0, len(freqs) - 1)])


def frames_to_seconds(n: int, cfg: Config) -> float:
    return n * cfg.HOP_LENGTH / cfg.SAMPLE_RATE


def resize_patch(patch: np.ndarray,
                 th: int, tw: int) -> np.ndarray:
    zy = th / max(patch.shape[0], 1)
    zx = tw / max(patch.shape[1], 1)
    return zoom(patch, (zy, zx))


# ─────────────────────────────────────────────
# EVENT DETECTION
# ─────────────────────────────────────────────

def detect_events(spec: np.ndarray, cfg: Config) -> list:
    """
    Find narrowband energy bursts:
    active = high energy AND low spectral flatness (tonal).
    """
    T = spec.shape[1]

    energy   = spec.mean(axis=0)
    e_thresh = np.percentile(energy, cfg.EVENT_ENERGY_PERCENTILE)

    flatness = np.zeros(T)
    for t in range(T):
        frame       = spec[:, t] + 1e-8
        flatness[t] = (np.exp(np.mean(np.log(frame))) /
                       np.mean(frame))
    f_thresh = np.percentile(flatness, cfg.FLATNESS_PERCENTILE)

    active = (energy > e_thresh) & (flatness < f_thresh)

    raw = []
    in_ev, start = False, 0
    for t, is_active in enumerate(active):
        if is_active and not in_ev:
            start = t; in_ev = True
        elif not is_active and in_ev:
            in_ev = False
            if t - start >= cfg.EVENT_MIN_DURATION_FRAMES:
                raw.append([start, t])
    if in_ev and T - start >= cfg.EVENT_MIN_DURATION_FRAMES:
        raw.append([start, T - 1])

    if not raw:
        return []

    merged = [raw[0]]
    for s, e in raw[1:]:
        if s - merged[-1][1] <= cfg.EVENT_MERGE_GAP_FRAMES:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    events = []
    for s, e in merged:
        s_pad = max(0, s - cfg.EVENT_PAD_FRAMES)
        e_pad = min(T - 1, e + cfg.EVENT_PAD_FRAMES)
        ev    = spec[:, s_pad:e_pad]
        fp    = ev.mean(axis=1)
        thresh_f = fp.max() * 0.30
        active_f = np.where(fp > thresh_f)[0]
        f_lo = int(active_f.min()) if len(active_f) > 0 else 0
        f_hi = int(active_f.max()) if len(active_f) > 0 \
               else spec.shape[0] - 1
        events.append({
            "t_start"   : s_pad,   "t_end"     : e_pad,
            "freq_lo"   : f_lo,    "freq_hi"   : f_hi,
            "hz_lo"     : mel_bin_to_hz(f_lo, cfg),
            "hz_hi"     : mel_bin_to_hz(f_hi, cfg),
            "duration_s": frames_to_seconds(e_pad - s_pad, cfg),
            "energy"    : float(ev.mean()),
        })
    return events


def extract_event_patches(specs: list, cfg: Config,
                           label: str) -> tuple:
    """
    Detect events → crop patches → resize to PATCH_H × PATCH_W.
    Returns:
        resized_patches : np.ndarray (N, PATCH_H, PATCH_W)
        raw_patches     : list of original-size patches + metadata
    """
    raw = []
    for clip_idx, spec in enumerate(
        tqdm(specs, desc=f"  Events [{label}]")
    ):
        for ev in detect_events(spec, cfg):
            f_lo, f_hi = ev["freq_lo"], ev["freq_hi"]
            t_s,  t_e  = ev["t_start"], ev["t_end"]
            if t_e <= t_s or f_hi <= f_lo:
                continue
            patch = spec[f_lo:f_hi + 1, t_s:t_e + 1]
            if patch.max() < 0.05:
                continue
            raw.append({
                "patch"     : patch,
                "clip_idx"  : clip_idx,
                "hz_lo"     : ev["hz_lo"],
                "hz_hi"     : ev["hz_hi"],
                "duration_s": ev["duration_s"],
                "energy"    : ev["energy"],
            })

    if not raw:
        print(f"  [{label}] No patches found.")
        return np.array([]), []

    resized = np.stack([
        resize_patch(p["patch"], cfg.PATCH_H, cfg.PATCH_W)
        for p in raw
    ])   # (N, PATCH_H, PATCH_W)

    print(f"  [{label}] {len(raw)} patches  "
          f"shape {resized.shape}")
    return resized, raw


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class PatchDataset(Dataset):
    """Simple dataset of normalised spectrogram patches."""

    def __init__(self, patches: np.ndarray):
        # patches: (N, H, W) → (N, 1, H, W)
        self.X = torch.tensor(
            patches[:, np.newaxis], dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


# ─────────────────────────────────────────────
# CONVOLUTIONAL AUTOENCODER
# ─────────────────────────────────────────────

class CPUAdaptiveAvgPool2d(nn.Module):
    """MPS-safe AdaptiveAvgPool2d."""
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return self.pool(x.cpu()).to(x.device)


class ConvAutoencoder(nn.Module):
    """
    Small convolutional autoencoder trained on event patches.

    The encoder learns 2D filters that detect recurring
    spatial patterns in the patches (frequency × time shapes).

    After training:
      - First conv layer filters = what shapes the CNN detects
      - Filter responses on patches = which patches trigger each filter

    Input/Output: (B, 1, PATCH_H, PATCH_W)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        ch = cfg.BASE_CH

        # Encoder — 3 conv layers, each learning more abstract patterns
        self.enc1 = nn.Sequential(
            nn.Conv2d(1,    ch,   3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ch,   ch*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ch*2, ch*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Bottleneck — compress to LATENT_DIM
        spatial    = (cfg.PATCH_H // 4) * (cfg.PATCH_W // 4)
        self.flat_dim = ch * 4 * spatial
        self.fc_enc   = nn.Linear(self.flat_dim, cfg.LATENT_DIM)
        self.fc_dec   = nn.Linear(cfg.LATENT_DIM, self.flat_dim)
        self._ch       = ch
        self._spatial  = (cfg.PATCH_H // 4, cfg.PATCH_W // 4)

        # Decoder — mirror of encoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(ch*4, ch*2, 4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(ch*2),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ch*2, ch,   4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(ch, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = x.flatten(1)
        return self.fc_enc(x)

    def decode(self, z):
        ch = self._ch
        sh = self._spatial
        x  = self.fc_dec(z).view(-1, ch*4, sh[0], sh[1])
        x  = self.dec3(x)
        x  = self.dec2(x)
        x  = self.dec1(x)
        return x

    def forward(self, x):
        z     = self.encode(x)
        recon = self.decode(z)
        recon = F.interpolate(recon, size=x.shape[2:],
                              mode="bilinear",
                              align_corners=False)
        return recon

    def get_first_layer_filters(self) -> np.ndarray:
        """
        Return the learned filters from the first conv layer.
        Shape: (n_filters, 1, 3, 3) → squeezed to (n_filters, 3, 3)
        These are the most interpretable — each is a small 2D detector.
        """
        w = self.enc1[0].weight.detach().cpu().numpy()
        w = w[:, 0, :, :]   # (n_filters, 3, 3)
        return w

    def get_filter_responses(self, patches_tensor: torch.Tensor,
                              device) -> np.ndarray:
        """
        Pass all patches through enc1 only.
        Returns the mean activation of each filter per patch.
        Shape: (n_patches, n_filters)
        """
        self.eval()
        all_responses = []
        with torch.no_grad():
            bs = 64
            for i in range(0, len(patches_tensor), bs):
                batch = patches_tensor[i:i+bs].to(device)
                fmaps = self.enc1(batch)   # (B, n_filters, H, W)
                # Mean activation per filter per patch
                resp  = fmaps.mean(dim=[2, 3])  # (B, n_filters)
                all_responses.append(resp.cpu().numpy())
        return np.vstack(all_responses)   # (N, n_filters)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_autoencoder(patches: np.ndarray,
                       cfg: Config) -> ConvAutoencoder:
    """Train the convolutional autoencoder on event patches."""
    device  = torch.device(cfg.DEVICE)
    dataset = PatchDataset(patches)
    loader  = DataLoader(dataset, batch_size=cfg.BATCH_SIZE,
                         shuffle=True, num_workers=0,
                         pin_memory=False)

    model     = ConvAutoencoder(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=1e-5
    )

    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")
    print(f"  Training on {len(patches)} patches  "
          f"device={cfg.DEVICE}")

    losses = []
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for X in loader:
            X     = X.to(device)
            recon = model(X)
            loss  = F.mse_loss(recon, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(X)
        ep_loss /= len(patches)
        losses.append(ep_loss)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg.EPOCHS}  "
                  f"loss={ep_loss:.5f}")

    return model, losses


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_training_loss(losses: list, label: str,
                        save_path: str):
    plt.figure(figsize=(7, 3))
    plt.plot(losses, color="steelblue", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss (MSE)")
    plt.title(f"[{label}] Autoencoder Training Loss\n"
              f"Decreasing = CNN is learning to reconstruct patches")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_filter_gallery(model: ConvAutoencoder,
                         patches_np: np.ndarray,
                         raw_patches: list,
                         cfg: Config,
                         label: str,
                         save_path: str):
    """
    For each learned filter:
      Row 1: the filter itself (3×3 kernel — what shape it detects)
      Row 2-4: the top patches that activate this filter most strongly
               shown in their ORIGINAL frequency/time bounds

    This directly answers: "what did the CNN learn to look for?"
    """
    device = torch.device(cfg.DEVICE)
    model.eval()

    # Get filter kernels and responses
    filters   = model.get_first_layer_filters()  # (n_filters, 3, 3)
    X_tensor  = torch.tensor(
        patches_np[:, np.newaxis], dtype=torch.float32
    )
    responses = model.get_filter_responses(
        X_tensor, device
    )   # (N_patches, n_filters)

    n_show  = min(cfg.N_FILTERS_SHOW, filters.shape[0])
    n_top   = cfg.N_TOP_PATCHES

    # Rank filters by how much variance they explain
    # (high variance = filter responds differently to different patches
    #  = it detected something meaningful, not just constant noise)
    filter_variance = responses.var(axis=0)
    ranked_filters  = np.argsort(filter_variance)[::-1][:n_show]

    fig = plt.figure(figsize=(3 * (n_top + 1), 4 * n_show))
    gs  = gridspec.GridSpec(n_show, n_top + 1, figure=fig,
                             hspace=0.5, wspace=0.3)

    for row, fi in enumerate(ranked_filters):
        filt    = filters[fi]               # (3, 3) kernel
        resp_fi = responses[:, fi]          # (N,) response scores
        top_idx = np.argsort(resp_fi)[::-1][:n_top]

        # ── Col 0: the filter kernel ──────────────────────
        ax_k = fig.add_subplot(gs[row, 0])
        vmax = max(abs(filt.max()), abs(filt.min()))
        ax_k.imshow(filt, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax,
                    aspect="auto")
        ax_k.set_title(
            f"Filter {fi+1}\n"
            f"var={filter_variance[fi]:.3f}",
            fontsize=8
        )
        ax_k.set_xticks([]); ax_k.set_yticks([])
        ax_k.set_xlabel("← time\n(3 frames)", fontsize=6)
        ax_k.set_ylabel("freq →\n(3 bins)",   fontsize=6)

        # Add colourbar annotation
        plt.colorbar(
            ax_k.images[0], ax=ax_k,
            fraction=0.046, pad=0.04
        )

        # ── Cols 1..n_top: top activating patches ─────────
        for col, pi in enumerate(top_idx):
            meta = raw_patches[pi]
            ax   = fig.add_subplot(gs[row, col + 1])

            im = ax.imshow(
                meta["patch"],
                aspect="auto", origin="lower",
                cmap="magma",
                extent=[
                    0,            meta["duration_s"],
                    meta["hz_lo"],
                    max(meta["hz_hi"], meta["hz_lo"] + 1)
                ]
            )
            ax.set_title(
                f"Rank #{col+1}\n"
                f"{meta['hz_lo']:.0f}–{meta['hz_hi']:.0f}Hz\n"
                f"{meta['duration_s']:.2f}s",
                fontsize=7
            )
            ax.set_xlabel("Time (s)", fontsize=6)
            if col == 0:
                ax.set_ylabel("Hz", fontsize=6)
            ax.tick_params(labelsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"[{label}] CNN Filter Analysis\n"
        f"Left column: learned 3×3 filter kernel  |  "
        f"Right columns: real patches that activate each filter most\n"
        f"Filters ranked by response variance "
        f"(high variance = detects a consistent pattern)",
        fontsize=11, y=1.01
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_reconstruction_examples(model: ConvAutoencoder,
                                   patches_np: np.ndarray,
                                   raw_patches: list,
                                   cfg: Config,
                                   label: str,
                                   save_path: str):
    """
    Show original patches alongside their reconstructions.
    Good reconstruction = CNN understood the patch structure.
    Bad reconstruction = CNN couldn't learn that pattern.
    """
    device  = torch.device(cfg.DEVICE)
    model.eval()
    n_show  = min(8, len(patches_np))

    # Pick a mix of high and low energy patches
    energies = [p["energy"] for p in raw_patches]
    idx_high = np.argsort(energies)[::-1][:n_show//2]
    idx_low  = np.argsort(energies)[:n_show//2]
    show_idx = list(idx_high) + list(idx_low)

    fig, axes = plt.subplots(3, n_show, figsize=(2.5 * n_show, 8))

    with torch.no_grad():
        X_show = torch.tensor(
            patches_np[show_idx, np.newaxis],
            dtype=torch.float32
        ).to(device)
        recon = model(X_show).cpu().numpy()[:, 0]

    for col, pi in enumerate(show_idx):
        orig  = patches_np[pi]
        rec   = recon[col]
        diff  = np.abs(orig - rec)
        meta  = raw_patches[pi]

        ext = [0, meta["duration_s"],
               meta["hz_lo"],
               max(meta["hz_hi"], meta["hz_lo"] + 1)]

        axes[0, col].imshow(orig, aspect="auto",
                             origin="lower", cmap="magma",
                             extent=ext)
        axes[0, col].set_title(
            f"Original\n{meta['hz_lo']:.0f}–"
            f"{meta['hz_hi']:.0f}Hz", fontsize=7
        )

        axes[1, col].imshow(rec,  aspect="auto",
                             origin="lower", cmap="magma",
                             extent=ext)
        axes[1, col].set_title("Reconstructed", fontsize=7)

        axes[2, col].imshow(diff, aspect="auto",
                             origin="lower", cmap="hot",
                             extent=ext)
        mse = float(np.mean(diff**2))
        axes[2, col].set_title(
            f"Difference\nMSE={mse:.4f}", fontsize=7
        )

        for row in range(3):
            axes[row, col].tick_params(labelsize=6)
            if row == 2:
                axes[row, col].set_xlabel("Time (s)",
                                           fontsize=6)

    axes[0, 0].set_ylabel("Frequency (Hz)", fontsize=7)
    axes[1, 0].set_ylabel("Frequency (Hz)", fontsize=7)
    axes[2, 0].set_ylabel("Frequency (Hz)", fontsize=7)

    plt.suptitle(
        f"[{label}] Reconstruction Quality\n"
        "Top: original  |  Middle: reconstructed  |  "
        "Bottom: difference (bright = high error)\n"
        "Low MSE = CNN understood this patch type  |  "
        "High MSE = CNN found this pattern unusual",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_latent_space(model_w: ConvAutoencoder,
                       patches_w: np.ndarray,
                       model_nw: ConvAutoencoder,
                       patches_nw: np.ndarray,
                       cfg: Config,
                       save_path: str):
    """
    Project whale and no_whale patches into the shared latent space
    using UMAP (or PCA if UMAP not available).
    Shows whether whale and no_whale events cluster separately.
    """
    device = torch.device(cfg.DEVICE)

    def get_latents(model, patches):
        model.eval()
        latents = []
        with torch.no_grad():
            bs = 64
            X  = torch.tensor(patches[:, np.newaxis],
                               dtype=torch.float32)
            for i in range(0, len(X), bs):
                batch = X[i:i+bs].to(device)
                z     = model.encode(batch).cpu().numpy()
                latents.append(z)
        return np.vstack(latents)

    print("  Computing latent representations...")
    z_w  = get_latents(model_w,  patches_w)
    z_nw = get_latents(model_nw, patches_nw)

    # Use the whale autoencoder's encoder for both
    # so they are in the same latent space
    z_w2  = get_latents(model_w, patches_w)
    z_nw2 = get_latents(model_w, patches_nw)

    Z      = np.vstack([z_w2, z_nw2])
    labels = np.array(
        [1] * len(z_w2) + [0] * len(z_nw2)
    )

    # Dimensionality reduction
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=cfg.SEED)
        method  = "UMAP"
        print("  Running UMAP...")
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        method  = "PCA"
        print("  UMAP not found — using PCA instead. "
              "Install with: pip install umap-learn")

    Z_2d = reducer.fit_transform(Z)

    plt.figure(figsize=(8, 6))
    plt.scatter(Z_2d[labels == 0, 0],
                Z_2d[labels == 0, 1],
                c="tomato",    alpha=0.4, s=15,
                label=f"no_whale ({len(z_nw2)} patches)")
    plt.scatter(Z_2d[labels == 1, 0],
                Z_2d[labels == 1, 1],
                c="steelblue", alpha=0.4, s=15,
                label=f"whale ({len(z_w2)} patches)")
    plt.title(
        f"Latent Space ({method}) — Whale vs No-Whale Event Patches\n"
        "Separated clusters = CNN found distinct patterns\n"
        "Mixed clusters = patterns are similar between classes"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Whale Pattern Discovery — Unsupervised CNN")
    print("=" * 60)
    print(f"\nDevice: {cfg.DEVICE}")
    print(f"Output: {cfg.OUTPUT_DIR}/\n")

    # ── WHALE DATA ────────────────────────────────────────
    print("── WHALE ───────────────────────────────────────────────")
    whale_specs = load_all_specs(
        os.path.join(cfg.DATA_DIR, "whale"), cfg, cfg.MAX_FILES
    )
    w_patches_np, w_raw = extract_event_patches(
        whale_specs, cfg, "whale"
    )

    if len(w_patches_np) == 0:
        print("No whale patches found — adjust event detection.")
        exit()

    print("\n  Training CNN autoencoder on whale patches...")
    model_w, losses_w = train_autoencoder(w_patches_np, cfg)

    plot_training_loss(
        losses_w, "whale",
        os.path.join(cfg.OUTPUT_DIR, "whale_training_loss.png")
    )
    plot_filter_gallery(
        model_w, w_patches_np, w_raw, cfg, "whale",
        os.path.join(cfg.OUTPUT_DIR, "whale_filter_gallery.png")
    )
    plot_reconstruction_examples(
        model_w, w_patches_np, w_raw, cfg, "whale",
        os.path.join(cfg.OUTPUT_DIR,
                     "whale_reconstructions.png")
    )

    # ── NO-WHALE DATA ─────────────────────────────────────
    print("\n── NO-WHALE ─────────────────────────────────────────────")
    nw_specs = load_all_specs(
        os.path.join(cfg.DATA_DIR, "no_whale"), cfg, cfg.MAX_FILES
    )
    nw_patches_np, nw_raw = extract_event_patches(
        nw_specs, cfg, "no_whale"
    )

    if len(nw_patches_np) > 0:
        print("\n  Training CNN autoencoder on no_whale patches...")
        model_nw, losses_nw = train_autoencoder(
            nw_patches_np, cfg
        )
        plot_training_loss(
            losses_nw, "no_whale",
            os.path.join(cfg.OUTPUT_DIR,
                         "no_whale_training_loss.png")
        )
        plot_filter_gallery(
            model_nw, nw_patches_np, nw_raw, cfg, "no_whale",
            os.path.join(cfg.OUTPUT_DIR,
                         "no_whale_filter_gallery.png")
        )
        plot_reconstruction_examples(
            model_nw, nw_patches_np, nw_raw, cfg, "no_whale",
            os.path.join(cfg.OUTPUT_DIR,
                         "no_whale_reconstructions.png")
        )

        # ── LATENT SPACE COMPARISON ───────────────────────
        print("\n── LATENT SPACE COMPARISON ─────────────────────────")
        plot_latent_space(
            model_w, w_patches_np,
            model_nw, nw_patches_np,
            cfg,
            os.path.join(cfg.OUTPUT_DIR,
                         "latent_space_comparison.png")
        )

    print("\n" + "=" * 60)
    print("  Done. Output files:")
    print("=" * 60)
    for f in sorted(os.listdir(cfg.OUTPUT_DIR)):
        print(f"  {cfg.OUTPUT_DIR}/{f}")
