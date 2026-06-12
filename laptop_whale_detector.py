"""
Whale Presence Detector — Laptop
==================================
Inference-only script for running on your laptop after training.
Identical logic to the Pi version but with:
  - MPS support (Apple Silicon) and CUDA support
  - CPUAdaptiveAvgPool2d workaround kept for MPS
  - A visual report saved as a PNG after batch/watch runs

Prerequisites:
    pip install torch librosa numpy matplotlib

Files needed (produced by the training script):
    best_autoencoder.pt       ← trained model weights
    detection_threshold.npy   ← calibrated detection threshold

Usage:
    # Classify a single file:
    python3 laptop_whale_detector.py --file recording.mp3

    # Classify all MP3s in a folder:
    python3 laptop_whale_detector.py --batch /path/to/folder

    # Watch a folder for new clips:
    python3 laptop_whale_detector.py --watch /path/to/folder
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s  %(levelname)s  %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("whale_detections.log")
    ]
)
log = logging.getLogger("whale_detector")


# ─────────────────────────────────────────────
# CONFIGURATION
# Must match EXACTLY what was used during training.
# ─────────────────────────────────────────────

class Config:
    # Audio — must match training
    SAMPLE_RATE  = 22050
    DURATION     = 8.0
    N_MELS       = 64
    N_FFT        = 1024
    HOP_LENGTH   = 512
    F_MIN        = 0
    F_MAX        = 1000

    # Model architecture — must match training
    LATENT_DIM  = 64
    BASE_CH     = 32

    # Paths
    CHECKPOINT_PATH = "best_autoencoder.pt"
    THRESHOLD_PATH  = "detection_threshold.npy"

    # Device — auto-detected
    DEVICE = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )


# ─────────────────────────────────────────────
# MODEL
# Exact copy from training script.
# ─────────────────────────────────────────────

class CPUAdaptiveAvgPool2d(nn.Module):
    """MPS-safe AdaptiveAvgPool2d."""
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return self.pool(x.cpu()).to(x.device)


class ConvAutoencoder(nn.Module):
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
            CPUAdaptiveAvgPool2d((4, 4)),
        )

        self._ch            = ch
        self.bottleneck_in  = nn.Linear(ch*8 * 4 * 4, cfg.LATENT_DIM)
        self.bottleneck_out = nn.Linear(cfg.LATENT_DIM, ch*8 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ch*8, ch*4, 4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(ch*4), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch*4, ch*2, 4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(ch*2), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch*2, ch, 4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch, 1, 4, stride=2,
                               padding=1, bias=False),
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
        recon = F.interpolate(
            recon, size=x.shape[2:],
            mode="bilinear", align_corners=False
        )
        return recon


# ─────────────────────────────────────────────
# DETECTOR CLASS
# ─────────────────────────────────────────────

class WhaleDetector:
    """
    Loads the pretrained model once at startup.
    Call predict() for every new clip.
    """

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)

        # Validate required files
        for path in [cfg.CHECKPOINT_PATH, cfg.THRESHOLD_PATH]:
            if not os.path.exists(path):
                log.error(f"Required file not found: {path}")
                log.error(
                    "Run the training script first to produce "
                    "this file, then re-run inference."
                )
                sys.exit(1)

        # Load threshold
        self.threshold = float(np.load(cfg.THRESHOLD_PATH))
        log.info(f"Threshold          : {self.threshold:.6f}")

        # Load model
        self.model = ConvAutoencoder(cfg).to(self.device)
        self.model.load_state_dict(
            torch.load(cfg.CHECKPOINT_PATH,
                       map_location=self.device)
        )
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        log.info(f"Model loaded       : {n_params:,} parameters")
        log.info(f"Device             : {cfg.DEVICE.upper()}")

        # Pre-compute fixed lengths
        self._target_len    = int(cfg.SAMPLE_RATE * cfg.DURATION)
        self._target_frames = int(
            np.ceil(cfg.SAMPLE_RATE * cfg.DURATION /
                    cfg.HOP_LENGTH)
        )

    # ── Preprocessing ─────────────────────────────────────

    def _load_audio(self, path: str) -> np.ndarray:
        y, _ = librosa.load(
            path,
            sr       = self.cfg.SAMPLE_RATE,
            duration = self.cfg.DURATION,
            mono     = True
        )
        tl = self._target_len
        y  = np.pad(y, (0, max(0, tl - len(y))))[:tl]
        return y

    def _to_melspec(self, y: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        S   = librosa.feature.melspectrogram(
            y=y, sr=cfg.SAMPLE_RATE, n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LENGTH, n_mels=cfg.N_MELS,
            fmin=cfg.F_MIN, fmax=cfg.F_MAX
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db - S_db.min()) / \
               (S_db.max() - S_db.min() + 1e-8)
        T = self._target_frames
        if S_db.shape[1] < T:
            S_db = np.pad(S_db,
                          ((0, 0), (0, T - S_db.shape[1])))
        else:
            S_db = S_db[:, :T]
        return S_db.astype(np.float32)

    # ── Core prediction ───────────────────────────────────

    @torch.no_grad()
    def predict(self, mp3_path: str) -> dict:
        """
        Classify one MP3 clip.

        Returns:
            file         : filename
            error        : reconstruction error
            threshold    : decision boundary
            prediction   : "WHALE" or "no_whale"
            confidence   : proximity to threshold (0–1)
            time_ms      : processing time in milliseconds
            spectrogram  : the mel spectrogram as numpy array
                           (useful for plotting)
        """
        t0 = time.time()

        y    = self._load_audio(mp3_path)
        spec = self._to_melspec(y)
        X    = torch.tensor(
            spec[np.newaxis, np.newaxis],
            dtype=torch.float32
        ).to(self.device)

        recon = self.model(X)
        error = float(((X - recon) ** 2).mean().item())

        is_whale   = error <= self.threshold
        prediction = "WHALE" if is_whale else "no_whale"
        confidence = float(np.clip(
            1.0 - abs(error - self.threshold) /
            (self.threshold + 1e-8), 0.0, 1.0
        ))

        return {
            "file"       : os.path.basename(mp3_path),
            "error"      : round(error, 6),
            "threshold"  : round(self.threshold, 6),
            "prediction" : prediction,
            "confidence" : round(confidence, 3),
            "time_ms"    : round((time.time() - t0) * 1000, 1),
            "spectrogram": spec,   # (N_MELS, T) — for plotting
        }


# ─────────────────────────────────────────────
# VISUAL REPORT
# ─────────────────────────────────────────────

def save_visual_report(results: list, cfg: Config,
                        save_path: str = "detection_report.png"):
    """
    Save a visual summary of a batch of predictions:
      - Error distribution with threshold marked
      - Spectrogram grid for top whale detections
      - Summary stats
    """
    if not results:
        return

    errors      = [r["error"]      for r in results]
    predictions = [r["prediction"] for r in results]
    filenames   = [r["file"]       for r in results]
    n_whale     = sum(1 for p in predictions if p == "WHALE")
    n_total     = len(results)
    threshold   = results[0]["threshold"]

    # Separate whale and no_whale results
    whale_results = [r for r in results if r["prediction"] == "WHALE"]
    n_show        = min(6, len(whale_results))

    fig = plt.figure(figsize=(16, 10))

    # ── Top section: error distribution ──────────────────
    ax_dist = fig.add_axes([0.05, 0.55, 0.55, 0.38])
    colors  = ["steelblue" if p == "WHALE" else "tomato"
               for p in predictions]
    ax_dist.bar(range(n_total), errors, color=colors, alpha=0.7)
    ax_dist.axhline(threshold, color="black", linestyle="--",
                    lw=2, label=f"threshold = {threshold:.5f}")
    ax_dist.set_xlabel("Clip index")
    ax_dist.set_ylabel("Reconstruction error")
    ax_dist.set_title(
        f"Reconstruction Error per Clip\n"
        f"Blue = WHALE ({n_whale})  |  "
        f"Red = no_whale ({n_total - n_whale})  |  "
        f"Total = {n_total}"
    )
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)

    # ── Top right: summary stats ──────────────────────────
    ax_stats = fig.add_axes([0.65, 0.55, 0.30, 0.38])
    ax_stats.axis("off")
    stats_text = (
        f"SUMMARY\n"
        f"{'─'*30}\n"
        f"Total clips    :  {n_total}\n"
        f"Whale detected :  {n_whale}  "
        f"({100*n_whale/max(n_total,1):.1f}%)\n"
        f"No whale       :  {n_total - n_whale}  "
        f"({100*(n_total-n_whale)/max(n_total,1):.1f}%)\n\n"
        f"Threshold      :  {threshold:.6f}\n"
        f"Error mean     :  {np.mean(errors):.6f}\n"
        f"Error std      :  {np.std(errors):.6f}\n"
        f"Error min      :  {np.min(errors):.6f}\n"
        f"Error max      :  {np.max(errors):.6f}\n"
    )
    ax_stats.text(
        0.05, 0.95, stats_text,
        transform=ax_stats.transAxes,
        fontsize=10, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow",
                  alpha=0.8)
    )

    # ── Bottom section: top whale spectrograms ────────────
    freqs = librosa.mel_frequencies(
        n_mels=cfg.N_MELS, fmin=cfg.F_MIN, fmax=cfg.F_MAX
    )
    if n_show > 0:
        for i in range(n_show):
            r  = whale_results[i]
            ax = fig.add_axes([
                0.05 + i * (0.90 / n_show),
                0.05,
                0.85 / n_show,
                0.42
            ])
            ax.imshow(
                r["spectrogram"],
                aspect="auto", origin="lower",
                cmap="magma",
                extent=[0, cfg.DURATION, freqs[0], freqs[-1]]
            )
            ax.set_title(
                f"{r['file'][:18]}\n"
                f"error={r['error']:.4f}",
                fontsize=7
            )
            ax.set_xlabel("Time (s)", fontsize=7)
            if i == 0:
                ax.set_ylabel("Frequency (Hz)", fontsize=7)
            ax.tick_params(labelsize=6)

        fig.text(
            0.5, 0.48,
            f"Top {n_show} Whale Detections (spectrograms)",
            ha="center", fontsize=10, fontweight="bold"
        )
    else:
        fig.text(
            0.5, 0.25,
            "No whale detections in this batch.",
            ha="center", fontsize=12, color="gray"
        )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Visual report saved → {save_path}")


# ─────────────────────────────────────────────
# RUNNING MODES
# ─────────────────────────────────────────────

def run_single(detector: WhaleDetector, path: str):
    """Classify one file, print result, show spectrogram."""
    if not os.path.exists(path):
        log.error(f"File not found: {path}")
        return

    result = detector.predict(path)
    _print_result(result)

    # Show the spectrogram with the decision
    cfg   = detector.cfg
    freqs = librosa.mel_frequencies(
        n_mels=cfg.N_MELS, fmin=cfg.F_MIN, fmax=cfg.F_MAX
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].imshow(
        result["spectrogram"],
        aspect="auto", origin="lower",
        cmap="magma",
        extent=[0, cfg.DURATION, freqs[0], freqs[-1]]
    )
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title(
        f"{result['file']}\n"
        f"Prediction: {result['prediction']}  |  "
        f"Error: {result['error']:.5f}  |  "
        f"Threshold: {result['threshold']:.5f}"
    )

    # Error vs threshold bar
    axes[1].barh(
        ["reconstruction\nerror", "threshold"],
        [result["error"], result["threshold"]],
        color=["steelblue" if result["prediction"] == "WHALE"
               else "tomato", "black"],
        alpha=0.8
    )
    axes[1].set_xlabel("Value")
    axes[1].set_title(
        f"Decision\n"
        f"{'Error ≤ threshold → WHALE ✔' if result['prediction'] == 'WHALE' else 'Error > threshold → no whale ✗'}"
    )
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.suptitle(
        f"Whale Detector — Single File Analysis",
        fontsize=12
    )
    plt.tight_layout()
    out = f"result_{os.path.splitext(result['file'])[0]}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"Spectrogram saved → {out}")


def run_batch(detector: WhaleDetector, folder: str):
    """Classify all MP3s in a folder and save a visual report."""
    if not os.path.isdir(folder):
        log.error(f"Folder not found: {folder}")
        return

    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".mp3")
    ])

    if not files:
        log.warning(f"No MP3 files found in {folder}")
        return

    log.info(f"Processing {len(files)} files...")
    results = []
    for path in files:
        result = detector.predict(path)
        _print_result(result)
        results.append(result)

    n_whale = sum(1 for r in results if r["prediction"] == "WHALE")
    log.info(f"\nDone.  {n_whale}/{len(results)} clips = WHALE")

    save_visual_report(results, detector.cfg)


def run_watch(detector: WhaleDetector, folder: str,
              poll_interval: float = 2.0):
    """Watch a folder and classify new clips as they appear."""
    if not os.path.isdir(folder):
        log.error(f"Watch folder not found: {folder}")
        return

    log.info(f"Watching {folder}  (Ctrl+C to stop)")
    processed = set()
    results   = []

    for f in os.listdir(folder):
        if f.lower().endswith(".mp3"):
            processed.add(os.path.join(folder, f))
    log.info(f"Skipping {len(processed)} existing files.")

    try:
        while True:
            current   = set(
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(".mp3")
            )
            new_files = sorted(current - processed)

            for path in new_files:
                time.sleep(0.3)
                try:
                    result = detector.predict(path)
                    _print_result(result)
                    results.append(result)
                    processed.add(path)
                except Exception as e:
                    log.warning(f"Could not process {path}: {e}")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        log.info("Stopped.")
        if results:
            save_visual_report(results, detector.cfg,
                               "watch_session_report.png")


# ─────────────────────────────────────────────
# OUTPUT FORMATTING
# ─────────────────────────────────────────────

def _print_result(result: dict):
    icon = "🐋" if result["prediction"] == "WHALE" else "·"
    log.info(
        f"{icon}  {result['prediction']:10s}  "
        f"error={result['error']:.5f}  "
        f"threshold={result['threshold']:.5f}  "
        f"conf={result['confidence']:.2f}  "
        f"({result['time_ms']}ms)  "
        f"{result['file']}"
    )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Whale presence detector — Laptop"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",  type=str,
        help="Classify a single MP3 file"
    )
    group.add_argument(
        "--batch", type=str, metavar="FOLDER",
        help="Classify all MP3s in a folder"
    )
    group.add_argument(
        "--watch", type=str, metavar="FOLDER",
        help="Watch a folder and classify new clips as they arrive"
    )
    parser.add_argument(
        "--interval", type=float, default=2.0,
        help="Poll interval in seconds for --watch (default: 2.0)"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  Whale Presence Detector — Laptop")
    print("=" * 55)

    cfg      = Config()
    detector = WhaleDetector(cfg)
    print()

    if args.file:
        run_single(detector, args.file)
    elif args.batch:
        run_batch(detector, args.batch)
    elif args.watch:
        run_watch(detector, args.watch,
                  poll_interval=args.interval)


if __name__ == "__main__":
    main()
