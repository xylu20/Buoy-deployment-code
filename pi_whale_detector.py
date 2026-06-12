"""
Whale Presence Detector — Raspberry Pi 5
=========================================
Inference-only script. Loads a pretrained model and threshold
produced by the training script, then monitors a folder for
new MP3 clips and classifies each one.

Prerequisites (run once on the Pi):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install librosa numpy

Files needed on the Pi (copy from your laptop after training):
    best_autoencoder.pt       ← trained model weights
    detection_threshold.npy   ← calibrated detection threshold

Usage:
    # Watch a folder for new clips:
    python3 pi_whale_detector.py --watch /home/pi/recordings

    # Classify a single file:
    python3 pi_whale_detector.py --file recording.mp3

    # Classify all MP3s in a folder (no watching):
    python3 pi_whale_detector.py --batch /home/pi/recordings
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
        logging.StreamHandler(sys.stdout),          # print to terminal
        logging.FileHandler("whale_detections.log") # save to log file
    ]
)
log = logging.getLogger("whale_detector")


# ─────────────────────────────────────────────
# CONFIGURATION
# Must match EXACTLY what was used during training.
# Do not change these values unless you retrain the model.
# ─────────────────────────────────────────────

class Config:
    # Audio — must match training
    SAMPLE_RATE = 22050
    DURATION    = 8.0
    N_MELS      = 128
    N_FFT       = 2048
    HOP_LENGTH  = 256
    F_MIN       = 0
    F_MAX       = 1000

    # Model architecture — must match training
    LATENT_DIM  = 64
    BASE_CH     = 32

    # Paths — files copied from laptop after training
    CHECKPOINT_PATH = "best_autoencoder.pt"
    THRESHOLD_PATH  = "detection_threshold.npy"

    # Pi 5 always uses CPU
    DEVICE = "cpu"


# ─────────────────────────────────────────────
# MODEL
# Exact copy from training script — must not change.
# ─────────────────────────────────────────────

class CPUAdaptiveAvgPool2d(nn.Module):
    """Standard AdaptiveAvgPool2d — no MPS workaround needed on Pi."""
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return self.pool(x)


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
# Loads model once at startup, reuses for every clip.
# ─────────────────────────────────────────────

class WhaleDetector:
    """
    Wraps the pretrained autoencoder for fast repeated inference.
    Load once, call predict() many times.
    """

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)

        # ── Validate required files exist ─────────────────
        for path in [cfg.CHECKPOINT_PATH, cfg.THRESHOLD_PATH]:
            if not os.path.exists(path):
                log.error(f"Required file not found: {path}")
                log.error("Copy this file from your laptop "
                          "after training.")
                sys.exit(1)

        # ── Load threshold ────────────────────────────────
        self.threshold = float(np.load(cfg.THRESHOLD_PATH))
        log.info(f"Threshold loaded: {self.threshold:.6f}")

        # ── Load model ────────────────────────────────────
        self.model = ConvAutoencoder(cfg).to(self.device)
        self.model.load_state_dict(
            torch.load(cfg.CHECKPOINT_PATH,
                       map_location=self.device)
        )
        self.model.eval()

        # Count parameters
        n_params = sum(
            p.numel() for p in self.model.parameters()
        )
        log.info(f"Model loaded  ({n_params:,} parameters)")
        log.info(f"Running on: {cfg.DEVICE.upper()}")

        # Pre-compute fixed lengths
        self._target_len    = int(cfg.SAMPLE_RATE * cfg.DURATION)
        self._target_frames = int(
            np.ceil(cfg.SAMPLE_RATE * cfg.DURATION /
                    cfg.HOP_LENGTH)
        )

    # ── Audio preprocessing ───────────────────────────────

    def _load_audio(self, path: str) -> np.ndarray:
        """Load MP3 → fixed-length waveform."""
        y, _ = librosa.load(
            path,
            sr       = self.cfg.SAMPLE_RATE,
            duration = self.cfg.DURATION,
            mono     = True
        )
        tl = self._target_len
        if len(y) < tl:
            y = np.pad(y, (0, tl - len(y)))
        else:
            y = y[:tl]
        return y

    def _to_melspec(self, y: np.ndarray) -> np.ndarray:
        """Waveform → normalised log-mel spectrogram."""
        cfg = self.cfg
        S   = librosa.feature.melspectrogram(
            y          = y,
            sr         = cfg.SAMPLE_RATE,
            n_fft      = cfg.N_FFT,
            hop_length = cfg.HOP_LENGTH,
            n_mels     = cfg.N_MELS,
            fmin       = cfg.F_MIN,
            fmax       = cfg.F_MAX
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = ((S_db - S_db.min()) /
                (S_db.max() - S_db.min() + 1e-8))

        T = self._target_frames
        if S_db.shape[1] < T:
            S_db = np.pad(S_db, ((0, 0), (0, T - S_db.shape[1])))
        else:
            S_db = S_db[:, :T]
        return S_db.astype(np.float32)

    # ── Core prediction ───────────────────────────────────

    @torch.no_grad()
    def predict(self, mp3_path: str) -> dict:
        """
        Classify a single MP3 clip.

        Returns:
            file         : filename
            error        : reconstruction error (lower = more whale-like)
            threshold    : decision boundary
            prediction   : "WHALE" or "no_whale"
            confidence   : how far from the threshold (0–1 scale)
            time_ms      : processing time in milliseconds
        """
        t_start = time.time()

        # Preprocess
        y    = self._load_audio(mp3_path)
        spec = self._to_melspec(y)
        X    = torch.tensor(
            spec[np.newaxis, np.newaxis],
            dtype=torch.float32
        ).to(self.device)

        # Forward pass
        recon = self.model(X)
        error = float(((X - recon) ** 2).mean().item())

        # Decision
        is_whale   = error <= self.threshold
        prediction = "WHALE" if is_whale else "no_whale"

        # Confidence: how far is the error from the threshold?
        # 1.0 = right at threshold, 0.0 = very far from threshold
        confidence = 1.0 - abs(error - self.threshold) / \
                     (self.threshold + 1e-8)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        elapsed_ms = (time.time() - t_start) * 1000

        return {
            "file"      : os.path.basename(mp3_path),
            "error"     : round(error, 6),
            "threshold" : round(self.threshold, 6),
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "time_ms"   : round(elapsed_ms, 1),
        }


# ─────────────────────────────────────────────
# RUNNING MODES
# ─────────────────────────────────────────────

def run_single(detector: WhaleDetector, path: str):
    """Classify one file and print the result."""
    if not os.path.exists(path):
        log.error(f"File not found: {path}")
        return

    result = detector.predict(path)
    _print_result(result)


def run_batch(detector: WhaleDetector, folder: str):
    """Classify all MP3s in a folder."""
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

    log.info(f"Processing {len(files)} files in {folder}")
    n_whale = 0

    for path in files:
        result  = detector.predict(path)
        _print_result(result)
        if result["prediction"] == "WHALE":
            n_whale += 1

    log.info(f"Done. {n_whale}/{len(files)} clips detected as whale.")


def run_watch(detector: WhaleDetector, folder: str,
              poll_interval: float = 2.0):
    """
    Watch a folder for new MP3 files.
    Classify each new file as it appears.
    Useful when another process is continuously saving recordings.

    poll_interval: how often to check for new files (seconds)
    """
    if not os.path.isdir(folder):
        log.error(f"Watch folder not found: {folder}")
        log.error(f"Create it with: mkdir -p {folder}")
        return

    log.info(f"Watching {folder} for new MP3 clips...")
    log.info(f"Press Ctrl+C to stop.")
    processed = set()

    # Mark existing files as already processed
    for f in os.listdir(folder):
        if f.lower().endswith(".mp3"):
            processed.add(os.path.join(folder, f))

    log.info(f"Skipping {len(processed)} existing files.")

    try:
        while True:
            current = set(
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(".mp3")
            )
            new_files = sorted(current - processed)

            for path in new_files:
                # Wait briefly to ensure file is fully written
                time.sleep(0.3)
                try:
                    result = detector.predict(path)
                    _print_result(result)
                    processed.add(path)
                except Exception as e:
                    log.warning(f"Could not process {path}: {e}")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        log.info("Stopped.")


# ─────────────────────────────────────────────
# OUTPUT FORMATTING
# ─────────────────────────────────────────────

def _print_result(result: dict):
    """Print and log a prediction result."""
    icon = "🐋" if result["prediction"] == "WHALE" else "·"
    msg  = (
        f"{icon}  {result['prediction']:10s}  "
        f"error={result['error']:.5f}  "
        f"threshold={result['threshold']:.5f}  "
        f"conf={result['confidence']:.2f}  "
        f"({result['time_ms']}ms)  "
        f"{result['file']}"
    )
    if result["prediction"] == "WHALE":
        log.info(msg)
    else:
        log.info(msg)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Whale presence detector — Raspberry Pi 5"
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
        help="Poll interval in seconds for --watch mode (default: 2.0)"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  Whale Presence Detector — Raspberry Pi 5")
    print("=" * 55)

    # Load model once
    cfg      = Config()
    detector = WhaleDetector(cfg)
    print()

    # Run selected mode
    if args.file:
        run_single(detector, args.file)

    elif args.batch:
        run_batch(detector, args.batch)

    elif args.watch:
        run_watch(detector, args.watch,
                  poll_interval=args.interval)


if __name__ == "__main__":
    main()
