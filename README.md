# North Atlantic Right Whale Presence Detector

Detects the presence of North Atlantic right whale upcalls in 8-second audio clips using a one-class convolutional autoencoder. No labelled negative examples required — the model learns only from whale recordings and flags anything that does not reconstruct well as a potential anomaly.

---

## How It Works

```
Training (laptop, done once):
  Whale audio clips → Mel spectrogram → CNN Autoencoder → learns to reconstruct whale sounds
                                                        → saves best_autoencoder.pt
                                                        → saves detection_threshold.npy

Inference (laptop or Pi):
  New audio clip → Mel spectrogram → Autoencoder tries to reconstruct it
                                   → reconstruction error ≤ threshold → WHALE
                                   → reconstruction error >  threshold → no whale
```

The model never sees non-whale examples. It simply learns what whale calls look like and flags anything unfamiliar as anomalous.

---

## Repository Structure

```
whale-detector/
│
├── README.md                      ← this file
│
├── model/
│   ├── best_autoencoder.pt        ← pretrained model weights
│   └── detection_threshold.npy   ← calibrated detection threshold
│
├── src/
│   ├── train.py                   ← training script (run on laptop)
│   ├── laptop_whale_detector.py   ← inference script for laptop
│   └── pi_whale_detector.py       ← inference script for Raspberry Pi 5
│
└── dataset/                       ← not included in repo (too large)
    ├── train/
    │   └── whale/   *.mp3
    ├── val/
    │   └── whale/   *.mp3
    └── test/
        └── whale/   *.mp3
```

> **Note:** The `dataset/` folder is not included in the repository because the audio files are too large. The pretrained model in `model/` is ready to use without retraining.

---

## Quick Start — Raspberry Pi 5

If you just want to run the detector on your Pi without retraining, follow these steps only.

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/whale-detector.git
cd whale-detector
```

### Step 2 — Install dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install librosa numpy
```

> This may take 10–20 minutes on the Pi. PyTorch is a large package.

### Step 3 — Run the detector

**Classify a single MP3 file:**
```bash
python3 src/pi_whale_detector.py --file recording.mp3
```

**Classify all MP3s in a folder:**
```bash
python3 src/pi_whale_detector.py --batch /home/pi/recordings
```

**Watch a folder continuously — classifies new clips as they arrive:**
```bash
python3 src/pi_whale_detector.py --watch /home/pi/recordings
```

### Step 4 — Read the output

Every prediction is printed to the terminal and saved to `whale_detections.log`:

```
2025-06-12 14:03:21  INFO  🐋  WHALE      error=0.00312  threshold=0.00412  conf=0.76  (843ms)  clip_001.mp3
2025-06-12 14:03:23  INFO  ·   no_whale   error=0.01823  threshold=0.00412  conf=0.00  (821ms)  clip_002.mp3
```

| Field | Meaning |
|---|---|
| `error` | Reconstruction error — lower means more whale-like |
| `threshold` | Decision boundary set during training |
| `conf` | Confidence — how far the error is from the threshold |
| `time_ms` | Processing time per clip |

---

## Laptop Usage

### Inference (no retraining needed)

Install dependencies:
```bash
pip install torch librosa numpy matplotlib
```

Run inference:
```bash
# Single file — saves a spectrogram PNG alongside the result
python3 src/laptop_whale_detector.py --file recording.mp3

# Whole folder — saves detection_report.png
python3 src/laptop_whale_detector.py --batch /path/to/recordings

# Watch folder — saves report when you press Ctrl+C
python3 src/laptop_whale_detector.py --watch /path/to/recordings
```

---

## Retraining From Scratch

Only needed if you want to train on your own dataset.

### Requirements

- Python 3.10+
- A Mac, Linux, or Windows machine with at least 8GB RAM
- GPU optional but speeds up training significantly

### Step 1 — Install training dependencies

```bash
pip install torch torchaudio librosa numpy matplotlib scikit-learn tqdm
```

### Step 2 — Prepare your dataset

Organise your MP3 files into this exact structure:

```
dataset/
├── train/
│   └── whale/      ← 70% of your whale clips here
├── val/
│   └── whale/      ← 15% of your whale clips here
└── test/
    └── whale/      ← 15% of your whale clips here
```

Each clip must be:
- Format: MP3
- Duration: 8 seconds (longer clips will be truncated, shorter clips will be padded)
- Mono or stereo (stereo is automatically converted to mono)

### Step 3 — Configure the training (optional)

Open `src/train.py` and adjust the `Config` class if needed:

```python
class Config:
    F_MAX                = 8000   # set to 1000 Hz for right whale upcalls
    THRESHOLD_PERCENTILE = 95     # lower = stricter detection
    EPOCHS               = 50     # increase for more training
    BATCH_SIZE           = 16     # reduce if you run out of memory
```

### Step 4 — Train

```bash
python3 src/train.py
```

Training produces:
- `best_autoencoder.pt` — the best model weights
- `detection_threshold.npy` — the detection threshold
- `autoencoder_history.png` — training loss curves
- `val_error_distribution.png` — threshold calibration plot
- `test_error_distribution.png` — test set evaluation

Training takes approximately:
- **CPU only:** 2–4 hours for 800 clips, 50 epochs
- **GPU (CUDA/MPS):** 15–30 minutes

### Step 5 — Replace the pretrained model

Copy the new model files into the `model/` folder:

```bash
cp best_autoencoder.pt     model/
cp detection_threshold.npy model/
```

Then update `CHECKPOINT_PATH` and `THRESHOLD_PATH` in the inference scripts to point to `model/`:

```python
CHECKPOINT_PATH = "model/best_autoencoder.pt"
THRESHOLD_PATH  = "model/detection_threshold.npy"
```

---

## Hardware Requirements

### Raspberry Pi 5
- Model: Raspberry Pi 5 (4GB or 8GB RAM)
- OS: Raspberry Pi OS (64-bit recommended)
- Storage: 16GB+ SD card or USB SSD
- Audio input: USB audio interface + hydrophone

### Laptop (training)
- RAM: 8GB minimum, 16GB recommended
- OS: macOS, Linux, or Windows
- GPU: optional (NVIDIA CUDA or Apple MPS supported automatically)

---

## Audio Setup for Real-Time Detection on Pi

To capture live audio from a hydrophone:

```
Hydrophone
    ↓
Preamplifier   (hydrophone signals are very low voltage — amplification needed)
    ↓
USB audio interface   (e.g. Focusrite Scarlett Solo)
    ↓
Raspberry Pi 5 USB port
```

Install the recording dependency:
```bash
pip install sounddevice soundfile
```

Record 8-second clips continuously and drop them into the watch folder:

```python
import sounddevice as sd
import soundfile as sf
import time, os

WATCH_FOLDER = "/home/pi/recordings"
SAMPLE_RATE  = 22050
DURATION     = 8        # seconds per clip
os.makedirs(WATCH_FOLDER, exist_ok=True)

print("Recording... Press Ctrl+C to stop.")
while True:
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1,
                   dtype="float32")
    sd.wait()
    fname = os.path.join(
        WATCH_FOLDER,
        f"clip_{int(time.time())}.wav"
    )
    sf.write(fname, audio, SAMPLE_RATE)
```

Run this alongside the detector in watch mode:

```bash
# Terminal 1 — record
python3 record.py

# Terminal 2 — detect
python3 src/pi_whale_detector.py --watch /home/pi/recordings
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `No module named 'librosa'` | Run `pip install librosa` |
| `No module named 'torch'` | Run `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `Required file not found: best_autoencoder.pt` | Check the file is in the same folder as the script, or update `CHECKPOINT_PATH` in `Config` |
| Very slow on Pi | Normal — Pi 5 processes ~1 clip/second on CPU |
| All clips predicted as `no_whale` | Threshold may be set too low — retrain and adjust `THRESHOLD_PERCENTILE` in `Config` |
| All clips predicted as `WHALE` | Threshold may be set too high — lower `THRESHOLD_PERCENTILE` |

---

## Detection Threshold

The threshold is set during training at the **95th percentile** of reconstruction errors on validation whale clips:

```
95% of known whale clips have error ≤ threshold  →  detected correctly
 5% of known whale clips have error >  threshold  →  missed
```

To make detection more or less sensitive, change `THRESHOLD_PERCENTILE` in `Config` before retraining:

```python
THRESHOLD_PERCENTILE = 99   # lenient  — misses fewer whales, more false alarms
THRESHOLD_PERCENTILE = 95   # balanced — default
THRESHOLD_PERCENTILE = 90   # strict   — fewer false alarms, misses more whales
```

---

## Species

This detector is tuned for **North Atlantic right whale** (*Eubalaena glacialis*) upcalls, which occur in the frequency range 50–500 Hz with a typical duration of 0.5–1 second. The spectrogram frequency range is set to 0–1000 Hz to capture this range with sufficient resolution.

To adapt for other species, change `F_MAX` in `Config` to cover the relevant frequency range and retrain.

---

## Acknowledgements

Built with [PyTorch](https://pytorch.org) and [librosa](https://librosa.org).
