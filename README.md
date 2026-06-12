# North Atlantic Right Whale Presence Detector

---

## Species

This detector is tuned for **North Atlantic right whale** (*Eubalaena glacialis*) upcalls, which occur in the frequency range 50–500 Hz with a typical duration of 0.5–1 second. The spectrogram frequency range is set to 0–1000 Hz to capture this range with sufficient resolution.

To adapt for other species, change `F_MAX` in `Config` to cover the relevant frequency range and retrain.

---

## Quick Start — Raspberry Pi 5

### Step 1 — Clone the repository

```bash
git clone https://github.com/xylu20/Buoy-deployment-code.git
cd Buoy-deployment-code
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

Every prediction is printed to the terminal and saved to `whale_detections.log`.

For example:

```
2025-06-12 14:03:21  INFO   whale      error=0.00312  threshold=0.00412  conf=0.76  (843ms)  recording_001.mp3
2025-06-12 14:03:23  INFO   no_whale   error=0.01823  threshold=0.00412  conf=0.00  (821ms)  recording_002.mp3
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
