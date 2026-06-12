import librosa, librosa.display
import numpy as np, matplotlib.pyplot as plt, os, random

random.seed(None)   # different random picks every run

data_dir = "dataset"
fig, axes = plt.subplots(2, 3, figsize=(15, 6))

for row, cls in enumerate(["whale", "no_whale"]):
    folder = os.path.join(data_dir, "train", cls)
    files  = [f for f in os.listdir(folder) if f.endswith(".mp3")]

    # Pick 3 random files
    chosen = random.sample(files, min(3, len(files)))

    for col, fname in enumerate(chosen):
        y, sr = librosa.load(
            os.path.join(folder, fname), sr=22050, duration=8.0
        )
        S    = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, n_fft=1024,
            hop_length=512, fmin=0, fmax=1000
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        librosa.display.specshow(
            S_db, sr=sr, hop_length=512,
            fmin=0, fmax=1000,
            x_axis="time", y_axis="mel",
            ax=axes[row, col], cmap="magma"
        )
        axes[row, col].set_title(f"{cls} — {fname[:20]}")

plt.tight_layout()
plt.savefig("diagnostic_random.png", dpi=150)
print("Saved → diagnostic_random.png")