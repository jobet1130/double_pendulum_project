import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import imageio.v2 as imageio
import numpy as np
import platform
import subprocess
import os

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

DATA_PATH = Path("../data/processed/double_pendulum_processed.csv").resolve()
PLOT_DIR = Path("../plots").resolve()
SNAPSHOT_DIR = PLOT_DIR / "snapshots"
MONTAGE_PATH = PLOT_DIR / "montage.png"
VIDEO_PATH = PLOT_DIR / "snapshot_video.mp4"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

try:
    df = pd.read_csv(DATA_PATH)
    required_cols = {
        "time", "theta1", "theta2", "omega1", "omega2",
        "x1", "y1", "x2", "y2", "KE", "PE", "TME"
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")
    logging.info(f"✅ Loaded {len(df)} rows from {DATA_PATH}")
except Exception as e:
    logging.error(str(e))
    exit()

x1, y1 = df["x1"].values, df["y1"].values
x2, y2 = df["x2"].values, df["y2"].values
t = df["time"].values
KE, PE, TME = df["KE"].values, df["PE"].values, df["TME"].values

interval = 1000
frames = range(0, len(df), interval)
snapshot_paths = []

for frame in frames:
    if frame >= len(df):
        continue

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.set_title(f"Frame {frame}", fontsize=12)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True)

    this_x = [0, x1[frame], x2[frame]]
    this_y = [0, y1[frame], y2[frame]]
    ax.plot(this_x, this_y, "o-", lw=2, color="teal")
    ax.plot(x2[:frame], y2[:frame], "-", lw=1, color="plum", alpha=0.6)

    ax.text(
        0.05, 0.05,
        f"t = {t[frame]:.2f}s\nKE = {KE[frame]:.2f} J\nPE = {PE[frame]:.2f} J\nTME = {TME[frame]:.2f} J",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    snapshot_file = SNAPSHOT_DIR / f"snapshot_{frame:05d}.png"
    fig.savefig(snapshot_file, dpi=100)
    plt.close(fig)

    if snapshot_file.exists():
        snapshot_paths.append(snapshot_file)
        logging.info(f"🖼️ Saved snapshot: {snapshot_file.name}")
    else:
        logging.warning(f"❌ Snapshot not saved for frame {frame}")

if not snapshot_paths:
    logging.error("❌ No snapshots saved. Aborting montage and video creation.")
    exit()

montage_cols = 4
montage_rows = int(np.ceil(len(snapshot_paths) / montage_cols))
fig, axes = plt.subplots(montage_rows, montage_cols, figsize=(4 * montage_cols, 4 * montage_rows))

for ax in axes.flat:
    ax.axis("off")

for i, path in enumerate(snapshot_paths):
    img = plt.imread(path)
    row, col = divmod(i, montage_cols)
    axes[row, col].imshow(img)
    axes[row, col].set_title(path.stem, fontsize=10)

plt.tight_layout()
plt.savefig(MONTAGE_PATH, dpi=150)
plt.close()
logging.info(f"🧩 Saved montage: {MONTAGE_PATH}")

try:
    with imageio.get_writer(VIDEO_PATH, fps=5, codec="libx264", bitrate="800k", format='ffmpeg') as writer:
        for path in snapshot_paths:
            image = imageio.imread(path)
            writer.append_data(image)
    logging.info(f"🎬 Saved video: {VIDEO_PATH}")
except Exception as e:
    logging.error(f"❌ Failed to write video: {e}")

def open_file(path):
    try:
        if platform.system() == "Windows":
            os.startfile(str(path))
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(path)])
        else:
            subprocess.run(["xdg-open", str(path)])
    except Exception as e:
        logging.warning(f"Couldn't open {path}: {e}")

open_file(MONTAGE_PATH)
open_file(VIDEO_PATH)
