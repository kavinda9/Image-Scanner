"""
compress_db.py
--------------
Converts processed .npy split files into a single HDF5 (.h5) archive
for faster I/O during training.

Output layout inside the HDF5:
  /emnist/train/images   float32 (N,28,28)
  /emnist/train/labels   int32   (N,)
  /emnist/val/...
  /emnist/test/...
  /kaggle/train/images   float32 (N,H,W)
  /kaggle/train/words    bytes   (N,)    ← UTF-8 encoded word strings
  /kaggle/val/...
  /kaggle/test/...

Usage:
    python compress_db.py --processed_dir data/processed \
                          --output data/prescription_ocr.h5
"""

import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPLITS = ["train", "val", "test"]


def compress_emnist(h5: h5py.File, processed_dir: Path):
    """Write EMNIST splits into /emnist/<split>/ groups."""
    log.info("Compressing EMNIST …")
    for split in SPLITS:
        split_dir = processed_dir / "emnist" / split
        img_path = split_dir / "images.npy"
        lbl_path = split_dir / "labels.npy"
        if not img_path.exists():
            log.warning(f"  Skipping EMNIST/{split} – files not found.")
            continue

        images = np.load(img_path)
        labels = np.load(lbl_path)

        grp = h5.require_group(f"emnist/{split}")
        grp.create_dataset("images", data=images,
                           compression="gzip", compression_opts=4)
        grp.create_dataset("labels", data=labels.astype(np.int32),
                           compression="gzip", compression_opts=4)
        log.info(f"  emnist/{split}: {len(images)} samples, "
                 f"images {images.shape}, labels {labels.shape}")


def compress_kaggle(h5: h5py.File, processed_dir: Path):
    """Write Kaggle splits into /kaggle/<split>/ groups."""
    log.info("Compressing Kaggle prescriptions …")
    for split in SPLITS:
        split_dir = processed_dir / "kaggle" / split
        img_path  = split_dir / "images.npy"
        csv_path  = split_dir / "labels.csv"
        if not img_path.exists():
            log.warning(f"  Skipping kaggle/{split} – files not found.")
            continue

        images = np.load(img_path)
        df     = pd.read_csv(csv_path)
        words  = df["word"].astype(str).values

        # Encode strings as fixed-length bytes for HDF5 compatibility
        dt      = h5py.special_dtype(vlen=str)
        grp     = h5.require_group(f"kaggle/{split}")
        grp.create_dataset("images", data=images,
                           compression="gzip", compression_opts=4)
        word_ds = grp.create_dataset("words", shape=(len(words),), dtype=dt)
        for i, w in enumerate(words):
            word_ds[i] = w

        log.info(f"  kaggle/{split}: {len(images)} samples, images {images.shape}")


def main():
    parser = argparse.ArgumentParser(description="Compress processed splits to HDF5.")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output",        default="data/prescription_ocr.h5")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_path      = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Writing HDF5 → {out_path}")
    with h5py.File(str(out_path), "w") as h5:
        compress_emnist(h5, processed_dir)
        compress_kaggle(h5, processed_dir)

    size_mb = out_path.stat().st_size / (1024 ** 2)
    log.info(f"Done. HDF5 size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()