"""
prepare_database.py
-------------------
Reads raw EMNIST (.npz or binary) and Kaggle prescription word images,
applies preprocessing, and saves organised train/val/test splits to:
  data/processed/emnist/   -> (images.npy, labels.npy) per split
  data/processed/kaggle/   -> folder of resized PNGs + split CSVs

Usage:
    python prepare_database.py --emnist_dir data/emnist \
                               --kaggle_dir data/kaggle_prescriptions \
                               --output_dir data/processed
"""

import os
import argparse
import struct
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
EMNIST_IMG_SIZE = (28, 28)
KAGGLE_IMG_SIZE = (64, 256)   # (height, width) for word images fed to CRNN
VAL_RATIO       = 0.10
TEST_RATIO      = 0.10
RANDOM_SEED     = 42


# ── EMNIST helpers ────────────────────────────────────────────────────────────

def load_emnist_npz(npz_path: str):
    """Load EMNIST from a .npz file (tensorflow_datasets export or manual)."""
    log.info(f"Loading EMNIST npz from {npz_path}")
    data = np.load(npz_path)
    images = data["images"]   # shape (N, 28, 28) or (N, 784)
    labels = data["labels"]   # shape (N,)
    if images.ndim == 2:
        images = images.reshape(-1, 28, 28)
    return images, labels


def load_emnist_binary(images_path: str, labels_path: str):
    """
    Load EMNIST from original IDX binary files
    (same format as classic MNIST).
    """
    log.info(f"Loading EMNIST binary: {images_path}")
    with open(images_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)

    log.info(f"Loading EMNIST labels: {labels_path}")
    with open(labels_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    # EMNIST images are transposed relative to standard orientation – fix that
    images = np.transpose(images, (0, 2, 1))
    return images, labels


def preprocess_emnist(images: np.ndarray) -> np.ndarray:
    """Normalise to [0,1] float32."""
    return (images.astype(np.float32) / 255.0)


def save_emnist_splits(images, labels, output_dir: Path):
    """Split into train/val/test and save as .npy files."""
    idx = np.arange(len(images))
    idx_train, idx_tmp = train_test_split(idx, test_size=VAL_RATIO + TEST_RATIO,
                                          random_state=RANDOM_SEED, stratify=labels)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
                                         random_state=RANDOM_SEED, stratify=labels[idx_tmp])

    for split_name, split_idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        split_dir = output_dir / "emnist" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "images.npy", images[split_idx])
        np.save(split_dir / "labels.npy", labels[split_idx])
        log.info(f"  EMNIST {split_name}: {len(split_idx)} samples saved → {split_dir}")


# ── Kaggle prescription helpers ───────────────────────────────────────────────

def load_kaggle_csv(csv_path: str) -> pd.DataFrame:
    """
    Expects a CSV with at least two columns: 'filename' and 'word'.
    Adjust column names if your CSV differs.
    """
    df = pd.read_csv(csv_path)
    # Normalise common column name variants
    df.columns = [c.strip().lower() for c in df.columns]
    if "image" in df.columns and "filename" not in df.columns:
        df.rename(columns={"image": "filename"}, inplace=True)
    if "label" in df.columns and "word" not in df.columns:
        df.rename(columns={"label": "word"}, inplace=True)
    required = {"filename", "word"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kaggle CSV missing columns: {missing}. Found: {list(df.columns)}")
    df["word"] = df["word"].astype(str).str.strip()
    df = df[df["word"].str.len() > 0]
    log.info(f"Loaded Kaggle CSV: {len(df)} rows, {df['word'].nunique()} unique words")
    return df


def preprocess_kaggle_image(img_path: str, target_size=KAGGLE_IMG_SIZE) -> np.ndarray | None:
    """
    Read, convert to grayscale, resize with aspect-ratio padding,
    and normalise a single word image.
    Returns float32 array (H, W) or None if image cannot be loaded.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log.warning(f"Cannot read image: {img_path}")
        return None

    h, w = img.shape
    target_h, target_w = target_size

    # ── Otsu threshold to clean background ───────────────────────────────────
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Resize preserving aspect ratio, pad to target ─────────────────────────
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top    = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left   = (target_w - new_w) // 2
    pad_right  = target_w - new_w - pad_left
    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                              cv2.BORDER_CONSTANT, value=255)

    return img.astype(np.float32) / 255.0


def build_char_vocab(words: list[str]) -> dict:
    """Build character-to-index vocabulary from all words in dataset."""
    chars = sorted(set("".join(words)))
    vocab = {"<BLANK>": 0, "<UNK>": 1}
    for i, c in enumerate(chars, start=2):
        vocab[c] = i
    log.info(f"Vocabulary size: {len(vocab)} chars")
    return vocab


def save_kaggle_splits(df: pd.DataFrame, image_dir: Path, output_dir: Path, vocab: dict):
    """
    Pre-process all Kaggle images, save as numpy arrays,
    and write split CSVs + vocabulary JSON.
    """
    # ── Splits ────────────────────────────────────────────────────────────────
    train_df, tmp_df = train_test_split(df, test_size=VAL_RATIO + TEST_RATIO,
                                        random_state=RANDOM_SEED)
    val_df, test_df  = train_test_split(tmp_df, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
                                        random_state=RANDOM_SEED)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_dir = output_dir / "kaggle" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        images_list, labels_list, filenames_ok = [], [], []

        for _, row in tqdm(split_df.iterrows(), total=len(split_df),
                           desc=f"Kaggle {split_name}"):
            img_path = str(image_dir / row["filename"])
            img = preprocess_kaggle_image(img_path)
            if img is None:
                continue
            images_list.append(img)
            labels_list.append(row["word"])
            filenames_ok.append(row["filename"])

        images_arr = np.stack(images_list, axis=0)          # (N, H, W)
        np.save(split_dir / "images.npy", images_arr)

        out_df = pd.DataFrame({"filename": filenames_ok, "word": labels_list})
        out_df.to_csv(split_dir / "labels.csv", index=False)
        log.info(f"  Kaggle {split_name}: {len(images_arr)} samples → {split_dir}")

    # ── Save vocabulary ───────────────────────────────────────────────────────
    vocab_path = output_dir / "kaggle" / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    log.info(f"Vocabulary saved → {vocab_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare EMNIST + Kaggle datasets.")
    parser.add_argument("--emnist_dir",  default="data/emnist",
                        help="Folder containing EMNIST files (.npz or IDX binaries)")
    parser.add_argument("--kaggle_dir",  default="data/kaggle_prescriptions",
                        help="Folder containing Kaggle images + CSV label file")
    parser.add_argument("--kaggle_csv",  default="labels.csv",
                        help="CSV filename inside kaggle_dir")
    parser.add_argument("--output_dir",  default="data/processed",
                        help="Where to write processed splits")
    parser.add_argument("--emnist_npz",  default=None,
                        help="Path to a single EMNIST .npz file (overrides binary search)")
    args = parser.parse_args()

    emnist_dir  = Path(args.emnist_dir)
    kaggle_dir  = Path(args.kaggle_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── EMNIST ────────────────────────────────────────────────────────────────
    log.info("=== Processing EMNIST ===")
    if args.emnist_npz:
        images_e, labels_e = load_emnist_npz(args.emnist_npz)
    else:
        # Try to find IDX binaries auto-detected from common filenames
        img_candidates = list(emnist_dir.glob("*images*"))
        lbl_candidates = list(emnist_dir.glob("*labels*"))
        if not img_candidates or not lbl_candidates:
            log.error("No EMNIST files found. Provide --emnist_npz or place IDX files in emnist_dir.")
        else:
            images_e, labels_e = load_emnist_binary(str(img_candidates[0]),
                                                     str(lbl_candidates[0]))
            images_e = preprocess_emnist(images_e)
            save_emnist_splits(images_e, labels_e, output_dir)

    # ── Kaggle ────────────────────────────────────────────────────────────────
    log.info("=== Processing Kaggle prescriptions ===")
    csv_path = kaggle_dir / args.kaggle_csv
    if not csv_path.exists():
        log.error(f"Kaggle CSV not found at {csv_path}. Check --kaggle_csv argument.")
    else:
        df_kaggle = load_kaggle_csv(str(csv_path))
        vocab = build_char_vocab(df_kaggle["word"].tolist())
        save_kaggle_splits(df_kaggle, kaggle_dir, output_dir, vocab)

    log.info("=== Data preparation complete ===")


if __name__ == "__main__":
    main()