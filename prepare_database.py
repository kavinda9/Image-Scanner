"""
prepare_database.py - SIMPLIFIED FOR FLAT STRUCTURE
Handles flat folder structure: train_v1/, train_v2/, etc.
Processes in BATCHES to avoid memory errors.
"""

import os
import argparse
import struct
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EMNIST_IMG_SIZE = (28, 28)
KAGGLE_IMG_SIZE = (64, 256)
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42


def load_emnist_binary(images_path: str, labels_path: str):
    """Load EMNIST from IDX binary files."""
    log.info(f"Loading EMNIST binary: {images_path}")
    with open(images_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    with open(labels_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    images = np.transpose(images, (0, 2, 1))
    return images, labels


def preprocess_emnist(images: np.ndarray) -> np.ndarray:
    """Normalise to [0,1] float32."""
    return images.astype(np.float32) / 255.0


def save_emnist_splits(images, labels, output_dir: Path):
    """Split EMNIST into train/val/test."""
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
        log.info(f"  EMNIST {split_name}: {len(split_idx)} samples")


def load_kaggle_csv(csv_path: str) -> pd.DataFrame:
    """Load Kaggle CSV - handles v1 and v2 formats."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # v1: IMAGE, MEDICINE_NAME
    # v2: FILENAME, IDENTITY
    if "image" in df.columns and "medicine_name" in df.columns:
        df.rename(columns={"image": "filename", "medicine_name": "word"}, inplace=True)
    elif "filename" in df.columns and "identity" in df.columns:
        df.rename(columns={"identity": "word"}, inplace=True)
    
    df["word"] = df["word"].astype(str).str.strip()
    df = df[df["word"].str.len() > 0]
    return df


def preprocess_kaggle_image(img_path: str, target_size=KAGGLE_IMG_SIZE):
    """Preprocess single Kaggle image."""
    if not os.path.exists(img_path):
        return None
    
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        h, w = img.shape
        target_h, target_w = target_size
        
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                  cv2.BORDER_CONSTANT, value=255)
        
        return img.astype(np.float32) / 255.0
    except Exception as e:
        return None


def build_char_vocab(words: list) -> dict:
    """Build vocabulary from words."""
    chars = sorted(set("".join(words)))
    vocab = {"<BLANK>": 0, "<UNK>": 1}
    for i, c in enumerate(chars, start=2):
        vocab[c] = i
    return vocab


def save_kaggle_splits_batched(df: pd.DataFrame, output_dir: Path, vocab: dict):
    """
    Save Kaggle splits PROCESSING IN BATCHES to avoid memory errors.
    """
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        
        if split_df.empty:
            log.warning(f"No data for split: {split_name}")
            continue
        
        split_dir = output_dir / "kaggle" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        batch_size = 5000
        all_batch_files = []
        labels_list = []
        filenames_list = []
        
        # Process in batches
        for batch_idx in range(0, len(split_df), batch_size):
            batch_df = split_df.iloc[batch_idx:batch_idx+batch_size]
            batch_images = []
            
            for _, row in tqdm(batch_df.iterrows(), total=len(batch_df),
                              desc=f"Kaggle {split_name} batch {batch_idx//batch_size + 1}"):
                img = preprocess_kaggle_image(row["full_path"])
                if img is not None:
                    batch_images.append(img)
                    labels_list.append(row["word"])
                    filenames_list.append(row["filename"])
            
            if batch_images:
                images_arr = np.stack(batch_images, axis=0)
                batch_file = split_dir / f"images_batch_{batch_idx//batch_size}.npy"
                np.save(batch_file, images_arr)
                all_batch_files.append(batch_file)
                log.info(f"    Batch {batch_idx//batch_size}: {len(images_arr)} samples")
        
        # Merge all batches into one file
        if all_batch_files:
            all_images = [np.load(f) for f in all_batch_files]
            final_images = np.concatenate(all_images, axis=0)
            np.save(split_dir / "images.npy", final_images)
            
            # Delete temp batch files
            for f in all_batch_files:
                f.unlink()
            
            log.info(f"  Kaggle {split_name}: {len(final_images)} total samples")
        
        # Save labels
        out_df = pd.DataFrame({"filename": filenames_list, "word": labels_list})
        out_df.to_csv(split_dir / "labels.csv", index=False)
    
    # Save vocabulary
    vocab_path = output_dir / "kaggle" / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    log.info(f"Vocabulary saved → {vocab_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare EMNIST + Kaggle datasets.")
    parser.add_argument("--emnist_dir", default="data/emnist")
    parser.add_argument("--kaggle_dir", default="data/kaggle")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    emnist_dir = Path(args.emnist_dir)
    kaggle_dir = Path(args.kaggle_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── EMNIST ────────────────────────────────────────────────────────────────
    log.info("=== Processing EMNIST ===")
    img_files = list(emnist_dir.glob("*train-images*"))
    lbl_files = list(emnist_dir.glob("*train-labels*"))
    if img_files and lbl_files:
        images, labels = load_emnist_binary(str(img_files[0]), str(lbl_files[0]))
        images = preprocess_emnist(images)
        save_emnist_splits(images, labels, output_dir)
    else:
        log.warning("No EMNIST files found")

    # ── Kaggle ────────────────────────────────────────────────────────────────
    log.info("=== Processing Kaggle (batched) ===")
    
    all_dfs = []
    
    # Load all CSVs and match with image folders
    for csv_file in kaggle_dir.glob("written_name_*.csv"):
        df = load_kaggle_csv(str(csv_file))
        
        # Determine split type (train, validation, test) and version (v1, v2)
        if "train" in csv_file.name:
            split_type = "train"
        elif "validation" in csv_file.name:
            split_type = "val"
        elif "test" in csv_file.name:
            split_type = "test"
        else:
            continue
        
        version = "v1" if "v1" in csv_file.name else "v2"
        
        # Image folder: train_v1, train_v2, validation_v1, etc.
        if split_type == "val":
            img_folder = kaggle_dir / f"validation_{version}"
        else:
            img_folder = kaggle_dir / f"{split_type}_{version}"
        
        if not img_folder.exists():
            log.warning(f"Image folder not found: {img_folder}")
            continue
        
        # Build full paths
        df["full_path"] = df["filename"].apply(lambda x: str(img_folder / x))
        df["split"] = split_type
        
        # Filter to existing files
        df_valid = df[df["full_path"].apply(os.path.exists)]
        
        if len(df_valid) > 0:
            log.info(f"  {split_type} {version}: {len(df_valid)} valid images")
            all_dfs.append(df_valid)
        else:
            log.warning(f"  {split_type} {version}: NO valid images found")
    
    if not all_dfs:
        log.error("❌ No valid Kaggle images found in any split!")
        return
    
    df_kaggle = pd.concat(all_dfs, ignore_index=True)
    log.info(f"Total Kaggle samples: {len(df_kaggle)}")
    
    # Limit to 200k
    max_samples = 200000
    if len(df_kaggle) > max_samples:
        log.info(f"Limiting to {max_samples} samples (from {len(df_kaggle)})")
        df_kaggle = df_kaggle.head(max_samples)
    
    vocab = build_char_vocab(df_kaggle["word"].tolist())
    save_kaggle_splits_batched(df_kaggle, output_dir, vocab)
    
    log.info("✅ === Complete ===")


if __name__ == "__main__":
    main()