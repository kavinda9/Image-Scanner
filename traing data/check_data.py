"""
check_data.py
-------------
Sanity-checks processed datasets and optionally visualises random samples.

Checks performed:
  • Image / label count match
  • No NaN / Inf in image arrays
  • Pixel value range [0, 1]
  • Class distribution (EMNIST)
  • Word length distribution (Kaggle)
  • Duplicate filename detection (Kaggle)

Usage:
    python check_data.py --processed_dir data/processed --visualise
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPLITS = ["train", "val", "test"]


# ── EMNIST checks ─────────────────────────────────────────────────────────────

def check_emnist(processed_dir: Path, visualise: bool = False):
    log.info("── Checking EMNIST ──────────────────────────────")
    all_ok = True

    char_map_path = Path("character_db.json")
    char_map = {}
    if char_map_path.exists():
        with open(char_map_path) as f:
            db = json.load(f)
        char_map = {int(k): v for k, v in db.get("emnist_balanced", {}).items()}

    for split in SPLITS:
        split_dir = processed_dir / "emnist" / split
        img_path  = split_dir / "images.npy"
        lbl_path  = split_dir / "labels.npy"

        if not img_path.exists():
            log.warning(f"  [{split}] MISSING – skipping")
            continue

        images = np.load(img_path)
        labels = np.load(lbl_path)

        n_img, n_lbl = len(images), len(labels)
        count_ok  = n_img == n_lbl
        range_ok  = float(images.min()) >= 0.0 and float(images.max()) <= 1.0
        finite_ok = bool(np.isfinite(images).all())

        log.info(f"  [{split}] samples: {n_img} | count_match: {count_ok} "
                 f"| range[0,1]: {range_ok} | finite: {finite_ok}")

        if not (count_ok and range_ok and finite_ok):
            all_ok = False

        # Class distribution
        unique, counts = np.unique(labels, return_counts=True)
        log.info(f"    classes: {len(unique)} | min_per_class: {counts.min()} "
                 f"| max_per_class: {counts.max()}")

        if visualise and split == "train":
            _plot_emnist_samples(images, labels, char_map, split)
            _plot_class_distribution(labels, char_map, split)

    return all_ok


def _plot_emnist_samples(images, labels, char_map, split, n=20):
    idx = np.random.choice(len(images), n, replace=False)
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 1.2, 5))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(images[idx[i]], cmap="gray", vmin=0, vmax=1)
        lbl = char_map.get(int(labels[idx[i]]), str(labels[idx[i]]))
        ax.set_title(lbl, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"EMNIST {split} – random samples", fontsize=12)
    plt.tight_layout()
    out = Path("outputs") / f"emnist_{split}_samples.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=100)
    log.info(f"    Saved sample grid → {out}")
    plt.close()


def _plot_class_distribution(labels, char_map, split):
    unique, counts = np.unique(labels, return_counts=True)
    xticks = [char_map.get(int(u), str(u)) for u in unique]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(xticks, counts, color="steelblue")
    ax.set_title(f"EMNIST {split} – class distribution")
    ax.set_xlabel("Character")
    ax.set_ylabel("Count")
    plt.tight_layout()
    out = Path("outputs") / f"emnist_{split}_class_dist.png"
    plt.savefig(out, dpi=100)
    log.info(f"    Saved class distribution → {out}")
    plt.close()


# ── Kaggle checks ─────────────────────────────────────────────────────────────

def check_kaggle(processed_dir: Path, visualise: bool = False):
    log.info("── Checking Kaggle prescriptions ────────────────")
    all_ok = True

    for split in SPLITS:
        split_dir = processed_dir / "kaggle" / split
        img_path  = split_dir / "images.npy"
        csv_path  = split_dir / "labels.csv"

        if not img_path.exists():
            log.warning(f"  [{split}] MISSING – skipping")
            continue

        images = np.load(img_path)
        df     = pd.read_csv(csv_path)

        n_img, n_lbl  = len(images), len(df)
        count_ok      = n_img == n_lbl
        range_ok      = float(images.min()) >= 0.0 and float(images.max()) <= 1.0
        finite_ok     = bool(np.isfinite(images).all())
        dup_count     = int(df["filename"].duplicated().sum())

        log.info(f"  [{split}] samples: {n_img} | count_match: {count_ok} "
                 f"| range[0,1]: {range_ok} | finite: {finite_ok} "
                 f"| duplicate_filenames: {dup_count}")

        if not (count_ok and range_ok and finite_ok) or dup_count > 0:
            all_ok = False

        word_lengths = df["word"].str.len()
        log.info(f"    word_len  min:{word_lengths.min()} max:{word_lengths.max()} "
                 f"mean:{word_lengths.mean():.1f}")

        if visualise and split == "train":
            _plot_kaggle_samples(images, df["word"].values, split)
            _plot_word_length_dist(word_lengths, split)

    return all_ok


def _plot_kaggle_samples(images, words, split, n=12):
    idx = np.random.choice(len(images), min(n, len(images)), replace=False)
    cols = 4
    rows = (len(idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < len(idx):
            ax.imshow(images[idx[i]], cmap="gray", vmin=0, vmax=1, aspect="auto")
            ax.set_title(words[idx[i]], fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Kaggle {split} – random samples", fontsize=12)
    plt.tight_layout()
    out = Path("outputs") / f"kaggle_{split}_samples.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=100)
    log.info(f"    Saved sample grid → {out}")
    plt.close()


def _plot_word_length_dist(word_lengths, split):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(word_lengths, bins=range(1, word_lengths.max() + 2),
                 ax=ax, color="teal", edgecolor="white")
    ax.set_title(f"Kaggle {split} – word length distribution")
    ax.set_xlabel("Word length (chars)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    out = Path("outputs") / f"kaggle_{split}_word_length.png"
    plt.savefig(out, dpi=100)
    log.info(f"    Saved word-length distribution → {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sanity-check processed datasets.")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--visualise", action="store_true",
                        help="Save sample grids and distribution plots to outputs/")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)

    e_ok = check_emnist(processed_dir, args.visualise)
    k_ok = check_kaggle(processed_dir, args.visualise)

    if e_ok and k_ok:
        log.info("✅  All checks passed.")
    else:
        log.warning("⚠️   Some checks failed – review warnings above.")


if __name__ == "__main__":
    main()