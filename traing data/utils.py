"""
utils.py
--------
Shared helper functions used across the project:
  • Image augmentation (rotation, shear, noise, elastic distortion)
  • CTC label encoding / decoding
  • CER / WER metric computation
  • Logging setup
  • Reproducibility seed setter
"""

import random
import numpy as np
import cv2
import os
import logging
from scipy.ndimage import map_coordinates, gaussian_filter


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ── Image Augmentation ────────────────────────────────────────────────────────

def random_rotation(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """Rotate image by a random angle in [-max_angle, max_angle] degrees."""
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)


def random_shear(image: np.ndarray, max_shear: float = 0.2) -> np.ndarray:
    """Apply a random horizontal shear to simulate slanted handwriting."""
    h, w = image.shape[:2]
    shear = random.uniform(-max_shear, max_shear)
    M = np.float32([[1, shear, 0], [0, 1, 0]])
    new_w = int(w + abs(shear) * h)
    return cv2.warpAffine(image, M, (new_w, h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)


def random_scale(image: np.ndarray, min_scale: float = 0.85,
                 max_scale: float = 1.15) -> np.ndarray:
    """Scale image by a random factor, then crop/pad back to original size."""
    h, w = image.shape[:2]
    scale = random.uniform(min_scale, max_scale)
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Crop to original size or pad with white
    out = np.ones((h, w), dtype=np.float32)
    crop_h, crop_w = min(new_h, h), min(new_w, w)
    out[:crop_h, :crop_w] = scaled[:crop_h, :crop_w]
    return out


def add_gaussian_noise(image: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """Add zero-mean Gaussian noise and clip to [0, 1]."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


def elastic_distortion(image: np.ndarray, alpha: float = 8.0,
                       sigma: float = 3.0) -> np.ndarray:
    """
    Apply elastic deformation (Simard et al.) to simulate natural
    handwriting variation.  Only operates on 2-D (H×W) float arrays.
    """
    h, w = image.shape[:2]
    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = (np.clip(y + dy, 0, h - 1).ravel(),
               np.clip(x + dx, 0, w - 1).ravel())
    return map_coordinates(image, indices, order=1).reshape(h, w)


def augment_image(image: np.ndarray,
                  rotate: bool = True,
                  shear: bool = True,
                  noise: bool = True,
                  elastic: bool = True) -> np.ndarray:
    """
    Apply a random combination of augmentations to a single float32 image.
    Expects input shape (H, W) with values in [0, 1].
    """
    if rotate and random.random() < 0.5:
        image = random_rotation(image)
    if shear and random.random() < 0.4:
        image = random_shear(image)
    if noise and random.random() < 0.4:
        image = add_gaussian_noise(image)
    if elastic and random.random() < 0.3:
        image = elastic_distortion(image)
    return image


# ── CTC helpers ───────────────────────────────────────────────────────────────

def encode_label(word: str, vocab: dict) -> list[int]:
    """Convert a word string to a list of integer indices using vocab."""
    unk = vocab.get("<UNK>", 1)
    return [vocab.get(c, unk) for c in word]


def decode_ctc_greedy(indices: np.ndarray, inv_vocab: dict,
                      blank_idx: int = 0) -> str:
    """
    Greedy CTC decode: collapse repeated chars, remove blanks.
    `indices` is a 1-D array of class indices (time-steps).
    """
    result = []
    prev = None
    for idx in indices:
        if idx != prev:
            if idx != blank_idx:
                result.append(inv_vocab.get(int(idx), ""))
            prev = idx
    return "".join(result)


# ── Metrics ───────────────────────────────────────────────────────────────────

def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def character_error_rate(predictions: list[str], targets: list[str]) -> float:
    """CER = total edit distance / total target characters."""
    total_dist = sum(edit_distance(p, t) for p, t in zip(predictions, targets))
    total_chars = sum(len(t) for t in targets)
    return total_dist / max(total_chars, 1)


def word_error_rate(predictions: list[str], targets: list[str]) -> float:
    """WER = number of incorrect word predictions / total words."""
    wrong = sum(1 for p, t in zip(predictions, targets) if p != t)
    return wrong / max(len(targets), 1)


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s – %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger