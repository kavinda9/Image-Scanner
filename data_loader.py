"""
data_loader.py
--------------
TensorFlow data pipelines for efficient training and inference.

Classes:
  • EMNISTDataset  – character dataset from EMNIST splits
  • KaggleDataset  – word dataset from Kaggle prescription images
  • HDF5Dataset    – loads from compressed HDF5 archive
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
from pathlib import Path
import logging
from utils import augment_image, encode_label

log = logging.getLogger(__name__)


class EMNISTDataset:
    """
    TensorFlow dataset for EMNIST character recognition.
    
    Loads .npy files, applies augmentation, batches with padding.
    """
    
    def __init__(self, data_dir: str, vocab: dict, batch_size: int = 32,
                 augment: bool = False, shuffle: bool = True):  # Changed True → False
        """
        Args:
            data_dir: path to split folder (train/val/test)
            vocab: character-to-index dictionary
            batch_size: batch size
            augment: apply augmentation if True
            shuffle: shuffle samples if True
        """
        self.data_dir = Path(data_dir)
        self.vocab = vocab
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        
        self.inv_vocab = {v: k for k, v in vocab.items()}
        
        # Load arrays
        self.images = np.load(self.data_dir / "images.npy")  # (N, 28, 28)
        self.labels = np.load(self.data_dir / "labels.npy")  # (N,)
        
        self.n_samples = len(self.images)
        log.info(f"Loaded {self.n_samples} EMNIST samples from {data_dir}")
    
    def __call__(self):
        """Generate batches."""
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for batch_start in range(0, self.n_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.n_samples)
            batch_idx = indices[batch_start:batch_end]
            
            batch_images = self.images[batch_idx]  # (B, 28, 28)
            batch_labels = self.labels[batch_idx]  # (B,)
            
            # Augment
            if self.augment:
                batch_images = np.array([augment_image(img) for img in batch_images])
            
            # Expand channel dimension: (B, 28, 28) -> (B, 28, 28, 1)
            batch_images = np.expand_dims(batch_images, axis=-1)
            
            yield batch_images.astype(np.float32), batch_labels.astype(np.int32)
    
    def get_tf_dataset(self):
        """Return a tf.data.Dataset."""
        output_signature = (
            tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
        return tf.data.Dataset.from_generator(
            self, output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)


class KaggleDataset:
    """
    TensorFlow dataset for word-level recognition (Kaggle prescription images).
    
    Handles variable-length sequences via bucketing or padding.
    """
    
    def __init__(self, data_dir: str, vocab: dict, batch_size: int = 16,
                 augment: bool = True, shuffle: bool = True,
                 target_img_height: int = 64, target_img_width: int = 256):
        """
        Args:
            data_dir: path to split folder (train/val/test)
            vocab: character-to-index dictionary
            batch_size: batch size
            augment: apply augmentation
            shuffle: shuffle samples
            target_img_height, target_img_width: image resize target
        """
        self.data_dir = Path(data_dir)
        self.vocab = vocab
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.target_h = target_img_height
        self.target_w = target_img_width
        
        self.inv_vocab = {v: k for k, v in vocab.items()}
        
        # Load arrays and labels
        self.images = np.load(self.data_dir / "images.npy")  # (N, H, W)
        self.labels_df = pd.read_csv(self.data_dir / "labels.csv")
        self.words = self.labels_df["word"].values  # (N,)
        
        self.n_samples = len(self.images)
        log.info(f"Loaded {self.n_samples} Kaggle word samples from {data_dir}")
    
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image preserving aspect ratio with padding."""
        h, w = img.shape[:2]
        scale = min(self.target_w / w, self.target_h / h)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        import cv2
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to target
        pad_top = (self.target_h - new_h) // 2
        pad_bottom = self.target_h - new_h - pad_top
        pad_left = (self.target_w - new_w) // 2
        pad_right = self.target_w - new_w - pad_left
        
        padded = np.pad(resized, ((pad_top, pad_bottom), (pad_left, pad_right)),
                       mode="constant", constant_values=1.0)
        return padded
    
    def __call__(self):
        """Generate batches with variable-length label padding."""
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for batch_start in range(0, self.n_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.n_samples)
            batch_idx = indices[batch_start:batch_end]
            
            batch_images = []
            batch_labels = []
            max_len = 0
            
            # First pass: collect and find max label length
            for idx in batch_idx:
                img = self.images[idx]
                word = self.words[idx]
                
                if self.augment:
                    img = augment_image(img)
                
                img = self._resize_image(img)
                img = np.expand_dims(img, axis=-1)
                batch_images.append(img)
                
                label = encode_label(word, self.vocab)
                batch_labels.append(label)
                max_len = max(max_len, len(label))
            
            # Pad labels to max_len
            batch_labels_padded = []
            for label in batch_labels:
                padded = label + [0] * (max_len - len(label))  # 0 = blank
                batch_labels_padded.append(padded)
            
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_labels_padded = np.array(batch_labels_padded, dtype=np.int32)
            
            yield batch_images, batch_labels_padded
    
    def get_tf_dataset(self):
        """Return a tf.data.Dataset."""
        output_signature = (
            tf.TensorSpec(shape=(None, self.target_h, self.target_w, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        )
        return tf.data.Dataset.from_generator(
            self, output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)


class HDF5Dataset:
    """
    Loads data from compressed HDF5 archive (faster I/O).
    """
    
    def __init__(self, h5_path: str, dataset_type: str = "kaggle",
                 split: str = "train", batch_size: int = 16, vocab: dict = None):
        """
        Args:
            h5_path: path to .h5 file
            dataset_type: "emnist" or "kaggle"
            split: "train", "val", or "test"
            batch_size: batch size
            vocab: character vocabulary (required for kaggle)
        """
        self.h5_path = h5_path
        self.dataset_type = dataset_type
        self.split = split
        self.batch_size = batch_size
        self.vocab = vocab
        
        with h5py.File(h5_path, "r") as f:
            self.n_samples = len(f[f"{dataset_type}/{split}/images"])
        
        log.info(f"HDF5 dataset: {dataset_type}/{split}, {self.n_samples} samples")
    
    def __call__(self):
        """Generator for batches."""
        with h5py.File(self.h5_path, "r") as f:
            group = f[f"{self.dataset_type}/{self.split}"]
            images = group["images"]
            
            if self.dataset_type == "emnist":
                labels = group["labels"]
            else:
                words = group["words"]
            
            for batch_start in range(0, self.n_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, self.n_samples)
                
                batch_images = images[batch_start:batch_end]
                
                if self.dataset_type == "emnist":
                    batch_labels = labels[batch_start:batch_end]
                    batch_images = np.expand_dims(batch_images, axis=-1)
                    yield batch_images.astype(np.float32), batch_labels.astype(np.int32)
                else:
                    batch_words = [w.decode('utf-8') if isinstance(w, bytes) else w
                                   for w in words[batch_start:batch_end]]
                    batch_labels = [encode_label(w, self.vocab) for w in batch_words]
                    
                    # Pad labels
                    max_len = max(len(l) for l in batch_labels)
                    batch_labels = [l + [0] * (max_len - len(l)) for l in batch_labels]
                    
                    batch_images = np.expand_dims(batch_images, axis=-1)
                    yield batch_images.astype(np.float32), np.array(batch_labels, dtype=np.int32)
    
    def get_tf_dataset(self):
        """Return a tf.data.Dataset."""
        if self.dataset_type == "emnist":
            output_signature = (
                tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            )
        else:
            output_signature = (
                tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            )
        
        return tf.data.Dataset.from_generator(
            self, output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)