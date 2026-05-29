"""
train.py
--------
Training orchestration for both character (EMNIST) and word (Kaggle) models.

Stages:
  1. Character model – simple CNN on EMNIST, saves as char_model.h5
  2. Word model – CRNN+CTC on Kaggle, optionally initialised from char model
  3. Fine-tuning – optional end-to-end refinement on mixed/real data

Usage:
    # Stage 1: Train character model
    python train.py --stage char --epochs 30 --batch_size 128

    # Stage 2: Train word model with pretrained CNN
    python train.py --stage word --epochs 50 --batch_size 32 \
                    --pretrained models/char_model.h5

    # Stage 3: Fine-tune on full prescriptions
    python train.py --stage word --epochs 10 --batch_size 16 \
                    --pretrained models/word_model.h5
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)

from data_loader import EMNISTDataset, KaggleDataset, HDF5Dataset
from model import build_char_cnn, build_crnn_ctc, print_model_summary
from utils import set_seed, get_logger

log = get_logger(__name__)


# ── CTC Loss Custom Training Loop ────────────────────────────────────────────

class CTCTrainer:
    """Custom training loop for CRNN+CTC models."""
    
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = keras.losses.CTC()
        self.train_loss_metric = keras.metrics.Mean()
        self.val_loss_metric = keras.metrics.Mean()
    
    def compute_loss(self, y_true, y_pred):
        """Compute CTC loss."""
        batch_size = tf.shape(y_pred)[0]
        input_length = tf.shape(y_pred)[1]
        
        # Label length: count non-zero entries per sequence
        label_length = tf.reduce_sum(
            tf.cast(y_true != 0, tf.int32), axis=1
        )
        
        # Input length: same for all in batch
        input_length = tf.tile([input_length], [batch_size])
        
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        return tf.reduce_mean(loss)
    
    def train_step(self, x, y):
        """Single training step."""
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.compute_loss(y, y_pred)
        
        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_weights)
        )
        self.train_loss_metric.update_state(loss)
        return loss
    
    def val_step(self, x, y):
        """Single validation step."""
        y_pred = self.model(x, training=False)
        loss = self.compute_loss(y, y_pred)
        self.val_loss_metric.update_state(loss)
        return loss


# ── Training Functions ───────────────────────────────────────────────────────

def train_char_model(processed_dir: str, output_dir: str = "models/",
                     epochs: int = 30, batch_size: int = 128,
                     learning_rate: float = 1e-3):
    """
    Train character classifier on EMNIST.
    """
    log.info("=" * 70)
    log.info("STAGE 1: Training Character Model (EMNIST)")
    log.info("=" * 70)
    
    set_seed(42)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load character mapping
    with open("character_db.json") as f:
        char_db = json.load(f)
    
    num_classes = len(char_db["emnist_balanced"])
    log.info(f"Number of character classes: {num_classes}")
    
    # Build model
    model = build_char_cnn(num_classes=num_classes)
    print_model_summary(model)
    
    # Compile with custom learning rate
    model.optimizer.learning_rate = learning_rate
    
    # Load data
    train_dataset = EMNISTDataset(
        f"{processed_dir}/emnist/train",
        vocab=char_db["emnist_balanced"],
        batch_size=batch_size,
        augment=True,
        shuffle=True
    )
    
    val_dataset = EMNISTDataset(
        f"{processed_dir}/emnist/val",
        vocab=char_db["emnist_balanced"],
        batch_size=batch_size,
        augment=False,
        shuffle=False
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            f"{output_dir}/char_model_best.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(log_dir="logs/char", histogram_freq=1)
    ]
    
    # Train
    history = model.fit(
        train_dataset.get_tf_dataset(),
        epochs=epochs,
        steps_per_epoch=None,
        validation_data=val_dataset.get_tf_dataset(),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save
    model.save(f"{output_dir}/char_model.h5")
    log.info(f"✅  Character model saved → {output_dir}/char_model.h5")
    
    return model, history


def train_word_model(processed_dir: str, output_dir: str = "models/",
                     epochs: int = 50, batch_size: int = 32,
                     learning_rate: float = 1e-3,
                     pretrained_weights: str = None):
    """
    Train word recognizer (CRNN+CTC) on Kaggle dataset.
    """
    log.info("=" * 70)
    log.info("STAGE 2: Training Word Model (Kaggle + CTC)")
    log.info("=" * 70)
    
    set_seed(42)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load vocabulary
    with open("character_db.json") as f:
        char_db = json.load(f)
    
    with open(f"{processed_dir}/kaggle/vocab.json") as f:
        vocab = json.load(f)
    
    num_classes = len(vocab)
    log.info(f"Vocabulary size: {num_classes}")
    
    # Build model
    model = build_crnn_ctc(
        num_classes=num_classes,
        rnn_units=128,
        pretrained_cnn_weights=pretrained_weights
    )
    print_model_summary(model)
    
    # Compile with CTC loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CTC()
    )
    
    # Load data
    train_dataset = KaggleDataset(
        f"{processed_dir}/kaggle/train",
        vocab=vocab,
        batch_size=batch_size,
        augment=True,
        shuffle=True
    )
    
    val_dataset = KaggleDataset(
        f"{processed_dir}/kaggle/val",
        vocab=vocab,
        batch_size=batch_size,
        augment=False,
        shuffle=False
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            f"{output_dir}/word_model_best.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(log_dir="logs/word", histogram_freq=1)
    ]
    
    # Train
    history = model.fit(
        train_dataset.get_tf_dataset(),
        epochs=epochs,
        steps_per_epoch=None,
        validation_data=val_dataset.get_tf_dataset(),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save
    model.save(f"{output_dir}/word_model.h5")
    log.info(f"✅  Word model saved → {output_dir}/word_model.h5")
    
    return model, history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train OCR models.")
    parser.add_argument("--stage", choices=["char", "word"],
                        default="char", help="Which stage to train")
    parser.add_argument("--processed_dir", default="data/processed",
                        help="Directory with processed splits")
    parser.add_argument("--output_dir", default="models/",
                        help="Where to save model weights")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--pretrained", default=None,
                        help="Path to pretrained model weights")
    
    args = parser.parse_args()
    
    if args.stage == "char":
        model, history = train_char_model(
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    else:  # word
        model, history = train_word_model(
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            pretrained_weights=args.pretrained
        )
    
    log.info("Training complete.")


if __name__ == "__main__":
    main()