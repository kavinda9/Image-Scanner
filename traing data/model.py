"""
model.py
--------
Neural network architectures for OCR:
  1. Character CNN (EMNIST) – simple CNN classifier
  2. CRNN + CTC (Kaggle) – convolutional + recurrent + temporal classification

Both use TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import logging

log = logging.getLogger(__name__)


# ── Character CNN (Stage 1) ──────────────────────────────────────────────────

def build_char_cnn(num_classes: int, input_shape: tuple = (28, 28, 1)) -> Model:
    """
    Simple CNN for character classification (EMNIST).
    
    Architecture:
      Conv2D(32) + ReLU + MaxPool
      Conv2D(64) + ReLU + MaxPool
      Flatten + Dense(128) + ReLU + Dropout
      Dense(num_classes) + Softmax
    
    Args:
        num_classes: number of character classes (e.g., 47 for EMNIST Balanced)
        input_shape: (height, width, channels)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3 (optional)
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    log.info(f"Built character CNN: {num_classes} classes")
    return model


# ── CRNN + CTC (Stage 2) ─────────────────────────────────────────────────────

def build_crnn_ctc(num_classes: int,
                   input_shape: tuple = (64, 256, 1),
                   rnn_units: int = 128,
                   pretrained_cnn_weights: str = None) -> Model:
    """
    CRNN (CNN + Bidirectional LSTM + CTC) for word recognition.
    
    Architecture:
      CNN feature extraction (2 Conv blocks)
        ↓
      Reshape for RNN (flatten spatial dims)
        ↓
      Bidirectional LSTM (2 layers, 128 units each)
        ↓
      Dense layer (num_classes) for character probabilities
        ↓
      CTC loss for alignment-free training
    
    Args:
        num_classes: alphabet size (e.g., 47 for EMNIST Balanced)
        input_shape: (height, width, channels) – typically (64, 256, 1)
        rnn_units: hidden units in LSTM layers
        pretrained_cnn_weights: path to .h5 character model (optional)
    
    Returns:
        Uncompiled model (CTC loss is handled separately)
    """
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # ── CNN Feature Extraction ───────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Shape after pooling: (H/4, W/4, 64)
    # For input (64, 256, 1) → (16, 64, 64)
    
    # ── Reshape for RNN ──────────────────────────────────────────────────────
    # Collapse height dimension, keep width as sequence length
    x = layers.Reshape(target_shape=(x.shape[2] * x.shape[1], x.shape[3]))(x)
    # Shape: (seq_len, features) where seq_len = W/4
    
    # ── RNN Layers (Bidirectional LSTM) ──────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(rnn_units, return_sequences=True, dropout=0.5)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(rnn_units, return_sequences=True, dropout=0.5)
    )(x)
    
    # ── Output layer (logits for CTC) ────────────────────────────────────────
    # No softmax – CTC will handle it
    outputs = layers.Dense(num_classes, activation="linear")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Optionally load pretrained CNN weights from character model
    if pretrained_cnn_weights:
        try:
            char_model = keras.models.load_model(pretrained_cnn_weights)
            cnn_weights = char_model.layers[0:4]  # First 4 layers are CNN
            for i, layer in enumerate(model.layers[:4]):
                if layer.get_weights():
                    layer.set_weights(cnn_weights[i].get_weights())
            log.info(f"Loaded pretrained CNN weights from {pretrained_cnn_weights}")
        except Exception as e:
            log.warning(f"Could not load pretrained weights: {e}")
    
    log.info(f"Built CRNN+CTC: {num_classes} classes, {rnn_units} RNN units")
    return model


# ── CTC Loss & Training Helpers ──────────────────────────────────────────────

class CTCLayer(layers.Layer):
    """
    Custom layer that wraps CTC loss.
    Useful for end-to-end trainable models.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, y_true, y_pred):
        # y_true: sparse tensor of shape (batch,) with max word length encoded
        # y_pred: (batch, seq_len, num_classes) logits from dense layer
        
        batch_len = tf.cast(tf.shape(y_pred)[0], tf.int64)
        input_length = tf.cast(tf.shape(y_pred)[1], tf.int64)
        
        label_length = tf.cast(tf.shape(y_true)[1], tf.int64)
        
        input_length = input_length * tf.ones(shape=(batch_len,), dtype=tf.int64)
        label_length = label_length * tf.ones(shape=(batch_len,), dtype=tf.int64)
        
        loss = keras.backend.ctc_batch_cost(
            y_true, y_pred, input_length, label_length
        )
        self.add_loss(loss)
        
        return y_pred


def ctc_decode(y_pred, blank_idx: int = 0) -> list[str]:
    """
    Greedy CTC decoding: collapse repeats, remove blanks.
    
    Args:
        y_pred: (seq_len, num_classes) logits or (batch, seq_len, num_classes)
        blank_idx: index of blank token (typically 0)
    
    Returns:
        List of decoded strings (if batch) or single string (if single sample)
    """
    is_batch = len(y_pred.shape) == 3
    
    if is_batch:
        results = []
        for sample in y_pred:
            results.append(ctc_decode(sample, blank_idx))
        return results
    
    # Single sample: (seq_len, num_classes)
    indices = tf.argmax(y_pred, axis=1).numpy()
    
    result = []
    prev_idx = None
    for idx in indices:
        if idx != prev_idx and idx != blank_idx:
            result.append(int(idx))
        prev_idx = idx
    
    return result


# ── Model Summary Utility ────────────────────────────────────────────────────

def print_model_summary(model: Model):
    """Pretty-print model architecture."""
    log.info("=" * 70)
    log.info("Model Summary:")
    log.info("=" * 70)
    model.summary(print_fn=lambda x: log.info(x))
    log.info("=" * 70)