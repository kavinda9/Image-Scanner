"""
evaluate.py
-----------
Model evaluation on test sets.

Metrics:
  • Character Error Rate (CER) – edit distance / total chars
  • Word Error Rate (WER) – incorrect words / total words
  • Overall accuracy
  • Confusion matrices (character-level)

Usage:
    python evaluate.py --model models/char_model.h5 --stage char --split test
    python evaluate.py --model models/word_model.h5 --stage word --split test
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import EMNISTDataset, KaggleDataset
from model import ctc_decode
from utils import get_logger, character_error_rate, word_error_rate, edit_distance

log = get_logger(__name__)


# ── Metrics Computation ──────────────────────────────────────────────────────

def evaluate_char_model(model, test_dataset, inv_vocab: dict) -> dict:
    """
    Evaluate character model on test set.
    
    Returns:
        {
            'accuracy': float,
            'cer': float,
            'predictions': list[str],
            'targets': list[str],
            'confusion_matrix': np.ndarray
        }
    """
    log.info("Evaluating character model...")
    
    predictions_all = []
    targets_all = []
    correct = 0
    total = 0
    
    # Confusion matrix
    num_classes = len(inv_vocab)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    
    for batch_images, batch_labels in test_dataset:
        logits = model.predict(batch_images, verbose=0)
        preds = np.argmax(logits, axis=1)
        
        # Track predictions
        for pred, target in zip(preds, batch_labels):
            pred_char = inv_vocab.get(int(pred), "?")
            target_char = inv_vocab.get(int(target), "?")
            predictions_all.append(pred_char)
            targets_all.append(target_char)
            
            if pred == target:
                correct += 1
            total += 1
            
            # Confusion matrix
            confusion[int(target), int(pred)] += 1
    
    accuracy = correct / max(total, 1)
    cer = character_error_rate(predictions_all, targets_all)
    
    log.info(f"  Accuracy: {accuracy:.4f}")
    log.info(f"  CER: {cer:.4f}")
    
    return {
        'accuracy': accuracy,
        'cer': cer,
        'predictions': predictions_all,
        'targets': targets_all,
        'confusion_matrix': confusion
    }


def evaluate_word_model(model, test_dataset, inv_vocab: dict) -> dict:
    """
    Evaluate word model (CRNN+CTC) on test set.
    
    Returns:
        {
            'cer': float,
            'wer': float,
            'predictions': list[str],
            'targets': list[str],
            'confidence': list[float]
        }
    """
    log.info("Evaluating word model (CTC)...")
    
    predictions_all = []
    targets_all = []
    confidences_all = []
    
    for batch_images, batch_labels in test_dataset:
        logits = model.predict(batch_images, verbose=0)  # (batch, seq_len, num_classes)
        
        # Greedy CTC decode
        for i in range(len(logits)):
            pred_indices = ctc_decode(logits[i])
            pred_word = "".join([inv_vocab.get(idx, "?") for idx in pred_indices])
            
            # Get confidence (mean softmax prob of top predictions)
            softmax_probs = tf.nn.softmax(logits[i]).numpy()
            top_probs = np.max(softmax_probs, axis=1)
            confidence = float(np.mean(top_probs))
            
            predictions_all.append(pred_word)
            confidences_all.append(confidence)
            
            # Ground truth (decode from sparse labels)
            target_indices = batch_labels[i][batch_labels[i] != 0]  # Remove padding
            target_word = "".join([inv_vocab.get(int(idx), "?") for idx in target_indices])
            targets_all.append(target_word)
    
    cer = character_error_rate(predictions_all, targets_all)
    wer = word_error_rate(predictions_all, targets_all)
    
    log.info(f"  CER: {cer:.4f}")
    log.info(f"  WER: {wer:.4f}")
    
    return {
        'cer': cer,
        'wer': wer,
        'predictions': predictions_all,
        'targets': targets_all,
        'confidence': confidences_all
    }


# ── Visualization ────────────────────────────────────────────────────────────

def plot_confusion_matrix(confusion: np.ndarray, inv_vocab: dict, output_path: str):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Limit to top N classes for readability
    N = min(30, confusion.shape[0])
    confusion_subset = confusion[:N, :N]
    
    labels = [inv_vocab.get(i, str(i)) for i in range(N)]
    
    sns.heatmap(confusion_subset, annot=False, cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels, cbar_kws={"label": "Count"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_title(f"Confusion Matrix (top {N} classes)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    log.info(f"Saved confusion matrix → {output_path}")
    plt.close()


def plot_error_distribution(predictions: list[str], targets: list[str],
                            output_path: str):
    """Plot edit distance distribution."""
    distances = [edit_distance(p, t) for p, t in zip(predictions, targets)]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(distances, bins=range(0, max(distances) + 2), edgecolor="black", color="steelblue")
    ax.set_xlabel("Edit Distance")
    ax.set_ylabel("Count")
    ax.set_title("Edit Distance Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    log.info(f"Saved error distribution → {output_path}")
    plt.close()


def plot_confidence_distribution(confidences: list[float], output_path: str):
    """Plot model confidence distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(confidences, bins=50, edgecolor="black", color="coral")
    ax.set_xlabel("Confidence (mean softmax prob)")
    ax.set_ylabel("Count")
    ax.set_title("Model Confidence Distribution")
    plt.axvline(np.mean(confidences), color="red", linestyle="--", label=f"Mean: {np.mean(confidences):.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    log.info(f"Saved confidence distribution → {output_path}")
    plt.close()


# ── Error Analysis ───────────────────────────────────────────────────────────

def analyze_errors(predictions: list[str], targets: list[str], output_path: str, top_n: int = 20):
    """Find and save top N most common errors."""
    errors = []
    for pred, target in zip(predictions, targets):
        if pred != target:
            dist = edit_distance(pred, target)
            errors.append({
                'target': target,
                'prediction': pred,
                'distance': dist
            })
    
    # Sort by edit distance (descending)
    errors = sorted(errors, key=lambda x: x['distance'], reverse=True)[:top_n]
    
    df = pd.DataFrame(errors)
    df.to_csv(output_path, index=False)
    
    log.info(f"Saved top {top_n} errors → {output_path}")
    log.info("\nTop errors:")
    for i, err in enumerate(errors[:5], 1):
        log.info(f"  {i}. Target: '{err['target']}' → Pred: '{err['prediction']}' (dist={err['distance']})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument("--model", required=True, help="Path to trained .h5 model")
    parser.add_argument("--stage", choices=["char", "word"], default="word",
                        help="Model stage")
    parser.add_argument("--processed_dir", default="data/processed",
                        help="Directory with processed splits")
    parser.add_argument("--split", choices=["train", "val", "test"],
                        default="test", help="Which split to evaluate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="outputs/",
                        help="Where to save evaluation plots")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    log.info(f"Loading model from {args.model}")
    model = keras.models.load_model(args.model)
    
    # Load vocabulary
    if args.stage == "char":
        with open("character_db.json") as f:
            char_db = json.load(f)
        inv_vocab = {int(k): v for k, v in char_db["emnist_balanced"].items()}
    else:
        with open(f"{args.processed_dir}/kaggle/vocab.json") as f:
            vocab = json.load(f)
        inv_vocab = {v: k for k, v in vocab.items()}
    
    # Load test dataset
    if args.stage == "char":
        test_dataset = EMNISTDataset(
            f"{args.processed_dir}/emnist/{args.split}",
            vocab=inv_vocab,
            batch_size=args.batch_size,
            augment=False,
            shuffle=False
        )
        results = evaluate_char_model(model, test_dataset.get_tf_dataset(), inv_vocab)
        
        # Plots
        plot_confusion_matrix(results['confusion_matrix'], inv_vocab,
                              f"{args.output_dir}/confusion_matrix.png")
        plot_error_distribution(results['predictions'], results['targets'],
                                f"{args.output_dir}/error_distribution.png")
        analyze_errors(results['predictions'], results['targets'],
                       f"{args.output_dir}/top_errors.csv")
    else:
        test_dataset = KaggleDataset(
            f"{args.processed_dir}/kaggle/{args.split}",
            vocab=inv_vocab,
            batch_size=args.batch_size,
            augment=False,
            shuffle=False
        )
        results = evaluate_word_model(model, test_dataset.get_tf_dataset(), inv_vocab)
        
        # Plots
        plot_confidence_distribution(results['confidence'],
                                     f"{args.output_dir}/confidence_dist.png")
        plot_error_distribution(results['predictions'], results['targets'],
                                f"{args.output_dir}/error_distribution.png")
        analyze_errors(results['predictions'], results['targets'],
                       f"{args.output_dir}/top_errors.csv")
    
    # Summary
    log.info("=" * 70)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 70)
    for key, value in results.items():
        if key not in ['predictions', 'targets', 'confidence', 'confusion_matrix']:
            log.info(f"{key}: {value:.4f}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()