"""
inference.py
------------
End-to-end inference pipeline + FastAPI REST service.

Pipeline:
  1. Load prescription image
  2. Segment into word crops (segment.py)
  3. Recognize each word (word model)
  4. Post-process (spell-check, drug dict lookup)
  5. Return recognized text with confidence

Modes:
  • CLI: single image prediction
  • FastAPI REST server: POST image → JSON response

Usage (CLI):
    python inference.py --image prescription.jpg \
                        --word_model models/word_model.h5

Usage (REST API):
    uvicorn inference:app --host 0.0.0.0 --port 8000
    curl -X POST -F "image=@prescription.jpg" http://localhost:8000/recognize
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from segment import segment_words_opencv
from model import ctc_decode
from utils import get_logger

# FastAPI imports (optional)
try:
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

log = get_logger(__name__)


# ── Inference Pipeline ───────────────────────────────────────────────────────

class PrescriptionOCR:
    """End-to-end OCR pipeline for prescription images."""
    
    def __init__(self, word_model_path: str, vocab_path: Optional[str] = None,
                 drug_dict_path: Optional[str] = None,
                 confidence_threshold: float = 0.7):
        """
        Args:
            word_model_path: path to trained word CRNN model
            vocab_path: path to vocab.json (auto-finds if None)
            drug_dict_path: path to drug name dictionary JSON (optional)
            confidence_threshold: flag words below this confidence for review
        """
        self.word_model = keras.models.load_model(word_model_path)
        log.info(f"Loaded word model from {word_model_path}")
        
        # Load vocabulary
        if vocab_path is None:
            vocab_path = "data/processed/kaggle/vocab.json"
        
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        log.info(f"Loaded vocabulary: {len(self.vocab)} characters")
        
        # Load drug dictionary (optional)
        self.drug_dict = set()
        if drug_dict_path and Path(drug_dict_path).exists():
            with open(drug_dict_path) as f:
                self.drug_dict = set(json.load(f))
            log.info(f"Loaded {len(self.drug_dict)} known drug names")
        
        self.confidence_threshold = confidence_threshold
    
    def recognize_word(self, word_image: np.ndarray) -> tuple[str, float]:
        """
        Recognize a single word image.
        
        Args:
            word_image: (H, W) or (H, W, 3) image
        
        Returns:
            (recognized_text, confidence)
        """
        # Preprocess
        if len(word_image.shape) == 3:
            word_image = cv2.cvtColor(word_image, cv2.COLOR_BGR2GRAY)
        
        # Normalize
        word_image = word_image.astype(np.float32) / 255.0
        
        # Resize to model input size (64, 256)
        h, w = word_image.shape[:2]
        scale = min(256 / w, 64 / h)
        new_h, new_w = int(h * scale), int(w * scale)
        word_image = cv2.resize(word_image, (new_w, new_h))
        
        # Pad to (64, 256)
        pad_top = (64 - new_h) // 2
        pad_bottom = 64 - new_h - pad_top
        pad_left = (256 - new_w) // 2
        pad_right = 256 - new_w - pad_left
        
        word_image = np.pad(word_image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                           mode='constant', constant_values=1.0)
        
        # Add batch and channel dims
        word_image = np.expand_dims(np.expand_dims(word_image, 0), -1)
        
        # Predict
        logits = self.word_model.predict(word_image, verbose=0)  # (1, seq_len, num_classes)
        
        # Decode
        pred_indices = ctc_decode(logits[0])
        text = "".join([self.inv_vocab.get(idx, "?") for idx in pred_indices])
        
        # Confidence: mean softmax probability
        softmax_probs = tf.nn.softmax(logits[0]).numpy()
        top_probs = np.max(softmax_probs, axis=1)
        confidence = float(np.mean(top_probs))
        
        return text, confidence
    
    def recognize_prescription(self, image_path: str,
                               return_word_details: bool = False) -> dict:
        """
        Recognize full prescription image.
        
        Args:
            image_path: path to prescription image
            return_word_details: include per-word details in output
        
        Returns:
            {
                'full_text': str,
                'words': [{'text': str, 'confidence': float, 'flagged': bool}, ...],
                'needs_review': bool,
                'timestamp': str
            }
        """
        log.info(f"Processing prescription: {image_path}")
        
        # Segment into words
        word_files = segment_words_opencv(image_path, output_dir="/tmp/ocr_words/")
        log.info(f"Segmented {len(word_files)} word regions")
        
        words = []
        full_text_parts = []
        needs_review = False
        
        for word_file in word_files:
            word_img = cv2.imread(word_file)
            if word_img is None:
                continue
            
            text, confidence = self.recognize_word(word_img)
            
            # Check if flagged
            flagged = confidence < self.confidence_threshold
            if flagged:
                needs_review = True
            
            # Drug name validation (optional)
            drug_match = False
            if self.drug_dict and text.lower() in self.drug_dict:
                drug_match = True
            
            word_info = {
                'text': text,
                'confidence': float(confidence),
                'flagged': flagged,
                'drug_match': drug_match
            }
            words.append(word_info)
            full_text_parts.append(text)
        
        full_text = " ".join(full_text_parts)
        
        result = {
            'full_text': full_text,
            'needs_review': needs_review,
            'words': words if return_word_details else None,
            'timestamp': str(Path(image_path).stat().st_mtime)
        }
        
        log.info(f"Recognition complete. Needs review: {needs_review}")
        return result


# ── FastAPI Application ──────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    app = FastAPI(title="Prescription OCR", version="1.0.0")
    
    # Global OCR instance
    ocr = None
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize OCR pipeline on startup."""
        global ocr
        try:
            ocr = PrescriptionOCR(
                word_model_path="models/word_model.h5",
                vocab_path="data/processed/kaggle/vocab.json"
            )
            log.info("✅ OCR pipeline initialized")
        except Exception as e:
            log.error(f"Failed to initialize OCR: {e}")
    
    @app.post("/recognize")
    async def recognize_prescription(image: UploadFile = File(...)):
        """
        Recognize text in prescription image.
        
        Args:
            image: prescription image file (POST multipart/form-data)
        
        Returns:
            JSON with recognized text, confidence, flags for review
        """
        if ocr is None:
            return JSONResponse(
                status_code=500,
                content={"error": "OCR pipeline not initialized"}
            )
        
        try:
            # Save uploaded image temporarily
            temp_path = f"/tmp/{image.filename}"
            contents = await image.read()
            with open(temp_path, "wb") as f:
                f.write(contents)
            
            # Recognize
            result = ocr.recognize_prescription(temp_path, return_word_details=True)
            
            return JSONResponse(content=result)
        
        except Exception as e:
            log.error(f"Recognition error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "ocr_ready": ocr is not None}


# ── CLI Interface ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prescription OCR inference.")
    parser.add_argument("--image", help="Input prescription image (CLI mode)")
    parser.add_argument("--word_model", default="models/word_model.h5",
                        help="Path to trained word model")
    parser.add_argument("--vocab", default="data/processed/kaggle/vocab.json",
                        help="Path to vocabulary JSON")
    parser.add_argument("--drug_dict", default=None,
                        help="Path to drug dictionary JSON")
    parser.add_argument("--output", default="prescription_output.json",
                        help="Output JSON file")
    parser.add_argument("--server", action="store_true",
                        help="Start FastAPI server instead of CLI mode")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    
    if args.server:
        if not FASTAPI_AVAILABLE:
            log.error("FastAPI not available. Install: pip install fastapi uvicorn")
            return
        log.info(f"Starting OCR API server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        if not args.image:
            log.error("Provide --image for CLI mode or --server for API mode")
            return
        
        ocr = PrescriptionOCR(args.word_model, args.vocab, args.drug_dict)
        result = ocr.recognize_prescription(args.image, return_word_details=True)
        
        # Save and print
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        
        log.info("=" * 70)
        log.info("RECOGNITION RESULT")
        log.info("=" * 70)
        log.info(f"Full text:\n{result['full_text']}")
        log.info(f"\nNeeds review: {result['needs_review']}")
        if result['words']:
            log.info("\nPer-word details:")
            for i, w in enumerate(result['words'][:10], 1):
                flag = "⚠️ " if w['flagged'] else "✓ "
                log.info(f"  {i}. {flag} '{w['text']}' (confidence: {w['confidence']:.3f})")
        log.info(f"\nResult saved → {args.output}")


if __name__ == "__main__":
    main()