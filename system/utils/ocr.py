"""
Module for extracting the text from an image using PaddleOCR.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from spellchecker import SpellChecker
from paddleocr import PaddleOCR

# Initialize PaddleOCR engine with enable_mkldnn=False to prevent PIR executor crash
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    enable_mkldnn=False
)

spellchecker = SpellChecker()
common_mistakes = {
    'ls': 'is',
    'lt': 'it',
    'l': 'i',
    'ln': 'in',
    'lts': 'its',
    'll': 'it',
}

def text_from_image(img: np.ndarray) -> str:
    """
    Extract the text from an image using PaddleOCR with a robust, hybrid parsing loop.
    Supports both classic list-of-lists format and the new dictionary-based PP-Structure/PaddleX format.
    """
    temp_path = "temp_ocr_image.png"
    cv2.imwrite(temp_path, img)

    result = ocr_engine.ocr(temp_path)
    
    # Remove temp image
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass

    extracted_text = []
    
    if result is not None:
        for item in result:
            if isinstance(item, dict):
                # Handle new PaddleOCR/PaddleX dict-based format
                if 'rec_texts' in item and item['rec_texts'] is not None:
                    extracted_text.extend(item['rec_texts'])
            elif isinstance(item, list):
                # Handle classic list-of-lists format
                for word in item:
                    if word is not None and len(word) > 1 and word[1] is not None:
                        extracted_text.append(word[1][0])

    text = " ".join(extracted_text)
    print("OCR Result:", text)

    return perform_spellchecking(text)

def perform_spellchecking(text: str) -> str:
    """
    Fix spelling errors caused by OCR mispredictions.
    """
    if not text.strip():
        return ""
        
    words = text.split(' ')
    words = [common_mistakes.get(word, word) for word in words]
    corrected_words = [spellchecker.correction(word) or word for word in words]
    print(f'After spellchecking: {" ".join(corrected_words)}')
    return ' '.join(corrected_words)
