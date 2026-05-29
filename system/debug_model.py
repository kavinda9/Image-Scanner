import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

import cv2
import numpy as np
from utils.ocr import text_from_image

# Create a test image: white background, black text "HELLO WORLD"
test_img = np.ones((120, 600), dtype=np.uint8) * 255
cv2.putText(test_img, "HELLO WORLD", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)

# Perform OCR using the actual function
print("Running text_from_image with PaddleOCR...")
try:
    result = text_from_image(test_img)
    print("Final Output:", result)
except Exception as e:
    print(f"Error occurred: {e}")
