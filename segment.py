"""
segment.py
----------
Text detection and word/line segmentation for prescription images.

Methods:
  1. OpenCV-based: simple thresholding + contour detection (fast, less robust)
  2. EAST detector: deep learning-based text detection (robust, slower)

Usage:
    python segment.py --image prescription.jpg --method opencv --output words/
    python segment.py --image prescription.jpg --method east --output words/
"""

import argparse
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Constants
EAST_MODEL_PATH = "frozen_east_text_detection.pb"
MIN_AREA = 50
MAX_AREA = 100000


def preprocess_image(image_path: str):
    """Read image, convert to grayscale, and apply thresholding."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return img, binary


def find_lines_opencv(binary):
    """Detect horizontal text lines using morphological operations."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        if MIN_AREA < area < MAX_AREA and w > 3 * h:
            lines.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
    
    lines = sorted(lines, key=lambda r: r["y"])
    return lines


def find_words_in_line(line_img):
    """Detect individual words within a line using vertical projection."""
    projection = np.sum(line_img, axis=0)
    threshold = np.max(projection) * 0.1
    word_mask = projection > threshold
    
    diff = np.diff(word_mask.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) == 0:
        return []
    
    if len(ends) == 0:
        ends = np.array([len(word_mask)])
    elif len(starts) > len(ends):
        ends = np.append(ends, len(word_mask))
    
    words = []
    for start, end in zip(starts, ends):
        w = end - start
        if w > 5:
            words.append({"x": int(start), "w": int(w)})
    
    return words


def segment_words_opencv(image_path: str, output_dir: str = "words/"):
    """Segment prescription image into words using OpenCV."""
    img, binary = preprocess_image(image_path)
    h, w = img.shape[:2]
    
    lines = find_lines_opencv(binary)
    log.info(f"Detected {len(lines)} text lines")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filenames = []
    word_idx = 0
    
    for line_idx, line in enumerate(lines):
        line_y, line_h = line["y"], line["h"]
        line_x, line_w = line["x"], line["w"]
        
        line_img = binary[line_y:line_y + line_h, line_x:line_x + line_w]
        words = find_words_in_line(line_img)
        log.info(f"  Line {line_idx}: {len(words)} words")
        
        for word in words:
            x1 = line_x + word["x"]
            x2 = x1 + word["w"]
            y1, y2 = line_y, line_y + line_h
            
            word_img = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            filename = f"{output_dir}/word_{word_idx:05d}.png"
            cv2.imwrite(filename, word_img)
            filenames.append(filename)
            word_idx += 1
    
    log.info(f"Saved {word_idx} word crops to {output_dir}")
    return filenames


def detect_text_east(image_path: str, output_dir: str = "words/",
                     confidence_threshold: float = 0.5):
    """Use EAST text detector for word segmentation."""
    if not os.path.exists(EAST_MODEL_PATH):
        log.warning(f"EAST model not found. Falling back to OpenCV.")
        return segment_words_opencv(image_path, output_dir)
    
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    new_h, new_w = 320, 320
    ratio_w = w / float(new_w)
    ratio_h = h / float(new_h)
    
    resized = cv2.resize(img, (new_w, new_h))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (new_w, new_h),
                                  (123.68, 116.78, 103.94),
                                  swapRB=True, crop=False)
    
    net = cv2.dnn.readNet(EAST_MODEL_PATH)
    layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    
    net.setInput(blob)
    (scores, geometry) = net.forward(layer_names)
    
    boxes = []
    confidences = []
    
    (num_rows, num_cols) = scores.shape[2:4]
    for y in range(0, num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]
        
        for x in range(0, num_cols):
            if scores_data[x] < confidence_threshold:
                continue
            
            (offset_x, offset_y) = (x * 4.0, y * 4.0)
            (h_box, w_box) = (x_data1[x] + x_data3[x], x_data0[x] + x_data2[x])
            
            angle = angles[x]
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            h_box *= ratio_h
            w_box *= ratio_w
            
            offset_x *= ratio_w
            offset_y *= ratio_h
            
            p1_x = offset_x + cos_a * x_data0[x] * ratio_w
            p1_y = offset_y - sin_a * x_data0[x] * ratio_w
            p3_x = offset_x - sin_a * x_data1[x] * ratio_h
            p3_y = offset_y - cos_a * x_data1[x] * ratio_h
            
            boxes.append(((p1_x, p1_y), (p3_x, p3_y), (w_box, h_box), angle))
            confidences.append(float(scores_data[x]))
    
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confidence_threshold, 0.4)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filenames = []
    
    for i in indices:
        vertices = cv2.boxPoints(boxes[i[0]])
        vertices = np.int0(vertices)
        
        x_min = max(0, int(np.min(vertices[:, 0])))
        x_max = min(w, int(np.max(vertices[:, 0])))
        y_min = max(0, int(np.min(vertices[:, 1])))
        y_max = min(h, int(np.max(vertices[:, 1])))
        
        word_img = img[y_min:y_max, x_min:x_max]
        if word_img.size > 0:
            filename = f"{output_dir}/word_{len(filenames):05d}.png"
            cv2.imwrite(filename, word_img)
            filenames.append(filename)
    
    log.info(f"EAST detected {len(filenames)} text regions")
    return filenames


def visualize_detections(image_path: str, boxes: list, output_path: str):
    """Draw bounding boxes on image and save."""
    img = cv2.imread(image_path)
    
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imwrite(output_path, img)
    log.info(f"Saved visualisation → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Segment prescription into words.")
    parser.add_argument("--image", required=True, help="Input prescription image")
    parser.add_argument("--method", choices=["opencv", "east"],
                        default="opencv", help="Segmentation method")
    parser.add_argument("--output", default="words/", help="Output directory for word crops")
    parser.add_argument("--visualise", action="store_true",
                        help="Save visualisation of detections")
    args = parser.parse_args()
    
    if args.method == "opencv":
        filenames = segment_words_opencv(args.image, args.output)
    else:
        filenames = detect_text_east(args.image, args.output)
    
    if args.visualise:
        output_vis = "segmentation_visualisation.png"
        img, binary = preprocess_image(args.image)
        lines = find_lines_opencv(binary)
        visualize_detections(args.image, lines, output_vis)
    
    log.info(f"Segmentation complete. Saved {len(filenames)} words.")


if __name__ == "__main__":
    main()