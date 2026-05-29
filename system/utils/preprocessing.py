"""
Module for preprocessing the images, in order the get them to work best
with the character-cutting and the models.
"""
from typing import Optional

import numpy as np
import cv2
from scipy.spatial import distance

from hough_rect import find_hough_rect, rect_area, order_points


def preprocess_image(img: np.ndarray,
                     points: Optional[list[tuple[int, int]]] = None) -> np.ndarray:
    """
    Simplified preprocessing for PaddleOCR. 
    Avoids aggressive binarization/thresholding which destroys text features.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Apply soft denoising to retain fine stroke features
    gray = cv2.fastNlMeansDenoising(gray)

    # Save debug image to visually inspect text quality
    cv2.imwrite("debug.png", gray)
    
    return gray


def find_page_points(img: np.ndarray) -> list[tuple[int, int]]:
    """
    Find the four points defining the page (region-of-interest).
    If no such points found, return the edges of the image.
    """
    if (rect := find_hough_rect(img)) is None \
            or rect_area(rect) < 0.1 * img.shape[0] * img.shape[1]:
        return [(0, 0), (img.shape[1], 0), (img.shape[1], img.shape[0]), (0, img.shape[0])]
    return [tuple(pt) for pt in rect]


def four_point_transform(img: np.ndarray, rect: np.ndarray) -> np.ndarray:
    """Warp the image around it's region-of-interest."""
    # obtain a consistent order of the points
    width, height = calc_dimensions(rect)

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype=np.float32)

    # compute the perspective transform matrix and then apply it
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, matrix, (width, height))
    return warped


def calc_dimensions(ordered_pts: np.ndarray) -> tuple[int, int]:
    """Calculate the dimensions of the new warped image - width and height."""

    (tl, tr, br, bl) = ordered_pts

    # compute the width of the new image - the maximum distance between
    # right and left points
    width1 = distance.euclidean(br, bl)
    width2 = distance.euclidean(tr, tl)
    max_width = max(int(width1), int(width2))

    # compute the height of the new image - the maximum distance between
    # top and bottom points
    height1 = distance.euclidean(tr, br)
    height2 = distance.euclidean(tl, bl)
    max_height = max(int(height1), int(height2))

    return max_width, max_height
