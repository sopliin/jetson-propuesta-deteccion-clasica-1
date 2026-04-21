"""
Morphological cleaning and contour extraction.

Pipeline:
  raw fg_mask
    ↓  MORPH_OPEN  (erosion → dilation)   removes small noise specks
    ↓  MORPH_CLOSE (dilation → erosion)   fills holes inside blobs
    ↓  findContours                        extracts object outlines
    ↓  area filter                         discards implausibly small/large blobs
    →  list of (x, y, w, h) bounding boxes

Kernel shape:
  MORPH_ELLIPSE is preferred over MORPH_RECT for real-world blobs because
  objects are round/irregular, not rectangular, and ellipse kernels avoid
  creating artificial sharp corners in the mask.
"""
import cv2
import numpy as np


def clean_mask(fg_mask, open_ksize: int = 5, close_ksize: int = 15):
    """
    Apply morphological opening then closing to the foreground mask.

    Args:
        fg_mask      : binary mask from background subtractor
        open_ksize   : kernel side length for opening (noise removal)
        close_ksize  : kernel side length for closing (hole filling)

    Returns:
        Cleaned binary mask (same dtype as fg_mask)
    """
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize,  open_ksize))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))

    mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask,    cv2.MORPH_CLOSE, k_close)
    return mask


def extract_contours(mask) -> list:
    """
    Find external contours in the cleaned mask.

    RETR_EXTERNAL  : only the outermost boundary of each blob (no sub-contours).
    CHAIN_APPROX_SIMPLE : compresses collinear points — saves memory and is
                          sufficient because we only need bounding rectangles.

    Returns:
        List of raw contour arrays (each is ndarray of shape (N,1,2))
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours(contours, min_area: int = 800, max_area: int = 80_000) -> list[tuple]:
    """
    Keep contours whose bounding-box area falls within [min_area, max_area].

    min_area filters out residual noise after morphology.
    max_area filters out massive foreground blobs (trees, lighting changes).

    Returns:
        List of (x, y, w, h) tuples — bounding rectangles
    """
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            boxes.append(cv2.boundingRect(cnt))
    return boxes
