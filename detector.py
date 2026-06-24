from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
from PIL import Image


def _to_cv_gray(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _threshold_and_invert(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return binary


def _merge_strokes(binary: np.ndarray, merge_gap: int) -> np.ndarray:
    gap = max(1, int(merge_gap))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def _clamp_box(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
    padding: int,
) -> dict:
    pad = max(0, int(padding))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image_width, x + width + pad)
    y2 = min(image_height, y + height + pad)
    clamped_width = max(0, x2 - x1)
    clamped_height = max(0, y2 - y1)
    return {
        "x": int(x1),
        "y": int(y1),
        "width": int(clamped_width),
        "height": int(clamped_height),
        "area": int(clamped_width * clamped_height),
    }


def sort_boxes(boxes: Iterable[dict]) -> list[dict]:
    boxes = list(boxes)
    if not boxes:
        return []

    median_height = float(np.median([box["height"] for box in boxes]))
    row_tolerance = max(1.0, median_height * 0.75)
    rows: list[list[dict]] = []

    for box in sorted(boxes, key=lambda item: int(item["y"])):
        placed = False
        for row in rows:
            row_y = float(np.mean([item["y"] for item in row]))
            if abs(int(box["y"]) - row_y) <= row_tolerance:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    sorted_rows = sorted(rows, key=lambda row: min(int(item["y"]) for item in row))
    sorted_boxes = []
    for row in sorted_rows:
        sorted_boxes.extend(sorted(row, key=lambda item: int(item["x"])))
    return sorted_boxes


def detect_elements(
    image: Image.Image,
    min_area: int = 500,
    merge_gap: int = 8,
    padding: int = 10,
) -> list[dict]:
    gray = _to_cv_gray(image)
    binary = _threshold_and_invert(gray)
    merged = _merge_strokes(binary, merge_gap)
    image_height, image_width = gray.shape[:2]

    contours, _ = cv2.findContours(
        merged,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        raw_area = width * height
        if raw_area < int(min_area):
            continue
        boxes.append(
            _clamp_box(
                x,
                y,
                width,
                height,
                image_width,
                image_height,
                padding,
            )
        )

    return sort_boxes(boxes)
