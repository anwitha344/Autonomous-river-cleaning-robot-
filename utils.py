# -*- coding: utf-8 -*-
"""
Shared helper functions used across the training, inference, and
visualization scripts.
"""

import os
import cv2
import yaml


def load_class_names(yaml_path):
    """Load class names from a YOLO data.yaml file.

    Args:
        yaml_path (str): Path to data.yaml

    Returns:
        dict[int, str]: mapping of class index -> class name
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    names = data["names"]
    # Roboflow/Ultralytics yaml can store names as a list or a dict
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return names


def yolo_to_pixel_box(x, y, bw, bh, img_w, img_h):
    """Convert a normalized YOLO box (x_center, y_center, w, h) to pixel
    coordinates (x1, y1, x2, y2)."""
    x1 = int((x - bw / 2) * img_w)
    y1 = int((y - bh / 2) * img_h)
    x2 = int((x + bw / 2) * img_w)
    y2 = int((y + bh / 2) * img_h)
    return x1, y1, x2, y2


def get_quadrant(center_x, center_y, img_w, img_h):
    """Return which quadrant (Q1-Q4) a point falls into.

    Q1 = top-left, Q2 = top-right, Q3 = bottom-left, Q4 = bottom-right.
    Useful for downstream tasks such as steering a collection robot/skimmer
    toward the region of the frame containing detected waste.
    """
    if center_x < img_w // 2 and center_y < img_h // 2:
        return "Q1"
    elif center_x >= img_w // 2 and center_y < img_h // 2:
        return "Q2"
    elif center_x < img_w // 2 and center_y >= img_h // 2:
        return "Q3"
    else:
        return "Q4"


def draw_ground_truth_boxes(image, label_path, class_names, show_quadrants=False):
    """Draw YOLO-format ground-truth label boxes onto an image (in-place-ish,
    returns the modified image). Skips malformed label lines instead of
    crashing, since exported datasets sometimes contain empty/bad lines.
    """
    h, w = image.shape[:2]

    if show_quadrants:
        cv2.line(image, (0, h // 2), (w, h // 2), (255, 255, 255), 1)
        cv2.line(image, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)

    if not os.path.exists(label_path):
        return image

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            cls, x, y, bw, bh = map(float, line.split())
        except ValueError:
            continue

        cls = int(cls)
        x1, y1, x2, y2 = yolo_to_pixel_box(x, y, bw, bh, w, h)

        label = class_names.get(cls, str(cls))
        if show_quadrants:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            label = f"{label} ({get_quadrant(cx, cy, w, h)})"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, label, (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    return image
