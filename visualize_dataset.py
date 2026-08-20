# -*- coding: utf-8 -*-
"""
visualize_dataset.py

Sanity-check a YOLO-format dataset by drawing ground-truth boxes on a random
sample of training images. Useful for catching label/export issues before
you spend time training.

Usage:
    python src/visualize_dataset.py --data config/data.yaml --split train --n 20
"""

import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import yaml

from utils import load_class_names, draw_ground_truth_boxes


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize YOLO dataset labels")
    parser.add_argument("--data", type=str, default="config/data.yaml",
                         help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="train",
                         choices=["train", "val", "test"],
                         help="Which split to sample from")
    parser.add_argument("--n", type=int, default=20,
                         help="Number of sample images to show")
    parser.add_argument("--quadrants", action="store_true",
                         help="Overlay quadrant grid + label each box with its quadrant")
    parser.add_argument("--out", type=str, default=None,
                         help="Optional path to save the figure instead of / as well as showing it")
    return parser.parse_args()


def resolve_split_dirs(data_yaml_path, split):
    with open(data_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    root = cfg.get("path", ".")
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(data_yaml_path)), root)

    split_key = {"train": "train", "val": "val", "test": "test"}[split]
    images_rel = cfg[split_key]
    images_dir = os.path.join(root, images_rel)
    labels_dir = images_dir.replace("images", "labels")
    return images_dir, labels_dir


def main():
    args = parse_args()
    class_names = load_class_names(args.data)
    images_dir, labels_dir = resolve_split_dirs(args.data, args.split)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Could not find images directory: {images_dir}\n"
            "Check the 'path' and split keys in your data.yaml."
        )

    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        raise RuntimeError(f"No images found in {images_dir}")

    n = min(args.n, len(image_files))
    sample = random.sample(image_files, n)

    cols = 4
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 5 * rows))

    for i, img_file in enumerate(sample):
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(
            labels_dir,
            os.path.splitext(img_file)[0] + ".txt",
        )

        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = draw_ground_truth_boxes(image, label_path, class_names,
                                         show_quadrants=args.quadrants)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(img_file, fontsize=8)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
