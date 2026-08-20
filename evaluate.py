# -*- coding: utf-8 -*-
"""
evaluate.py

Run validation metrics (mAP50, mAP50-95, precision, recall) for a trained
checkpoint against a dataset split, and print/save them.

Usage:
    python src/evaluate.py --weights runs/detect/yolov8_plastic_waste_detection/weights/best.pt --data config/data.yaml
"""

import argparse
import json
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 checkpoint")
    parser.add_argument("--weights", type=str, required=True,
                         help="Path to trained weights (e.g. best.pt)")
    parser.add_argument("--data", type=str, default="config/data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=str, default="results/eval_metrics.json",
                         help="Where to save the metrics summary as JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz)

    summary = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"{k:12s}: {v:.4f}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {args.out}")


if __name__ == "__main__":
    main()
