# -*- coding: utf-8 -*-
"""
train.py

Train a YOLOv8 model on the plastic waste detection dataset.

Usage:
    python src/train.py --data config/data.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16

After training, Ultralytics automatically saves to runs/detect/<name>/:
    - weights/best.pt, weights/last.pt
    - results.csv, results.png   (loss/mAP/precision/recall curves)
    - confusion_matrix.png
    - PR_curve.png, F1_curve.png, P_curve.png, R_curve.png
    - val_batch*.jpg              (predictions on validation images)

Those files are exactly what you should drop into the README's Results
section — they're the real thing, not mockups.
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for plastic waste detection")
    parser.add_argument("--data", type=str, default="config/data.yaml",
                         help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="Base model to fine-tune (yolov8n/s/m/l/x.pt), "
                              "or a path to a checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None,
                         help="e.g. '0' for GPU 0, 'cpu' for CPU. Auto-detected if omitted.")
    parser.add_argument("--name", type=str, default="yolov8_plastic_waste_detection",
                         help="Run name; results saved under runs/detect/<name>")
    parser.add_argument("--patience", type=int, default=20,
                         help="Early-stopping patience (epochs with no improvement)")
    parser.add_argument("--resume", action="store_true",
                         help="Resume training from the last checkpoint of --model")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=args.patience,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
