# -*- coding: utf-8 -*-
"""
detect.py

Run inference with a trained YOLOv8 model on an image, folder of images,
video, or webcam, and optionally tag each detection with the frame quadrant
it falls in (handy if you're feeding detections to a skimmer/robot arm that
needs a rough steering direction).

Usage:
    # Single image or folder of images
    python src/detect.py --weights runs/detect/yolov8_plastic_waste_detection/weights/best.pt --source path/to/image_or_folder --conf 0.4

    # Video file
    python src/detect.py --weights best.pt --source path/to/video.mp4

    # Webcam
    python src/detect.py --weights best.pt --source 0

    # Add quadrant labels
    python src/detect.py --weights best.pt --source path/to/image.jpg --quadrants
"""

import argparse
import os

import cv2
from ultralytics import YOLO

from utils import get_quadrant


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference for plastic waste detection")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights (.pt)")
    parser.add_argument("--source", type=str, required=True,
                         help="Image path, folder path, video path, or webcam index (e.g. 0)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=str, default="results/predictions",
                         help="Directory to save annotated output")
    parser.add_argument("--quadrants", action="store_true",
                         help="Overlay a quadrant grid and label each detection's quadrant")
    parser.add_argument("--show", action="store_true", help="Display results in a window (local machines only)")
    return parser.parse_args()


def annotate_with_quadrants(frame, boxes, class_names):
    h, w = frame.shape[:2]
    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 255), 1)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        quad = get_quadrant(cx, cy, w, h)
        label = f"{class_names[cls_id]} {conf:.2f} ({quad})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = YOLO(args.weights)
    class_names = model.names

    source = int(args.source) if args.source.isdigit() else args.source

    if not args.quadrants:
        # Let Ultralytics handle saving/plotting natively — simplest path
        model.predict(
            source=source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            save=True,
            project=args.out,
            name="run",
            show=args.show,
        )
        print(f"Predictions saved under {os.path.join(args.out, 'run')}")
        return

    # Custom path: needed to draw quadrant labels ourselves
    results = model.predict(source=source, conf=args.conf, iou=args.iou,
                             imgsz=args.imgsz, stream=True)

    for i, result in enumerate(results):
        frame = result.orig_img.copy()
        frame = annotate_with_quadrants(frame, result.boxes, class_names)

        out_path = os.path.join(args.out, f"frame_{i:05d}.jpg")
        cv2.imwrite(out_path, frame)

        if args.show:
            cv2.imshow("Plastic Waste Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()
    print(f"Predictions saved to {args.out}")


if __name__ == "__main__":
    main()
