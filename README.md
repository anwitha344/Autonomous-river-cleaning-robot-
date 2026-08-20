# Plastic Waste on Water — Detection with YOLOv8

Real-time detection of floating plastic waste in water using a fine-tuned
[YOLOv8](https://github.com/ultralytics/ultralytics) object detector.

> ![Uploading image.png…]()

> [Unsplash](https://unsplash.com/s/photos/plastic-pollution-water) /
> [Pexels](https://www.pexels.com/search/ocean%20plastic/)).
> ```markdown
> ![banner](assets/banner.jpg)
> ```

---

## Overview

Plastic pollution in rivers, lakes, and oceans is a major environmental
problem, and manually monitoring it is slow and labor-intensive. This
project trains a YOLOv8 object detector to automatically locate plastic
waste in images and video frames, so that the output can feed into
downstream systems — e.g. alerting, mapping hotspots, or guiding an
autonomous skimmer/collection robot toward detected debris.

**Key features**
- Fine-tunes a pretrained YOLOv8 model on a custom plastic-waste dataset
- Scripts for dataset sanity-checking, training, evaluation, and inference
- Quadrant tagging: each detection is labeled with which quadrant of the
  frame it falls in (Q1–Q4), useful as a simple steering signal for a
  physical collection device
- Works on images, folders of images, video files, or a live webcam feed

---

## Example Detections

> Replace these with real screenshots from `src/detect.py` output
> (saved under `results/predictions/`) once you've trained your model.
> ```markdown
> ![sample detection 1](assets/sample_detection_1.jpg)
> ![sample detection 2](assets/sample_detection_2.jpg)
> ```

---

## Project Structure

```
plastic-waste-detection/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── config/
│   └── data.yaml              # dataset config (paths + class names)
├── src/
│   ├── utils.py                # shared helpers (box conversion, quadrants, drawing)
│   ├── visualize_dataset.py    # sanity-check dataset labels before training
│   ├── train.py                # fine-tune YOLOv8 on your dataset
│   ├── evaluate.py             # compute mAP / precision / recall on val or test split
│   └── detect.py                # run inference on images / video / webcam
├── assets/                      # README images go here
└── results/                     # created at runtime: predictions + eval metrics
```

---

## Dataset

This project expects a dataset exported in **YOLOv8 format** (e.g. from
[Roboflow](https://roboflow.com/)), with the standard structure:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```


Update [`config/data.yaml`](config/data.yaml) to point at your dataset's
location and list your actual class names.


