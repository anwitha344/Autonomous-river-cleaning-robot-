# Autonomous River Cleaning Robot

A computer vision system for detecting floating plastic waste and determining its location for use in an autonomous river cleaning robot.

## Final Architecture

```text
Camera
   |
   v
Video / Image Frame
   |
   v
YOLOv8 Plastic Detection
   |
   +----------------+
   |                |
   v                v
Bounding Box     Confidence
   |
   v
Object Center
   |
   v
Quadrant Detection
   |
   v
Navigation Controller
   |
   +------------+
   |            |
   v            v
Motor Control  Collection Mechanism
   |            |
   +------+-----+
          |
          v
   Plastic Collection
```

## Model

The project uses **YOLOv8** with transfer learning.

A pretrained YOLOv8 model is fine tuned on an annotated plastic waste dataset containing one class:

```text
0: plastic
```

The model outputs:

```text
Class
Confidence
Bounding Box
```

The bounding box center is then used to determine which quadrant of the camera frame contains the detected plastic.

```text
+-------------------+-------------------+
|        Q1         |        Q2         |
|                   |                   |
+-------------------+-------------------+
|        Q3         |        Q4         |
|                   |                   |
+-------------------+-------------------+
```

This provides a simple directional signal for the robot.

## Project Workflow

1. Prepare the plastic waste dataset in YOLO format.
2. Configure dataset paths and classes in `data.yaml`.
3. Visualize annotations using `visualize_dataset.py`.
4. Load pretrained YOLOv8 weights.
5. Fine tune the model using `train.py`.
6. Save the best trained checkpoint.
7. Evaluate the model using `evaluate.py`.
8. Calculate Precision, Recall, mAP50 and mAP50 to 95.
9. Run inference using `detect.py`.
10. Detect plastic from images, videos or a webcam.
11. Calculate the center of each detected bounding box.
12. Assign the detected plastic to a quadrant.
13. Pass the spatial information to the robot navigation system.

## Repository Structure

```text
Autonomous River Cleaning Robot
|
+-- data.yaml
+-- train.py
+-- evaluate.py
+-- detect.py
+-- visualize_dataset.py
+-- utils.py
+-- requirements.txt
+-- README.md
```

## Main Components

### `train.py`

Trains the YOLOv8 model using the plastic waste dataset.

Default configuration:

```text
Model: YOLOv8n
Epochs: 100
Image Size: 640
Batch Size: 16
Early Stopping: 20 epochs
```

### `evaluate.py`

Evaluates the trained model using:

```text
Precision
Recall
mAP50
mAP50 to 95
```

### `detect.py`

Runs the trained model on:

```text
Images
Videos
Webcam
```
![Uploading image.png…]()


It also performs quadrant detection for robotic navigation.

### `visualize_dataset.py`

Displays annotated dataset samples to verify that the bounding boxes and labels are correct before training.

### `utils.py`

Contains utilities for:

```text
YOLO coordinate conversion
Class loading
Bounding box visualization
Quadrant calculation
```
