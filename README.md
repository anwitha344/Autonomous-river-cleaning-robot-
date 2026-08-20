# Plastic Waste on Water — Detection with YOLOv8

Real-time detection of floating plastic waste in water using a fine-tuned
[YOLOv8](https://github.com/ultralytics/ultralytics) object detector.

> 🖼️ **Add a banner image here.** A wide photo of plastic debris in water
> works well (e.g. your own dataset sample, or a free photo from
> [Unsplash](https://unsplash.com/s/photos/plastic-pollution-water) /
> [Pexels](https://www.pexels.com/search/ocean%20plastic/)).
> ```markdown
> ![banner](assets/banner.jpg)
> ```

---

## 📌 Overview

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

## 🖼️ Example Detections

> Replace these with real screenshots from `src/detect.py` output
> (saved under `results/predictions/`) once you've trained your model.
> ```markdown
> ![sample detection 1](assets/sample_detection_1.jpg)
> ![sample detection 2](assets/sample_detection_2.jpg)
> ```

---

## 📁 Project Structure

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

## 🧰 Dataset

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

If you don't have your own dataset yet, Roboflow Universe hosts several
public "plastic waste on water" datasets you can fork and export in YOLOv8
format.

Update [`config/data.yaml`](config/data.yaml) to point at your dataset's
location and list your actual class names.

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/plastic-waste-detection.git
cd plastic-waste-detection

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Requires Python 3.9+. A GPU is strongly recommended for training (CPU works
but is much slower); inference runs fine on CPU.

---

## 🚀 Usage

### 1. Sanity-check the dataset (optional but recommended)

```bash
python src/visualize_dataset.py --data config/data.yaml --split train --n 20 --quadrants
```

### 2. Train

```bash
python src/train.py \
  --data config/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

`yolov8n.pt` (nano) trains fastest; swap in `yolov8s.pt` / `yolov8m.pt` for
higher accuracy at the cost of speed. Ultralytics downloads the pretrained
base weights automatically on first run.

Training outputs land in `runs/detect/<name>/`, including:
- `weights/best.pt`, `weights/last.pt`
- `results.png` — loss / mAP / precision / recall curves over training
- `confusion_matrix.png`
- `PR_curve.png`, `F1_curve.png`, `P_curve.png`, `R_curve.png`
- `val_batch*.jpg` — sample predictions on validation images

### 3. Evaluate

```bash
python src/evaluate.py \
  --weights runs/detect/yolov8_plastic_waste_detection/weights/best.pt \
  --data config/data.yaml \
  --split test
```

### 4. Run inference

```bash
# Single image
python src/detect.py --weights best.pt --source path/to/image.jpg --conf 0.4

# Folder of images
python src/detect.py --weights best.pt --source path/to/folder/

# Video
python src/detect.py --weights best.pt --source path/to/video.mp4

# Webcam
python src/detect.py --weights best.pt --source 0

# With quadrant tagging (for steering a skimmer/robot toward debris)
python src/detect.py --weights best.pt --source path/to/image.jpg --quadrants
```

---

## 📊 Results

> ⚠️ **Fill this in with your own numbers after training** — copy the
> values printed by `evaluate.py` and the plots from your
> `runs/detect/<name>/` folder. Don't publish placeholder/fabricated
> numbers here; an empty table is more trustworthy than a fake one.

| Metric        | Value |
|---------------|-------|
| mAP50         |       |
| mAP50-95      |       |
| Precision     |       |
| Recall        |       |
| Training time |       |
| Epochs        |       |

```markdown
![training curves](assets/results.png)
![confusion matrix](assets/confusion_matrix.png)
![PR curve](assets/PR_curve.png)
```

---

## 🔭 Future Work

- Deploy as a real-time pipeline on a Raspberry Pi / Jetson Nano mounted on
  a floating skimmer
- Expand class list beyond generic "plastic" (e.g. bottles, bags, foam,
  fishing nets) for material-specific sorting
- Add tracking (e.g. ByteTrack) across video frames to avoid double-counting
  the same object
- Export to ONNX / TensorRT for faster edge inference

---

## 📜 License

This project is released under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Dataset export format courtesy of [Roboflow](https://roboflow.com/)
