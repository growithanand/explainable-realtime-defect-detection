# Explainable Real-Time Industrial Defect Detection using PyTorch and OpenCV

This project is a portfolio-ready computer vision system for manufacturing/automotive inspection. It aims to detect whether an industrial part is **normal** or **defective** and explain the model decision using heatmaps.

## Project goals

- Train a PyTorch model for industrial defect classification.
- Run real-time inference on webcam/video using OpenCV.
- Display prediction, confidence score, and FPS.
- Add explainability using Grad-CAM heatmaps.
- Keep the project clean, modular, and deployable later.

## Initial version

```text
Image or webcam frame
→ OpenCV preprocessing
→ PyTorch model
→ normal/defective prediction
→ confidence + FPS overlay
```

## Extended version

```text
Image/video input
→ model prediction
→ Grad-CAM heatmap
→ defect explanation overlay
→ Streamlit/FastAPI/Docker/ONNX extensions
```

## Suggested dataset

Start with one category from the MVTec AD dataset, for example:

- bottle
- metal_nut
- capsule
- hazelnut
- screw

For the first version, frame the task as binary classification:

```text
good → normal
all defect folders → defective
```

## Folder structure

```text
explainable-realtime-defect-detection/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── outputs/
│   ├── predictions/
│   └── heatmaps/
│
├── reports/
│   └── figures/
│
├── src/
│   ├── data/
│   │   └── dataset.py
│   ├── inference/
│   │   └── predictor.py
│   ├── models/
│   │   └── model.py
│   ├── utils/
│   │   ├── config.py
│   │   └── helpers.py
│   └── visualization/
│       └── gradcam.py
│
├── train.py
├── evaluate.py
├── webcam_demo.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch installation fails, install it from the official PyTorch command for your system first, then rerun:

```bash
pip install -r requirements.txt
```

## First milestone

1. Download one MVTec category.
2. Create a processed binary dataset:

```text
data/processed/bottle/train/normal/
data/processed/bottle/train/defective/
data/processed/bottle/test/normal/
data/processed/bottle/test/defective/
```

3. Train a ResNet18 classifier.
4. Save the model checkpoint to `models/`.
5. Run `webcam_demo.py` for live predictions.

## Commands

Train:

```bash
python train.py --data_dir data/processed/bottle --epochs 10 --batch_size 16
```

Evaluate:

```bash
python evaluate.py --data_dir data/processed/bottle --checkpoint models/best_model.pth
```

Webcam demo:

```bash
python webcam_demo.py --checkpoint models/best_model.pth
```

Streamlit app:

```bash
streamlit run app/streamlit_app.py
```
