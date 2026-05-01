import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.inference.predictor import DefectPredictor
from src.models.model import load_model
from src.utils.helpers import get_device
from src.visualization.gradcam import dummy_heatmap, overlay_heatmap


st.set_page_config(page_title="Explainable Defect Detection", layout="wide")
st.title("Explainable Real-Time Industrial Defect Detection")

st.write(
    "Upload an image to get a normal/defective prediction. "
    "Grad-CAM will be implemented later; this app currently includes "
    "a placeholder heatmap for UI testing."
)

checkpoint_path = st.sidebar.text_input("Checkpoint path", "models/best_model.pth")
image_size = st.sidebar.number_input("Image size", min_value=128, max_value=512, value=224, step=32)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    frame_rgb = np.array(image)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input image")
        st.image(image, use_container_width=True)

    try:
        device = get_device()
        model = load_model(checkpoint_path, device=device, num_classes=2)
        predictor = DefectPredictor(
            model=model,
            device=device,
            class_names=["normal", "defective"],
            image_size=image_size,
        )
        label, confidence = predictor.predict_frame(frame_bgr)

        heatmap = dummy_heatmap(frame_bgr)
        overlay_bgr = overlay_heatmap(frame_bgr, heatmap)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Prediction + explanation overlay")
            st.image(overlay_rgb, use_container_width=True)
            st.metric("Prediction", label)
            st.metric("Confidence", f"{confidence:.2f}")

    except FileNotFoundError:
        st.error("Checkpoint not found. Train the model first or update the checkpoint path.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("Upload an image to test the app.")
