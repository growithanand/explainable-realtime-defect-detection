from io import BytesIO
from pathlib import Path
import sys

import streamlit as st
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.inference.predictor import DefectPredictor


MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_webcam_caps_finetuned_v2.pth"
SAMPLE_IMAGE_DIR = PROJECT_ROOT / "assets" / "sample_images"


def center_crop_square(image: Image.Image, crop_scale: float = 0.75) -> Image.Image:
    """
    Crops a centered square region from the uploaded image.

    This is useful for phone images because the model was trained mostly on
    fixed webcam ROI-style crops.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")

    width, height = image.size
    crop_size = int(min(width, height) * crop_scale)

    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    return image.crop((left, top, right, bottom))


def make_pretty_name(path: Path) -> str:
    """
    Converts file names like:
        defective_torn_red_cap.png

    into:
        🔴 Defective Torn Red Cap
    """
    name = path.stem.replace("_", " ").title()

    if path.stem.lower().startswith("defective"):
        return f"🔴 {name}"

    if path.stem.lower().startswith("normal"):
        return f"🟢 {name}"

    return name


def infer_expected_label(path: Path) -> str:
    """
    Infers the expected label from the sample image filename.

    Filenames should start with:
        normal_
        defective_
    """
    stem = path.stem.lower()

    if stem.startswith("defective"):
        return "defective"

    if stem.startswith("normal"):
        return "normal"

    return "unknown"


@st.cache_resource
def load_predictor():
    return DefectPredictor(
        model_path=MODEL_PATH,
        defect_threshold=0.5,
    )


def show_prediction(result, expected_label: str, threshold: float, input_mode: str):
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    normal_probability = result["normal_probability"]
    defective_probability = result["defective_probability"]

    st.subheader("Prediction Result")

    if predicted_class == "defective":
        st.error("Prediction: DEFECTIVE")
    else:
        st.success("Prediction: NORMAL")

    if expected_label != "unknown":
        if predicted_class == expected_label:
            st.success("Prediction matches the expected sample label.")
        else:
            st.warning("Prediction does not match the expected sample label.")

    st.metric("Confidence", f"{confidence:.2%}")

    st.write("### Class Probabilities")

    st.write(f"Normal probability: **{normal_probability:.2%}**")
    st.progress(normal_probability)

    st.write(f"Defective probability: **{defective_probability:.2%}**")
    st.progress(defective_probability)

    st.write("### Raw Output")

    st.json(
        {
            "predicted_class": predicted_class,
            "predicted_label": result["predicted_label"],
            "confidence": round(confidence, 4),
            "normal_probability": round(normal_probability, 4),
            "defective_probability": round(defective_probability, 4),
            "threshold": threshold,
            "input_mode": input_mode,
            "expected_label": expected_label,
            "device": result["device"],
        }
    )


st.set_page_config(
    page_title="Bottle Cap Defect Detection",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Bottle Cap Defect Detection")

st.write(
    "This demo uses a fine-tuned PyTorch model to classify bottle-cap inspection "
    "images as **normal** or **defective**."
)

st.warning(
    "This model is calibrated for a fixed webcam/ROI-style inspection setup. "
    "Arbitrary phone images may be out-of-distribution and can produce unreliable predictions."
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

predictor = load_predictor()

st.sidebar.header("Settings")

threshold = st.sidebar.slider(
    "Defect threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

predictor.defect_threshold = threshold

input_mode = st.radio(
    "Choose input mode",
    options=[
        "Try sample image",
        "Upload your own image",
    ],
)

image = None
original_image = None
image_caption = None
expected_label = "unknown"

if input_mode == "Try sample image":
    st.subheader("Try Curated Sample Images")

    sample_paths = sorted(
        [
            *SAMPLE_IMAGE_DIR.glob("*.png"),
            *SAMPLE_IMAGE_DIR.glob("*.jpg"),
            *SAMPLE_IMAGE_DIR.glob("*.jpeg"),
        ]
    )

    if len(sample_paths) == 0:
        st.error(
            f"No sample images found in: {SAMPLE_IMAGE_DIR}\n\n"
            "Create this folder and add sample images such as "
            "`normal_clean_cap.png` or `defective_cut_cap.png`."
        )
        st.stop()

    sample_labels = {make_pretty_name(path): path for path in sample_paths}

    placeholder = "-- Select a sample image --"

    selected_label = st.selectbox(
        "Select a sample image",
        options=[placeholder] + list(sample_labels.keys()),
        index=0,
    )

    if selected_label == placeholder:
        st.info("Please select a sample image from the dropdown.")

    else:
        selected_path = sample_labels[selected_label]

        image = Image.open(selected_path).convert("RGB")
        original_image = image
        image_caption = selected_path.name
        expected_label = infer_expected_label(selected_path)

        st.info(
            "These sample images are curated from the fixed inspection setup so users can "
            "test the model even without access to the webcam."
        )

elif input_mode == "Upload your own image":
    st.subheader("Upload Your Own Image")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
    )

    use_center_crop = st.checkbox(
        "Use center crop for prediction",
        value=True,
        help="Recommended for phone images. The model was trained on webcam ROI crops.",
    )

    crop_scale = st.slider(
        "Center crop scale",
        min_value=0.40,
        max_value=1.00,
        value=0.75,
        step=0.05,
        help="Lower value crops tighter around the center. Higher value keeps more of the image.",
    )

    if uploaded_file is not None:
        original_image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")

        if use_center_crop:
            image = center_crop_square(original_image, crop_scale=crop_scale)
        else:
            image = ImageOps.exif_transpose(original_image).convert("RGB")

        image_caption = uploaded_file.name
        expected_label = "unknown"

if image is not None:
    st.subheader("Input Image")

    if input_mode == "Upload your own image":
        st.write("Original uploaded image:")
        st.image(
            original_image,
            caption=image_caption,
            use_container_width=True,
        )

        st.write("Image region used for prediction:")
        st.image(
            image,
            caption="This is the image passed to the model.",
            use_container_width=True,
        )

    else:
        st.image(
            image,
            caption=image_caption,
            use_container_width=True,
        )

    if expected_label != "unknown":
        st.write(f"Expected sample label: **{expected_label.upper()}**")

    run_prediction = st.button(
        "Run Prediction",
        type="primary",
        use_container_width=True,
    )

    if run_prediction:
        with st.spinner("Running model inference..."):
            result = predictor.predict(image)

        show_prediction(
            result=result,
            expected_label=expected_label,
            threshold=threshold,
            input_mode=input_mode,
        )
    else:
        st.info("Click **Run Prediction** to classify this image.")

else:
    if input_mode == "Try sample image":
        st.info("Select a sample image to preview it.")
    else:
        st.info("Upload an image to preview it.")