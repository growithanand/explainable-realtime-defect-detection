from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from PIL import Image

from src.inference.predictor import DefectPredictor


MODEL_PATH = Path("models/resnet18_webcam_caps_finetuned_v2.pth")

app = FastAPI(
    title="Bottle Cap Defect Detection API",
    description="FastAPI service for image-based defect detection using a fine-tuned PyTorch model.",
    version="0.1.0",
)

predictor: Optional[DefectPredictor] = None


@app.on_event("startup")
def load_model():
    """
    Load the model once when the API starts.

    This is important because we do not want to reload the model
    for every prediction request.
    """
    global predictor

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    predictor = DefectPredictor(
        model_path=MODEL_PATH,
        defect_threshold=0.5,
    )

    print(f"Model loaded from: {MODEL_PATH}")
    print(f"Using device: {predictor.device}")


@app.get("/")
def root():
    return {
        "message": "Bottle Cap Defect Detection API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Defective probability threshold.",
    ),
):
    """
    Upload an image and get normal/defective prediction.
    """

    if predictor is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image file: {error}",
        )

    try:
        predictor.defect_threshold = threshold
        result = predictor.predict(image)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        )

    return {
        "filename": file.filename,
        "predicted_class": result["predicted_class"],
        "predicted_label": result["predicted_label"],
        "confidence": round(result["confidence"], 4),
        "normal_probability": round(result["normal_probability"], 4),
        "defective_probability": round(result["defective_probability"], 4),
        "threshold": result["defect_threshold"],
        "device": result["device"],
    }