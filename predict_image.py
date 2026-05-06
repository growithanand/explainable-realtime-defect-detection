import argparse
from pathlib import Path

from src.inference.predictor import DefectPredictor


MODEL_PATH = "models/resnet18_bottle_binary.pth"


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether an industrial image is normal or defective."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Defective probability threshold. Default is 0.5.",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    predictor = DefectPredictor(
        model_path=MODEL_PATH,
        defect_threshold=args.threshold,
    )

    result = predictor.predict(image_path)

    print("\nPrediction Result")
    print("-" * 30)
    print(f"Image path           : {image_path}")
    print(f"Predicted class      : {result['predicted_class']}")
    print(f"Confidence           : {result['confidence']:.4f}")
    print(f"Normal probability   : {result['normal_probability']:.4f}")
    print(f"Defective probability: {result['defective_probability']:.4f}")
    print(f"Defect threshold     : {result['defect_threshold']}")
    print(f"Device               : {result['device']}")


if __name__ == "__main__":
    main()
    