import argparse
import time

import cv2

from src.inference.predictor import DefectPredictor
from src.models.model import load_model
from src.utils.helpers import get_device


def draw_overlay(frame, label: str, confidence: float, fps: float):
    color = (0, 255, 0) if label.lower() == "normal" else (0, 0, 255)
    text = f"{label.upper()} | conf: {confidence:.2f} | FPS: {fps:.1f}"
    cv2.rectangle(frame, (10, 10), (760, 60), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device=device, num_classes=2)
    class_names = ["normal", "defective"]
    predictor = DefectPredictor(model=model, device=device, class_names=class_names, image_size=args.image_size)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam/video source.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, confidence = predictor.predict_frame(frame)

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-8)
        prev_time = current_time

        frame = draw_overlay(frame, label, confidence, fps)
        cv2.imshow("Real-Time Defect Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time defect detection webcam demo")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source", type=int, default=0, help="0 for webcam")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()
    main(args)
