import argparse
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from src.inference.predictor import DefectPredictor
from src.visualization.gradcam import GradCAM, overlay_cam_on_image


MODEL_PATH = "models/resnet18_webcam_caps_finetuned_v2.pth"
OUTPUT_DIR = Path("outputs/webcam_captures")


def draw_label(frame, text, position, color, scale=0.75, thickness=2):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def get_center_roi(frame, roi_scale=0.65):
    height, width = frame.shape[:2]
    roi_size = int(min(height, width) * roi_scale)

    x1 = (width - roi_size) // 2
    y1 = (height - roi_size) // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    roi = frame[y1:y2, x1:x2]

    return roi, (x1, y1, x2, y2)


def is_object_present(roi, edge_threshold=0.003, std_threshold=10.0):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)

    edge_ratio = np.count_nonzero(edges) / edges.size
    contrast = float(np.std(gray))

    object_present = edge_ratio > edge_threshold and contrast > std_threshold

    return object_present, edge_ratio, contrast


def get_smoothed_prediction(prediction_history, stability_ratio=0.7):
    if len(prediction_history) == 0:
        return None, 0.0

    counts = Counter(prediction_history)
    most_common_class, count = counts.most_common(1)[0]

    agreement = count / len(prediction_history)

    if agreement >= stability_ratio:
        return most_common_class, agreement

    return None, agreement


def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    return cap


def generate_gradcam_overlay(predictor, grad_cam, roi_bgr, target_label):
    """
    Generates Grad-CAM overlay for the current ROI.

    roi_bgr: OpenCV BGR image
    target_label:
        0 = normal
        1 = defective
    """

    pil_image = predictor._prepare_image(
        roi_bgr,
        input_format="bgr",
    )

    input_tensor = predictor.transform(pil_image)
    input_tensor = input_tensor.unsqueeze(0).to(predictor.device)

    cam, _ = grad_cam.generate(
        input_tensor=input_tensor,
        target_class=target_label,
    )

    overlay_rgb = overlay_cam_on_image(
        image=pil_image,
        cam=cam,
        alpha=0.45,
    )

    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    return overlay_bgr


def main():
    parser = argparse.ArgumentParser(
        description="Real-time industrial defect detection webcam demo."
    )

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--roi-scale", type=float, default=0.65)

    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--smoothing-window", type=int, default=10)
    parser.add_argument("--stability-ratio", type=float, default=0.7)

    parser.add_argument("--presence-edge-threshold", type=float, default=0.003)
    parser.add_argument("--presence-std-threshold", type=float, default=10.0)

    parser.add_argument(
        "--heatmap-duration",
        type=float,
        default=4.0,
        help="Seconds to display Grad-CAM overlay after pressing h.",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictor = DefectPredictor(
        model_path=MODEL_PATH,
        defect_threshold=args.threshold,
    )

    # Grad-CAM target layer for ResNet18
    target_layer = predictor.model.layer4[-1]
    grad_cam = GradCAM(
        model=predictor.model,
        target_layer=target_layer,
    )

    cap = open_camera(args.camera)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera with index {args.camera}")

    print("Webcam demo started.")
    print("Controls:")
    print("  q = quit")
    print("  s = save current frame")
    print("  h = generate Grad-CAM heatmap for current ROI")
    print("  + = increase defect threshold")
    print("  - = decrease defect threshold")

    current_threshold = args.threshold
    prediction_history = deque(maxlen=args.smoothing_window)

    frame_count = 0
    prev_time = time.time()
    last_result = None
    last_roi = None

    heatmap_overlay = None
    heatmap_until = 0
    heatmap_label = None

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Failed to read frame from camera.")
                break

            frame_count += 1

            roi, (x1, y1, x2, y2) = get_center_roi(
                frame,
                roi_scale=args.roi_scale,
            )

            last_roi = roi.copy()

            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-8)
            prev_time = current_time

            box_color = (255, 255, 255)
            status_text = "WAITING"
            confidence_text = "Confidence: -"
            probability_text = "P(normal): - | P(defective): -"
            presence_text = "Object: -"
            heatmap_text = "Heatmap: press h"

            if frame_count <= args.warmup_frames:
                status_text = "WARMING UP CAMERA"
                prediction_history.clear()
                heatmap_overlay = None

            else:
                object_present, edge_ratio, contrast = is_object_present(
                    roi,
                    edge_threshold=args.presence_edge_threshold,
                    std_threshold=args.presence_std_threshold,
                )

                presence_text = (
                    f"Object: {object_present} | "
                    f"Edges: {edge_ratio:.3f} | "
                    f"Contrast: {contrast:.1f}"
                )

                if not object_present:
                    status_text = "WAITING FOR OBJECT"
                    box_color = (255, 255, 255)
                    prediction_history.clear()
                    heatmap_overlay = None

                else:
                    predictor.defect_threshold = current_threshold

                    result = predictor.predict(
                        roi,
                        input_format="bgr",
                    )

                    last_result = result

                    raw_prediction = result["predicted_class"]
                    confidence = result["confidence"]
                    defective_probability = result["defective_probability"]
                    normal_probability = result["normal_probability"]

                    prediction_history.append(raw_prediction)

                    stable_prediction, agreement = get_smoothed_prediction(
                        prediction_history,
                        stability_ratio=args.stability_ratio,
                    )

                    if stable_prediction is None:
                        status_text = "ANALYZING..."
                        box_color = (0, 255, 255)

                    elif stable_prediction == "defective":
                        status_text = "DEFECTIVE"
                        box_color = (0, 0, 255)

                    else:
                        status_text = "NORMAL"
                        box_color = (0, 200, 0)

                    confidence_text = (
                        f"Confidence: {confidence:.2f} | "
                        f"Agreement: {agreement:.2f}"
                    )

                    probability_text = (
                        f"P(normal): {normal_probability:.2f} | "
                        f"P(defective): {defective_probability:.2f}"
                    )

            # Display Grad-CAM overlay if active
            if heatmap_overlay is not None and time.time() < heatmap_until:
                roi_h, roi_w = y2 - y1, x2 - x1
                overlay_resized = cv2.resize(heatmap_overlay, (roi_w, roi_h))
                frame[y1:y2, x1:x2] = overlay_resized

                if heatmap_label is not None:
                    heatmap_text = f"Heatmap: {heatmap_label}"

            elif heatmap_overlay is not None and time.time() >= heatmap_until:
                heatmap_overlay = None
                heatmap_label = None

            # Draw inspection ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

            # Draw top info panel
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 220), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

            draw_label(
                frame,
                f"Status: {status_text}",
                (20, 35),
                box_color,
                scale=0.9,
                thickness=2,
            )

            draw_label(
                frame,
                confidence_text,
                (20, 70),
                (255, 255, 255),
            )

            draw_label(
                frame,
                probability_text,
                (20, 100),
                (255, 255, 255),
            )

            draw_label(
                frame,
                presence_text,
                (20, 130),
                (255, 255, 255),
                scale=0.65,
            )

            draw_label(
                frame,
                f"FPS: {fps:.1f} | Threshold: {current_threshold:.2f}",
                (20, 160),
                (255, 255, 255),
            )

            draw_label(
                frame,
                heatmap_text,
                (20, 190),
                (255, 255, 255),
                scale=0.65,
            )

            draw_label(
                frame,
                "q: quit | s: save | h: heatmap | +/-: threshold",
                (20, 215),
                (200, 200, 200),
                scale=0.6,
                thickness=1,
            )

            cv2.imshow("Real-Time Defect Inspection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = OUTPUT_DIR / f"capture_{timestamp}_{status_text.lower()}.png"
                cv2.imwrite(str(save_path), frame)
                print(f"Saved frame to: {save_path}")

            elif key == ord("h"):
                if last_roi is None:
                    print("No ROI available for Grad-CAM.")
                    continue

                if last_result is None:
                    print("No prediction available yet. Place object in ROI first.")
                    continue

                target_label = last_result["predicted_label"]
                target_class = last_result["predicted_class"]

                print(f"Generating Grad-CAM for class: {target_class}")

                heatmap_overlay = generate_gradcam_overlay(
                    predictor=predictor,
                    grad_cam=grad_cam,
                    roi_bgr=last_roi,
                    target_label=target_label,
                )

                heatmap_until = time.time() + args.heatmap_duration
                heatmap_label = target_class

            elif key == ord("+") or key == ord("="):
                current_threshold = min(0.95, current_threshold + 0.05)
                prediction_history.clear()
                heatmap_overlay = None
                print(f"Threshold increased to: {current_threshold:.2f}")

            elif key == ord("-") or key == ord("_"):
                current_threshold = max(0.05, current_threshold - 0.05)
                prediction_history.clear()
                heatmap_overlay = None
                print(f"Threshold decreased to: {current_threshold:.2f}")

    finally:
        grad_cam.remove_hooks()
        cap.release()
        cv2.destroyAllWindows()

    print("Webcam demo stopped.")

    if last_result:
        print("\nLast prediction:")
        print(f"Class: {last_result['predicted_class']}")
        print(f"Confidence: {last_result['confidence']:.4f}")
        print(f"Normal probability: {last_result['normal_probability']:.4f}")
        print(f"Defective probability: {last_result['defective_probability']:.4f}")


if __name__ == "__main__":
    main()