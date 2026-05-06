import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2


OUTPUT_ROOT = Path("data/webcam_bottle_caps")


def get_center_roi(frame, roi_scale=0.65):
    height, width = frame.shape[:2]

    roi_size = int(min(height, width) * roi_scale)

    x1 = (width - roi_size) // 2
    y1 = (height - roi_size) // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    roi = frame[y1:y2, x1:x2]

    return roi, (x1, y1, x2, y2)


def draw_text(frame, text, position, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Capture webcam images for bottle cap defect training."
    )

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--roi-scale", type=float, default=0.65)
    parser.add_argument(
        "--save-interval",
        type=float,
        default=0.25,
        help="Seconds between saved frames while recording.",
    )

    args = parser.parse_args()

    normal_dir = OUTPUT_ROOT / "normal"
    defective_dir = OUTPUT_ROOT / "defective"

    normal_dir.mkdir(parents=True, exist_ok=True)
    defective_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_ANY)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    recording_label = None
    last_save_time = 0

    print("Webcam data capture started.")
    print("Controls:")
    print("  n = start/stop recording NORMAL images")
    print("  d = start/stop recording DEFECTIVE images")
    print("  s = save one frame manually")
    print("  q = quit")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Failed to read frame.")
            break

        roi, (x1, y1, x2, y2) = get_center_roi(frame, roi_scale=args.roi_scale)

        display = frame.copy()

        if recording_label == "normal":
            box_color = (0, 200, 0)
            status = "RECORDING NORMAL"
        elif recording_label == "defective":
            box_color = (0, 0, 255)
            status = "RECORDING DEFECTIVE"
        else:
            box_color = (255, 255, 255)
            status = "NOT RECORDING"

        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 3)

        draw_text(display, status, (20, 35), box_color)
        draw_text(display, "n: normal | d: defective | s: save once | q: quit", (20, 70))
        draw_text(display, "Move/rotate object slightly while recording.", (20, 105))

        current_time = time.time()

        if recording_label is not None:
            if current_time - last_save_time >= args.save_interval:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                if recording_label == "normal":
                    save_path = normal_dir / f"normal_{timestamp}.png"
                else:
                    save_path = defective_dir / f"defective_{timestamp}.png"

                cv2.imwrite(str(save_path), roi)
                last_save_time = current_time

        cv2.imshow("Bottle Cap Data Capture", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("n"):
            if recording_label == "normal":
                recording_label = None
                print("Stopped recording normal.")
            else:
                recording_label = "normal"
                print("Started recording normal.")

        elif key == ord("d"):
            if recording_label == "defective":
                recording_label = None
                print("Stopped recording defective.")
            else:
                recording_label = "defective"
                print("Started recording defective.")

        elif key == ord("s"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            if recording_label == "normal":
                save_path = normal_dir / f"normal_manual_{timestamp}.png"
            elif recording_label == "defective":
                save_path = defective_dir / f"defective_manual_{timestamp}.png"
            else:
                save_path = OUTPUT_ROOT / f"unlabeled_{timestamp}.png"

            cv2.imwrite(str(save_path), roi)
            print(f"Saved one frame to: {save_path}")

    cap.release()
    cv2.destroyAllWindows()

    print("Capture stopped.")
    print(f"Normal images saved to: {normal_dir}")
    print(f"Defective images saved to: {defective_dir}")


if __name__ == "__main__":
    main()