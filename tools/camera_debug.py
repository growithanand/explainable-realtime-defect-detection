import cv2
import time
import numpy as np


BACKENDS = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}

RESOLUTIONS = [
    (640, 480),
    (1280, 720),
    (320, 240),
]


def test_camera(camera_index, backend_name, backend, width, height):
    print(f"\nTesting camera={camera_index}, backend={backend_name}, resolution={width}x{height}")

    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        print("  Could not open camera.")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Warm up camera
    time.sleep(1)

    success = False

    for i in range(30):
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        mean_value = np.mean(frame)
        std_value = np.std(frame)

        print(f"  Frame read. Mean={mean_value:.2f}, Std={std_value:.2f}")

        if mean_value > 5 and std_value > 5:
            print("  Looks like a valid camera feed.")
            success = True

            while True:
                cv2.putText(
                    frame,
                    f"camera={camera_index}, backend={backend_name}, {width}x{height}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Working Camera Feed - Press q", frame)

                ret, frame = cap.read()
                if not ret:
                    break

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            break

    cap.release()
    cv2.destroyAllWindows()

    if not success:
        print("  No valid frame received.")

    return success


def main():
    for camera_index in range(5):
        for backend_name, backend in BACKENDS.items():
            for width, height in RESOLUTIONS:
                success = test_camera(
                    camera_index=camera_index,
                    backend_name=backend_name,
                    backend=backend,
                    width=width,
                    height=height,
                )

                if success:
                    print("\nSUCCESS")
                    print(f"Use camera index: {camera_index}")
                    print(f"Use backend: {backend_name}")
                    print(f"Use resolution: {width}x{height}")
                    return

    print("\nNo working camera configuration found.")


if __name__ == "__main__":
    main()