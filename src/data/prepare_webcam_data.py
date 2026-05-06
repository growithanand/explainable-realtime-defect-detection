from pathlib import Path
from sklearn.model_selection import train_test_split


def collect_webcam_cap_samples(root_dir: str):
    """
    Collects webcam bottle cap images.

    Expected folder structure:
        data/webcam_bottle_caps/
        ├── normal/
        └── defective/

    Labels:
        0 = normal
        1 = defective
    """

    root_dir = Path(root_dir)

    normal_dir = root_dir / "normal"
    defective_dir = root_dir / "defective"

    samples = []

    for image_path in normal_dir.glob("*.png"):
        samples.append((str(image_path), 0))

    for image_path in defective_dir.glob("*.png"):
        samples.append((str(image_path), 1))

    return samples


def create_webcam_train_val_test_split(
    root_dir: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    samples = collect_webcam_cap_samples(root_dir)

    image_paths = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_size + val_size,
        stratify=labels,
        random_state=random_state,
    )

    relative_val_size = val_size / (test_size + val_size)

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=1 - relative_val_size,
        stratify=temp_labels,
        random_state=random_state,
    )

    return {
        "train": list(zip(train_paths, train_labels)),
        "val": list(zip(val_paths, val_labels)),
        "test": list(zip(test_paths, test_labels)),
    }