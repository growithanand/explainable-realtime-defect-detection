from pathlib import Path
from sklearn.model_selection import train_test_split


def collect_mvtec_binary_samples(root_dir: str):
    root_dir = Path(root_dir)

    samples = []

    # Normal images from train/good
    train_good_dir = root_dir / "train" / "good"
    for image_path in train_good_dir.glob("*.png"):
        samples.append((str(image_path), 0))

    # Normal and defective images from test folders
    test_dir = root_dir / "test"
    for folder in test_dir.iterdir():
        if not folder.is_dir():
            continue

        label = 0 if folder.name == "good" else 1

        for image_path in folder.glob("*.png"):
            samples.append((str(image_path), label))

    return samples


def create_train_val_test_split(
    root_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    samples = collect_mvtec_binary_samples(root_dir)

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