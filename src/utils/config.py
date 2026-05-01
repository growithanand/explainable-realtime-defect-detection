from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = root / "data"
    models_dir: Path = root / "models"
    outputs_dir: Path = root / "outputs"
    reports_dir: Path = root / "reports"


@dataclass
class TrainingConfig:
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-4
    num_classes: int = 2
    class_names: tuple = ("normal", "defective")
    checkpoint_name: str = "best_model.pth"
