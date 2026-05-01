from pathlib import Path
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(model: torch.nn.Module, path: str | Path, class_names=None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names or ["normal", "defective"],
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device):
    return torch.load(path, map_location=device, weights_only=True)
