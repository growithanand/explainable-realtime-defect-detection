from pathlib import Path
from typing import Union, Dict, Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.model import create_resnet18_binary_model


class DefectPredictor:
    """
    Reusable inference utility for binary industrial defect detection.

    Classes:
        0 = normal
        1 = defective
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        image_size: int = 224,
        defect_threshold: float = 0.5,
        device: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.defect_threshold = defect_threshold

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.class_names = ["normal", "defective"]

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        model = create_resnet18_binary_model(pretrained=False)

        model.load_state_dict(
            torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True,
            )
        )

        model = model.to(self.device)
        model.eval()

        return model

    def _prepare_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        input_format: str = "rgb",
    ) -> Image.Image:
        """
        Converts different image input types into a PIL RGB image.

        Supports:
            - file path
            - PIL image
            - numpy array from OpenCV or other sources

        input_format:
            - "rgb" for normal RGB arrays
            - "bgr" for OpenCV frames
        """

        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            if input_format == "bgr":
                image = image[:, :, ::-1]

            return Image.fromarray(image).convert("RGB")

        raise TypeError(
            "Unsupported image type. Use path, PIL image, or numpy array."
        )

    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        input_format: str = "rgb",
    ) -> Dict[str, Any]:
        """
        Predicts whether an image is normal or defective.

        Returns:
            prediction label
            class name
            confidence
            normal probability
            defective probability
        """

        pil_image = self._prepare_image(image, input_format=input_format)

        input_tensor = self.transform(pil_image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        normal_probability = probabilities[0].item()
        defective_probability = probabilities[1].item()

        # Industrial inspection usually cares strongly about missed defects.
        # So we classify as defective if the defective probability crosses
        # a configurable threshold.
        predicted_label = 1 if defective_probability >= self.defect_threshold else 0

        if predicted_label == 1:
            confidence = defective_probability
        else:
            confidence = normal_probability

        return {
            "predicted_label": predicted_label,
            "predicted_class": self.class_names[predicted_label],
            "confidence": confidence,
            "normal_probability": normal_probability,
            "defective_probability": defective_probability,
            "defect_threshold": self.defect_threshold,
            "device": self.device,
        }