import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class DefectPredictor:
    def __init__(self, model: torch.nn.Module, device: torch.device, class_names=None, image_size: int = 224):
        self.model = model
        self.device = device
        self.class_names = class_names or ["normal", "defective"]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        return tensor

    @torch.inference_mode()
    def predict_frame(self, frame_bgr: np.ndarray):
        x = self.preprocess_frame(frame_bgr)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

        pred_idx = pred_idx.item()
        confidence = confidence.item()
        label = self.class_names[pred_idx]
        return label, confidence
