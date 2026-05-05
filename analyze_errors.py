from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from src.data.prepare_data import create_train_val_test_split
from src.data.dataset import ImageClassificationDataset
from src.models.model import create_resnet18_binary_model


# -------------------------
# Configuration
# -------------------------

DATA_DIR = "data/mvtec_ad/bottle"
MODEL_PATH = "models/resnet18_bottle_binary.pth"
OUTPUT_DIR = Path("outputs/misclassifications")

BATCH_SIZE = 16
IMAGE_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["normal", "defective"]


# -------------------------
# Transform
# -------------------------

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# -------------------------
# Data
# -------------------------

splits = create_train_val_test_split(DATA_DIR)

test_dataset = ImageClassificationDataset(
    samples=splits["test"],
    transform=eval_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# -------------------------
# Load model
# -------------------------

model = create_resnet18_binary_model(pretrained=False)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)

model = model.to(DEVICE)
model.eval()


# -------------------------
# Analyze wrong predictions
# -------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

misclassified = []
sample_index = 0

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)

        batch_size = labels.size(0)

        for i in range(batch_size):
            true_label = labels[i].item()
            predicted_label = predictions[i].item()
            confidence = confidences[i].item()

            image_path, _ = test_dataset.samples[sample_index]

            if true_label != predicted_label:
                misclassified.append(
                    {
                        "image_path": image_path,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "confidence": confidence,
                    }
                )

            sample_index += 1


print(f"Using device: {DEVICE}")
print(f"Total test samples: {len(test_dataset)}")
print(f"Misclassified samples: {len(misclassified)}")

if len(misclassified) == 0:
    print("No misclassified samples found.")
else:
    for idx, item in enumerate(misclassified):
        image_path = item["image_path"]
        true_label = item["true_label"]
        predicted_label = item["predicted_label"]
        confidence = item["confidence"]

        print("\nMisclassified image")
        print("-" * 30)
        print(f"Path      : {image_path}")
        print(f"True      : {CLASS_NAMES[true_label]}")
        print(f"Predicted : {CLASS_NAMES[predicted_label]}")
        print(f"Confidence: {confidence:.4f}")

        image = Image.open(image_path).convert("RGB")

        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"True: {CLASS_NAMES[true_label]} | "
            f"Pred: {CLASS_NAMES[predicted_label]} | "
            f"Conf: {confidence:.2f}"
        )

        save_path = OUTPUT_DIR / f"misclassified_{idx + 1}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()

        print(f"Saved to: {save_path}")