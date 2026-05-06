from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt

from src.data.prepare_webcam_data import create_webcam_train_val_test_split
from src.data.dataset import ImageClassificationDataset
from src.models.model import create_resnet18_binary_model


DATA_DIR = "data/webcam_bottle_caps"
MODEL_PATH = "models/resnet18_webcam_caps_finetuned.pth"
OUTPUT_DIR = "outputs"

BATCH_SIZE = 16
IMAGE_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["normal", "defective"]


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


splits = create_webcam_train_val_test_split(DATA_DIR)

test_dataset = ImageClassificationDataset(
    samples=splits["test"],
    transform=eval_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


model = create_resnet18_binary_model(pretrained=False)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)

model = model.to(DEVICE)
model.eval()


all_labels = []
all_predictions = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())


accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, zero_division=0)
recall = recall_score(all_labels, all_predictions, zero_division=0)
f1 = f1_score(all_labels, all_predictions, zero_division=0)

print(f"Using device: {DEVICE}")
print(f"Test samples: {len(test_dataset)}")

print("\nEvaluation Metrics")
print("-" * 30)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification Report")
print("-" * 30)
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)

cm = confusion_matrix(all_labels, all_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)

disp.plot()

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

confusion_matrix_path = Path(OUTPUT_DIR) / "confusion_matrix_webcam_caps.png"

plt.title("Confusion Matrix - Webcam Bottle Cap Defect Detection")
plt.savefig(confusion_matrix_path, bbox_inches="tight")
plt.show()

print(f"\nSaved confusion matrix to: {confusion_matrix_path}")