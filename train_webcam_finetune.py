from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.prepare_webcam_data import create_webcam_train_val_test_split
from src.data.dataset import ImageClassificationDataset
from src.models.model import create_resnet18_binary_model


# -------------------------
# Configuration
# -------------------------

DATA_DIR = "data/webcam_bottle_caps"

BASE_MODEL_PATH = "models/resnet18_bottle_binary.pth"
FINETUNED_MODEL_PATH = "models/resnet18_webcam_caps_finetuned.pth"

BATCH_SIZE = 16
NUM_EPOCHS = 12
LEARNING_RATE = 1e-5
IMAGE_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Transforms
# -------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=12),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

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

splits = create_webcam_train_val_test_split(DATA_DIR)

train_dataset = ImageClassificationDataset(
    samples=splits["train"],
    transform=train_transform,
)

val_dataset = ImageClassificationDataset(
    samples=splits["val"],
    transform=eval_transform,
)

test_dataset = ImageClassificationDataset(
    samples=splits["test"],
    transform=eval_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# -------------------------
# Model
# -------------------------

model = create_resnet18_binary_model(pretrained=False)

model.load_state_dict(
    torch.load(
        BASE_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze final ResNet block and classifier head.
# This lets the model adapt to your webcam/cap domain without changing everything.
for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)


# -------------------------
# Loss and optimizer
# -------------------------

# Your dataset is imbalanced: more normal than defective.
# Class weights make defective examples matter more during training.
num_normal = sum(label == 0 for _, label in train_dataset.samples)
num_defective = sum(label == 1 for _, label in train_dataset.samples)

total = num_normal + num_defective

weight_normal = total / (2 * num_normal)
weight_defective = total / (2 * num_defective)

class_weights = torch.tensor(
    [weight_normal, weight_defective],
    dtype=torch.float32,
).to(DEVICE)

loss_fn = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
)


# -------------------------
# Training helpers
# -------------------------

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_samples

    return avg_loss, accuracy


def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_samples

    return avg_loss, accuracy, all_labels, all_predictions


# -------------------------
# Main training loop
# -------------------------

print(f"Using device: {DEVICE}")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Train normal: {num_normal}")
print(f"Train defective: {num_defective}")
print(f"Class weights: normal={weight_normal:.3f}, defective={weight_defective:.3f}")

best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    train_loss, train_acc = train_one_epoch(
        model=model,
        dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE,
    )

    val_loss, val_acc, _, _ = evaluate(
        model=model,
        dataloader=val_loader,
        loss_fn=loss_fn,
        device=DEVICE,
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        Path("models").mkdir(parents=True, exist_ok=True)

        torch.save(
            model.state_dict(),
            FINETUNED_MODEL_PATH,
        )

        print(f"Saved best fine-tuned model to {FINETUNED_MODEL_PATH}")