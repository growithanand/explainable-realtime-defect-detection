from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.prepare_data import create_train_val_test_split
from src.data.dataset import ImageClassificationDataset
from src.models.model import create_resnet18_binary_model


DATA_DIR = "data/mvtec_ad/bottle"
MODEL_SAVE_PATH = "models/resnet18_bottle_binary.pth"

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
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


splits = create_train_val_test_split(DATA_DIR)

train_dataset = ImageClassificationDataset(
    samples=splits["train"],
    transform=train_transform,
)

val_dataset = ImageClassificationDataset(
    samples=splits["val"],
    transform=eval_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


model = create_resnet18_binary_model(pretrained=True)
model = model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

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
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


print(f"Using device: {DEVICE}")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

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

    val_loss, val_acc = evaluate(
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
            MODEL_SAVE_PATH,
        )

        print(f"Saved best model to {MODEL_SAVE_PATH}")