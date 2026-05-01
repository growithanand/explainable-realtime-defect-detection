import argparse

import torch
from torch import nn, optim
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.models.model import build_resnet18
from src.utils.config import TrainingConfig, ProjectPaths
from src.utils.helpers import get_device, save_checkpoint, set_seed


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.inference_mode()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main(args):
    set_seed(args.seed)
    cfg = TrainingConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
    )
    paths = ProjectPaths()
    device = get_device()

    print(f"Using device: {device}")

    train_loader, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Classes: {class_names}")

    model = build_resnet18(num_classes=len(class_names), pretrained=True).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_acc = 0.0
    checkpoint_path = paths.models_dir / cfg.checkpoint_name

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

        print(
            f"Epoch {epoch+1:03d}/{cfg.epochs} | "
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
            f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, checkpoint_path, class_names=class_names)
            print(f"Saved best model to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train defect detection classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to processed dataset category folder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0, help="Use 0 on Windows if multiprocessing causes issues")
    args = parser.parse_args()
    main(args)
