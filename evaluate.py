import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.data.dataset import create_dataloaders
from src.models.model import load_model
from src.utils.helpers import get_device


@torch.inference_mode()
def collect_predictions(model, dataloader, device):
    y_true, y_pred = [], []
    model.eval()

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        y_pred.extend(preds)
        y_true.extend(labels.numpy().tolist())

    return y_true, y_pred


def main(args):
    device = get_device()
    _, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = load_model(args.checkpoint, device=device, num_classes=len(class_names))
    y_true, y_pred = collect_predictions(model, test_loader, device)

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained defect detection model")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    main(args)
