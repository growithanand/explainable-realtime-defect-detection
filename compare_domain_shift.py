from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from src.data.prepare_webcam_data import create_webcam_train_val_test_split
from src.data.dataset import ImageClassificationDataset
from src.models.model import create_resnet18_binary_model


DATA_DIR = "data/webcam_bottle_caps"

BASE_MODEL_PATH = "models/resnet18_bottle_binary.pth"
FINETUNED_MODEL_PATH = "models/resnet18_webcam_caps_finetuned.pth"

OUTPUT_DIR = Path("outputs")
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


def load_model(model_path):
    model = create_resnet18_binary_model(pretrained=False)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def evaluate_model(model, dataloader):
    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for images, labels in dataloader:
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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "labels": all_labels,
        "predictions": all_predictions,
    }


def save_confusion_matrix(labels, predictions, title, save_path):
    cm = confusion_matrix(labels, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )

    disp.plot()
    plt.title(title)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    print(f"Using device: {DEVICE}")
    print(f"Webcam test samples: {len(test_dataset)}")

    models_to_compare = {
        "MVTec baseline model": BASE_MODEL_PATH,
        "Webcam fine-tuned model": FINETUNED_MODEL_PATH,
    }

    results = []

    for model_name, model_path in models_to_compare.items():
        print(f"\nEvaluating: {model_name}")
        print("-" * 40)

        model = load_model(model_path)
        metrics = evaluate_model(model, test_loader)

        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1_score']:.4f}")

        print("\nClassification Report")
        print(
            classification_report(
                metrics["labels"],
                metrics["predictions"],
                target_names=CLASS_NAMES,
                zero_division=0,
            )
        )

        results.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        })

        cm_path = OUTPUT_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"

        save_confusion_matrix(
            labels=metrics["labels"],
            predictions=metrics["predictions"],
            title=model_name,
            save_path=cm_path,
        )

        print(f"Saved confusion matrix to: {cm_path}")

    results_df = pd.DataFrame(results)

    csv_path = OUTPUT_DIR / "domain_shift_comparison.csv"
    results_df.to_csv(csv_path, index=False)

    print("\nComparison Summary")
    print("-" * 40)
    print(results_df)

    print(f"\nSaved comparison CSV to: {csv_path}")


if __name__ == "__main__":
    main()