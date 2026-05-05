from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from src.data.prepare_data import create_train_val_test_split
from src.models.model import create_resnet18_binary_model
from src.visualization.gradcam import GradCAM, overlay_cam_on_image


# -------------------------
# Configuration
# -------------------------

DATA_DIR = "data/mvtec_ad/bottle"
MODEL_PATH = "models/resnet18_bottle_binary.pth"
OUTPUT_DIR = Path("outputs/heatmaps")

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
# Load model
# -------------------------

model = create_resnet18_binary_model(pretrained=False)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)

model = model.to(DEVICE)
model.eval()


# -------------------------
# Create Grad-CAM object
# -------------------------

target_layer = model.layer4[-1]
grad_cam = GradCAM(model=model, target_layer=target_layer)


# -------------------------
# Load test samples
# -------------------------

splits = create_train_val_test_split(DATA_DIR)
test_samples = splits["test"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

misclassified_count = 0

print(f"Using device: {DEVICE}")
print(f"Test samples: {len(test_samples)}")


# -------------------------
# Analyze misclassified images
# -------------------------

for image_path, true_label in test_samples:
    image = Image.open(image_path).convert("RGB")

    input_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_label = torch.max(probabilities, dim=1)

    predicted_label = predicted_label.item()
    confidence = confidence.item()

    if predicted_label != true_label:
        misclassified_count += 1

        # Grad-CAM for the predicted class.
        # This shows WHY the model made the wrong prediction.
        cam_predicted, _ = grad_cam.generate(
            input_tensor=input_tensor,
            target_class=predicted_label,
        )

        overlay_predicted = overlay_cam_on_image(
            image=image,
            cam=cam_predicted,
            alpha=0.45,
        )

        # Grad-CAM for the true class.
        # This shows what the model might focus on for the correct class.
        cam_true, _ = grad_cam.generate(
            input_tensor=input_tensor,
            target_class=true_label,
        )

        overlay_true = overlay_cam_on_image(
            image=image,
            cam=cam_true,
            alpha=0.45,
        )

        print("\nMisclassified image")
        print("-" * 30)
        print(f"Path      : {image_path}")
        print(f"True      : {CLASS_NAMES[true_label]}")
        print(f"Predicted : {CLASS_NAMES[predicted_label]}")
        print(f"Confidence: {confidence:.4f}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(overlay_predicted)
        axes[1].set_title(
            f"Grad-CAM for predicted class:\n{CLASS_NAMES[predicted_label]}"
        )
        axes[1].axis("off")

        axes[2].imshow(overlay_true)
        axes[2].set_title(
            f"Grad-CAM for true class:\n{CLASS_NAMES[true_label]}"
        )
        axes[2].axis("off")

        plt.tight_layout()

        save_path = OUTPUT_DIR / f"misclassified_gradcam_{misclassified_count}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()

        print(f"Saved Grad-CAM visualization to: {save_path}")


grad_cam.remove_hooks()

print(f"\nTotal misclassified samples: {misclassified_count}")