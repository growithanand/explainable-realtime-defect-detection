import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.prepare_data import create_train_val_test_split
from src.data.dataset import ImageClassificationDataset


DATA_DIR = "data/mvtec_ad/bottle"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

splits = create_train_val_test_split(DATA_DIR)

train_dataset = ImageClassificationDataset(
    samples=splits["train"],
    transform=transform
)

val_dataset = ImageClassificationDataset(
    samples=splits["val"],
    transform=transform
)

test_dataset = ImageClassificationDataset(
    samples=splits["test"],
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

images, labels = next(iter(train_loader))

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Labels:", labels)