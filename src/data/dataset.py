from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(image_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def create_dataloaders(data_dir: str | Path, image_size: int = 224, batch_size: int = 16, num_workers: int = 0):
    """
    Expected folder structure:

    data_dir/
      train/
        normal/
        defective/
      test/
        normal/
        defective/
    """
    data_dir = Path(data_dir)
    train_transform, eval_transform = get_transforms(image_size)

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, train_dataset.classes
