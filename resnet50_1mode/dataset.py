from pathlib import Path

from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def build_datasets(root: str, image_size: int = 224):
    root = Path(root)
    train_tf, eval_tf = build_transforms(image_size)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    train_ds = datasets.ImageFolder(train_dir, train_tf)
    val_ds = datasets.ImageFolder(val_dir, eval_tf)

    test_ds = None
    if test_dir.exists() and any(test_dir.iterdir()):
        try:
            test_ds = datasets.ImageFolder(test_dir, eval_tf)
        except FileNotFoundError:
            test_ds = None

    return train_ds, val_ds, test_ds