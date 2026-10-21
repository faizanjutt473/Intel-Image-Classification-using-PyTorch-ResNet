"""Dataset loading, train/validation/test splitting and data augmentation."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch import cuda
from torchvision import datasets, transforms

import config


def resolve_split_dir(data_dir, split):
    """Locate ``<data_dir>/<split>/<split>`` (Kaggle layout) or ``<data_dir>/<split>``."""
    data_dir = Path(data_dir)
    for candidate in (data_dir / split / split, data_dir / split):
        if candidate.is_dir() and any(p.is_dir() for p in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(
        f"Could not find class folders for '{split}' under '{data_dir}'. "
        "Download the Kaggle dataset and extract it into the data/ folder."
    )


def build_transforms(img_size=config.IMG_SIZE):
    """Return (train_transform, eval_transform). Augmentation is train-only."""
    normalize = transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    classes: list
    split_counts: dict  # {"train": {class: n}, "val": {...}, "test": {...}}


def _count_per_class(targets, classes):
    counts = Counter(targets)
    return {name: counts.get(i, 0) for i, name in enumerate(classes)}


def _loader(dataset, batch_size, workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=cuda.is_available(),
        persistent_workers=workers > 0,
    )


def make_dataloaders(data_dir, batch_size, workers, img_size=config.IMG_SIZE,
                     val_fraction=config.VAL_FRACTION, seed=config.SEED):
    """Stratified train/val split of ``seg_train``; ``seg_test`` is the held-out test set."""
    train_tf, eval_tf = build_transforms(img_size)
    train_root = resolve_split_dir(data_dir, "seg_train")
    test_root = resolve_split_dir(data_dir, "seg_test")

    # Two views of the same files: augmented for training, clean for validation.
    train_view = datasets.ImageFolder(train_root, transform=train_tf)
    val_view = datasets.ImageFolder(train_root, transform=eval_tf)
    test_set = datasets.ImageFolder(test_root, transform=eval_tf)

    if train_view.classes != test_set.classes:
        raise ValueError("Train and test folders have different class names.")

    train_idx, val_idx = train_test_split(
        list(range(len(train_view))),
        test_size=val_fraction,
        stratify=train_view.targets,
        random_state=seed,
    )
    train_set, val_set = Subset(train_view, train_idx), Subset(val_view, val_idx)

    classes = train_view.classes
    split_counts = {
        "train": _count_per_class([train_view.targets[i] for i in train_idx], classes),
        "val": _count_per_class([train_view.targets[i] for i in val_idx], classes),
        "test": _count_per_class(test_set.targets, classes),
    }
    return DataBundle(
        train_loader=_loader(train_set, batch_size, workers, shuffle=True),
        val_loader=_loader(val_set, batch_size, workers, shuffle=False),
        test_loader=_loader(test_set, batch_size, workers, shuffle=False),
        classes=classes,
        split_counts=split_counts,
    )


def make_test_loader(data_dir, batch_size, workers, img_size=config.IMG_SIZE):
    """Return (loader, ImageFolder) for the official test set (no augmentation)."""
    _, eval_tf = build_transforms(img_size)
    test_set = datasets.ImageFolder(resolve_split_dir(data_dir, "seg_test"), transform=eval_tf)
    return _loader(test_set, batch_size, workers, shuffle=False), test_set


def summarize(data_dir):
    """Print class counts for every split found in the dataset folder."""
    for split in ("seg_train", "seg_test"):
        root = resolve_split_dir(data_dir, split)
        print(f"\n{split}  ({root})")
        total = 0
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            n = sum(1 for f in class_dir.iterdir() if f.is_file())
            total += n
            print(f"  {class_dir.name:<10} {n:>6}")
        print(f"  {'TOTAL':<10} {total:>6}")
    pred_dir = Path(data_dir) / "seg_pred" / "seg_pred"
    if pred_dir.is_dir():
        print(f"\nseg_pred (unlabeled): {sum(1 for f in pred_dir.iterdir() if f.is_file())} images")


if __name__ == "__main__":
    summarize(config.DATA_DIR)
