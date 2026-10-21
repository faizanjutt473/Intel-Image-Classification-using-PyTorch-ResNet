"""Evaluate a trained checkpoint on the held-out test set.

Produces accuracy, a per-class report, confusion-matrix plots and a grid of the
most confident mistakes in ``outputs/``.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import config
from dataset import make_test_loader
from model import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--weights", type=Path, default=config.BEST_MODEL_PATH)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tta", action="store_true", help="average predictions with a horizontally flipped copy")
    return p.parse_args()


@torch.no_grad()
def predict_probs(model, loader, device, tta=False):
    """Softmax probabilities for every image in ``loader`` (order preserved)."""
    outputs = []
    for images, _ in loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1)
        if tta:
            probs = (probs + torch.softmax(model(torch.flip(images, dims=[3])), dim=1)) / 2
        outputs.append(probs.cpu())
    return torch.cat(outputs).numpy()


def plot_confusion_matrix(cm, classes, path, normalize):
    values = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(values, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set(xlabel="Predicted label", ylabel="True label",
           title="Confusion matrix" + (" (row-normalised)" if normalize else " (counts)"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_misclassified_grid(paths, y_true, y_pred, probs, classes, out_path, max_images=12):
    """Save the most confident wrong predictions for error analysis."""
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return
    confidence = probs[wrong, y_pred[wrong]]
    chosen = wrong[np.argsort(-confidence)][:max_images]
    cols = 4
    rows = math.ceil(len(chosen) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, chosen):
        ax.imshow(Image.open(paths[idx]).convert("RGB"))
        ax.set_title(f"true: {classes[y_true[idx]]}\npred: {classes[y_pred[idx]]} "
                     f"({probs[idx, y_pred[idx]]:.0%})", fontsize=9)
    fig.suptitle("Most confident misclassifications", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.weights, device)
    classes = checkpoint["classes"]

    loader, test_set = make_test_loader(args.data_dir, args.batch_size, args.workers, checkpoint["img_size"])
    if test_set.classes != classes:
        raise ValueError("Test-set classes do not match the checkpoint classes.")

    probs = predict_probs(model, loader, device, args.tta)
    y_pred = probs.argmax(axis=1)
    y_true = np.array(test_set.targets)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Test accuracy: {accuracy * 100:.2f}%  ({len(y_true)} images, TTA={'on' if args.tta else 'off'})\n")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    out = config.OUTPUTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, classes, out / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm, classes, out / "confusion_matrix_normalized.png", normalize=True)
    save_misclassified_grid([s[0] for s in test_set.samples], y_true, y_pred, probs, classes,
                            out / "misclassified_examples.png")

    metrics = {
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {c: report[c] for c in classes},
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "arch": checkpoint["arch"],
        "best_val_acc": checkpoint["val_acc"],
        "num_test_images": int(len(y_true)),
        "tta": args.tta,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics and plots to {out}/\nNext: python generate_report.py")


if __name__ == "__main__":
    main()
