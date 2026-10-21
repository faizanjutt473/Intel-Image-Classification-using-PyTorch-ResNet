"""Generate assets/architecture.png and assets/preview.png (no dataset or GPU needed)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(exist_ok=True)

NAVY, BLUE, TEAL, ORANGE, GREEN, PURPLE = "#12263f", "#2f6fdb", "#12a4a0", "#f08a24", "#2e9d5b", "#7b5cd6"
LIGHT = "#f4f7fb"


def box(ax, x, y, w, h, title, lines=(), color=BLUE, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="white", ec=color, lw=2.2, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.62), w, 0.62, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=color, ec=color, lw=2.2, zorder=3))
    ax.text(x + w / 2, y + h - 0.31, title, ha="center", va="center", color="white",
            fontsize=fs, fontweight="bold", zorder=4)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 1.0 - i * 0.36, line, ha="center", va="center",
                color=NAVY, fontsize=fs - 2, zorder=4)


def arrow(ax, p1, p2, color=NAVY, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=18, lw=2,
                                 color=color, connectionstyle=f"arc3,rad={rad}", zorder=1))


ARCH_INFO = {
    "resnet18": ("ResNet-18", 512, "Layer1-4 (basic blocks)"),
    "resnet50": ("ResNet-50", 2048, "Layer1-4 (bottleneck blocks)"),
    "mobilenet_v3_large": ("MobileNetV3-Large", 960, "Inverted residual blocks"),
    "efficientnet_b0": ("EfficientNet-B0", 1280, "MBConv blocks"),
}


def architecture(arch="resnet18"):
    name, feats, blocks = ARCH_INFO.get(arch, (arch, "N", "Backbone layers"))
    fig, ax = plt.subplots(figsize=(18, 10.5))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10.5)
    ax.axis("off")
    fig.patch.set_facecolor(LIGHT)

    ax.text(9, 10.0, "Intel Scene Image Classifier: System Architecture", ha="center",
            fontsize=21, fontweight="bold", color=NAVY)
    ax.text(9, 9.55, f"Transfer learning with PyTorch ({name}), 6 scene classes",
            ha="center", fontsize=12, color="#51627a")

    # Row 1: data pipeline
    ax.text(0.4, 8.85, "1  DATA PIPELINE", fontsize=12, fontweight="bold", color=BLUE)
    y, h, w = 6.65, 2.0, 3.1
    xs = [0.4, 4.15, 7.9, 11.65, 15.0]
    box(ax, xs[0], y, w, h, "Kaggle Dataset", ["Intel Image Classification", "seg_train | seg_test", "6 classes, ~150 px"], BLUE)
    box(ax, xs[1], y, w, h, "Stratified Split", ["Train 85% of seg_train", "Validation 15%", "Test = seg_test"], BLUE)
    box(ax, xs[2], y, w, h, "Augmentation", ["(train set only)", "Random crop, flip", "Rotation, colour jitter"], ORANGE)
    box(ax, xs[3], y, w, h, "Preprocessing", ["Resize 224 x 224", "ImageNet mean / std", "DataLoader batches"], TEAL)
    box(ax, xs[4], y, 2.6, h, "Batch", ["Tensor", "3 x 224 x 224"], PURPLE)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + (w if a != xs[3] else w), y + h / 2), (b, y + h / 2))

    # Row 2: model
    ax.text(0.4, 6.1, "2  MODEL (fine-tuned)", fontsize=12, fontweight="bold", color=BLUE)
    ax.add_patch(FancyBboxPatch((0.4, 3.75), 17.2, 2.1, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="#eaf1fb", ec="#c4d4ee", lw=1.5, zorder=0))
    my, mh = 4.2, 1.5
    parts = [
        (0.7, 2.2, "Input", ["224 x 224 x 3"], NAVY),
        (3.3, 2.6, "Conv1 + Pool", ["7x7 conv, BN, ReLU"], BLUE),
        (6.3, 4.3, f"{name} Backbone", [blocks, "ImageNet pretrained"], BLUE),
        (11.0, 2.2, "Global AvgPool", [f"{feats} features"], TEAL),
        (13.6, 1.9, "Dropout", ["p = 0.3"], ORANGE),
        (15.9, 1.5, "Linear", [f"{feats} to 6"], ORANGE),
    ]
    for x, w_, t, ls, c in parts:
        box(ax, x, my, w_, mh, t, ls, c, fs=10)
    ends = [(p[0], p[0] + p[1]) for p in parts]
    for (a0, a1), (b0, _) in zip(ends[:-1], ends[1:]):
        arrow(ax, (a1, my + mh / 2 - 0.2), (b0, my + mh / 2 - 0.2))
    ax.text(9.0, 3.95, "Backbone frozen in warm-up (2 epochs), then everything fine-tuned",
            ha="center", fontsize=9.5, color="#51627a", style="italic")

    # Row 3: training / eval / deploy
    ax.text(0.4, 3.35, "3  TRAINING, EVALUATION AND DEPLOYMENT", fontsize=12, fontweight="bold", color=BLUE)
    y3, h3, w3 = 0.45, 2.55, 3.35
    x3 = [0.4, 4.45, 8.5, 12.55]
    box(ax, x3[0], y3, w3, h3, "Training (train.py)",
        ["Phase 1: head only", "Phase 2: full network", "AdamW, cosine LR", "Label smoothing 0.1"], ORANGE, fs=10.5)
    box(ax, x3[1], y3, w3, h3, "Best Checkpoint",
        ["Selected on validation acc.", "Early stopping", "models/best_model.pth"], PURPLE, fs=10.5)
    box(ax, x3[2], y3, w3, h3, "Evaluation (evaluate.py)",
        ["Test accuracy, F1", "Confusion matrix", "Error analysis, TTA"], GREEN, fs=10.5)
    box(ax, x3[3], y3, 5.05, h3, "Inference",
        ["predict.py: CLI for image / folder", "app.py: Gradio web interface", "Output: top-3 classes + probability"], TEAL, fs=10.5)
    for a, b in zip(x3[:-1], x3[1:]):
        arrow(ax, (a + w3, y3 + h3 / 2), (b, y3 + h3 / 2))
    ax.text(0.4, 0.12, "Outputs: REPORT.md, outputs/metrics.json, confusion_matrix.png, training_curves.png",
            fontsize=9.5, color="#51627a")
    fig.savefig(ASSETS / "architecture.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def preview(metrics):
    """Banner built from real evaluation results (outputs/metrics.json)."""
    classes = metrics["classes"]
    name = ARCH_INFO.get(metrics["arch"], (metrics["arch"],))[0]
    acc = metrics["accuracy"] * 100
    f1 = [metrics["per_class"][c]["f1-score"] for c in classes]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor(NAVY)
    ax.add_patch(Rectangle((0, 0), 16, 9, fc=NAVY, zorder=0))
    ax.add_patch(Polygon([(0, 0), (16, 0), (16, 2.6), (0, 0.9)], fc="#183558", zorder=0.5))

    ax.text(0.8, 7.9, "INTEL SCENE", fontsize=34, fontweight="bold", color="white", va="center")
    ax.text(0.8, 6.95, "IMAGE CLASSIFIER", fontsize=34, fontweight="bold", color="#5fb4ff", va="center")
    ax.text(0.8, 6.15, f"Fine-tuned {name}  |  PyTorch  |  Transfer Learning", fontsize=14, color="#c8d6ea")

    # headline metric card
    ax.add_patch(FancyBboxPatch((0.8, 3.0), 5.6, 2.5, boxstyle="round,pad=0.02,rounding_size=0.25",
                                fc="#1b3a60", ec="#2f6fdb", lw=2.5, zorder=2))
    ax.text(3.6, 4.55, f"{acc:.2f}%", ha="center", va="center", fontsize=54, fontweight="bold",
            color="white", zorder=3)
    ax.text(3.6, 3.65, "Test accuracy", ha="center", va="center", fontsize=17, color="#9fc4f5", zorder=3)
    ax.text(3.6, 3.22, f"{metrics['num_test_images']:,} unseen test images  |  Macro F1 {metrics['macro_f1']:.3f}",
            ha="center", va="center", fontsize=11, color="#c8d6ea", zorder=3)

    feats = ["Train / validation / test split", "Data augmentation + fine-tuning",
             "Confusion matrix and per-class report", "predict.py + Gradio web app"]
    for i, f in enumerate(feats):
        ax.text(0.85, 2.15 - i * 0.45, "-  " + f, fontsize=13.5, color="#e6eefb")
    ax.text(0.8, 0.12, "Dataset: Kaggle, Intel Image Classification", fontsize=10.5, color="#8da2bf")

    # per-class F1 chart card
    cx, cy, cw, ch = 8.1, 0.9, 7.4, 7.2
    ax.add_patch(FancyBboxPatch((cx, cy), cw, ch, boxstyle="round,pad=0.02,rounding_size=0.25",
                                fc="white", ec="#c4d4ee", lw=2, zorder=2))
    ax.text(cx + cw / 2, cy + ch - 0.55, "Per-class F1-score (test set)", ha="center", va="center",
            fontsize=16, fontweight="bold", color=NAVY, zorder=4)
    colors = [BLUE, GREEN, TEAL, PURPLE, "#2b7fb8", ORANGE]
    bar_x, bar_w = cx + 1.95, 4.9
    for i, (c, v, col) in enumerate(zip(classes, f1, colors)):
        by = cy + ch - 1.75 - i * 0.95
        ax.text(cx + 0.4, by + 0.27, c, fontsize=14, va="center", color=NAVY, fontweight="bold", zorder=4)
        ax.add_patch(Rectangle((bar_x, by), bar_w, 0.54, fc="#eaf1fb", zorder=3))
        ax.add_patch(Rectangle((bar_x, by), bar_w * v, 0.54, fc=col, zorder=4))
        ax.text(bar_x + bar_w * v - 0.12, by + 0.27, f"{v:.3f}", ha="right", va="center",
                fontsize=12.5, color="white", fontweight="bold", zorder=5)
    ax.text(cx + cw / 2, cy + 0.3, "Source: outputs/metrics.json", ha="center",
            fontsize=9, color="#8da2bf", style="italic", zorder=4)

    fig.savefig(ASSETS / "preview.png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=ASSETS.parent / "outputs" / "metrics.json")
    args = parser.parse_args()
    if not args.metrics.exists():
        raise SystemExit(f"{args.metrics} not found. Run evaluate.py first.")
    results = json.loads(args.metrics.read_text())
    architecture(results["arch"])
    preview(results)
    print(f"Saved images to {ASSETS}")
