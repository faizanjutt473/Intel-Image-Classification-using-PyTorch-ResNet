"""Predict the scene class of a new image (or every image in a folder).

Usage:
    python predict.py --image path/to/photo.jpg
    python predict.py --image path/to/folder --top-k 3
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

import config
from dataset import build_transforms
from model import load_checkpoint

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@torch.no_grad()
def predict_image(model, image, classes, transform, device, top_k=3):
    """Return the ``top_k`` (class_name, probability) pairs for a PIL image."""
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()
    values, indices = probs.topk(min(top_k, len(classes)))
    return [(classes[i], float(v)) for v, i in zip(values, indices)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True, help="image file or folder of images")
    p.add_argument("--weights", type=Path, default=config.BEST_MODEL_PATH)
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.weights, device)
    transform = build_transforms(checkpoint["img_size"])[1]

    if args.image.is_dir():
        files = sorted(p for p in args.image.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        files = [args.image]
    if not files:
        raise SystemExit(f"No images found at {args.image}")

    for path in files:
        results = predict_image(model, Image.open(path), checkpoint["classes"], transform, device, args.top_k)
        best_name, best_prob = results[0]
        print(f"\n{path.name}: {best_name} ({best_prob:.1%})")
        for name, prob in results[1:]:
            print(f"    {name:<10} {prob:.1%}")


if __name__ == "__main__":
    main()
