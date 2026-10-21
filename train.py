"""Fine-tune a pretrained CNN on the Intel Image Classification dataset.

Training runs in two phases:
  1. Warm-up: backbone frozen, only the new classification head is trained.
  2. Fine-tune: whole network unfrozen with a lower backbone learning rate and
     cosine learning-rate decay. Early stopping monitors validation accuracy.
"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

import config
from dataset import make_dataloaders
from model import SUPPORTED_ARCHS, build_model, set_backbone_trainable, split_parameters


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--arch", choices=SUPPORTED_ARCHS, default="resnet50")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--warmup-epochs", type=int, default=2, help="epochs with frozen backbone")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5, help="early-stopping patience (fine-tune phase)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--no-pretrained", action="store_true", help="random init (for quick tests only)")
    p.add_argument("--output", type=Path, default=config.BEST_MODEL_PATH)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    """One pass over ``loader``. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    desc = "train" if training else "val  "

    with torch.set_grad_enabled(training):
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            seen += images.size(0)
    return total_loss / seen, correct / seen


def configure_phase(model, phase, args):
    """Return (optimizer, scheduler) for the 'head' or 'finetune' phase."""
    backbone, head = split_parameters(model)
    if phase == "head":
        set_backbone_trainable(model, False)
        optimizer = optim.AdamW(head, lr=args.head_lr, weight_decay=args.weight_decay)
        return optimizer, None
    set_backbone_trainable(model, True)
    optimizer = optim.AdamW(
        [{"params": backbone, "lr": args.backbone_lr}, {"params": head, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=1e-6
    )
    return optimizer, scheduler


def plot_curves(history, path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(epochs, [h["train_acc"] * 100 for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] * 100 for h in history], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Architecture: {args.arch}")

    data = make_dataloaders(args.data_dir, args.batch_size, args.workers, config.IMG_SIZE,
                            config.VAL_FRACTION, args.seed)
    sizes = {k: sum(v.values()) for k, v in data.split_counts.items()}
    print(f"Classes: {data.classes}")
    print(f"Images -> train: {sizes['train']}, val: {sizes['val']}, test: {sizes['test']}")
    (config.OUTPUTS_DIR / "split_info.json").write_text(json.dumps(data.split_counts, indent=2))

    model = build_model(args.arch, len(data.classes), pretrained=not args.no_pretrained).to(device)
    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_val_acc, best_epoch, bad_epochs = [], 0.0, 0, 0
    phase, optimizer, scheduler = None, None, None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        new_phase = "head" if epoch <= args.warmup_epochs else "finetune"
        if new_phase != phase:
            phase = new_phase
            optimizer, scheduler = configure_phase(model, phase, args)
            print(f"\n--- Phase: {phase} ---")

        train_loss, train_acc = run_epoch(model, data.train_loader, train_criterion, device,
                                          optimizer, scaler, use_amp)
        val_loss, val_acc = run_epoch(model, data.val_loader, eval_criterion, device, use_amp=use_amp)
        lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        history.append({"epoch": epoch, "phase": phase, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "lr": lr})
        print(f"Epoch {epoch:02d}/{args.epochs} [{phase:<8}] "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, bad_epochs = val_acc, epoch, 0
            torch.save({"model_state": model.state_dict(), "classes": data.classes, "arch": args.arch,
                        "img_size": config.IMG_SIZE, "val_acc": best_val_acc, "epoch": epoch}, args.output)
            print(f"  saved best model (val acc {best_val_acc:.4f})")
        elif phase == "finetune":
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
                break

    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": str(device),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "train_time_sec": round(time.time() - start, 1),
    }
    (config.OUTPUTS_DIR / "history.json").write_text(json.dumps(summary, indent=2))
    plot_curves(history, config.OUTPUTS_DIR / "training_curves.png")
    print(f"\nBest validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Weights saved to: {args.output}\nNext: python evaluate.py --tta")


if __name__ == "__main__":
    main()
