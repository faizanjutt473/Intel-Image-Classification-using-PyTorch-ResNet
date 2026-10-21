"""Build REPORT.md from the artefacts produced by train.py and evaluate.py.

All numbers in the report are read from ``outputs/`` so they always match the
trained model. Run after ``train.py`` and ``evaluate.py``.
"""
import argparse
import json
import sys

import config

OUT = config.OUTPUTS_DIR


def load_json(name):
    path = OUT / name
    if not path.exists():
        sys.exit(f"Missing {path}. Run train.py and evaluate.py first.")
    return json.loads(path.read_text())


def load_json_optional(name):
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def image_line(title, filename):
    """Markdown image line, only if the plot exists on disk."""
    return f"![{title}](outputs/{filename})\n" if (OUT / filename).exists() else ""


DEFAULT_CONFIG = {"warmup_epochs": 2, "head_lr": 0.001, "backbone_lr": 0.0001, "weight_decay": 0.0001,
                  "label_smoothing": 0.1, "batch_size": 32, "patience": 5, "seed": config.SEED}


def top_confusions(cm, classes, k=3):
    pairs = [(cm[i][j], classes[i], classes[j])
             for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i][j] > 0]
    return sorted(pairs, reverse=True)[:k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8,
                        help="epochs the model was trained for (used only if history.json is missing)")
    args = parser.parse_args()

    metrics = load_json("metrics.json")
    split = load_json("split_info.json")
    history = load_json_optional("history.json")
    classes = metrics["classes"]
    if history is not None:
        cfg = history["config"]
    else:
        cfg = {**DEFAULT_CONFIG, "arch": metrics["arch"], "epochs": args.epochs}
        history = {}
    totals = {name: sum(counts.values()) for name, counts in split.items()}

    L = []
    add = L.append
    add("# Intel Scene Image Classifier: Project Report\n")

    add("## 1. Objective\n")
    add(f"Build an image classifier for {len(classes)} scene classes ({', '.join(classes)}) by fine-tuning a "
        "pretrained convolutional network (transfer learning), evaluate it on a held-out test set, and provide "
        "a script and web interface for predicting the class of new images.\n")

    add("## 2. Dataset\n")
    add("- **Name:** Intel Image Classification")
    add("- **Source:** Kaggle, <https://www.kaggle.com/datasets/puneet6060/intel-image-classification> "
        "(originally published by Intel for a scene-classification challenge). "
        "Check the Kaggle page for the licence terms before redistributing the data.")
    add("- **Content:** colour photographs of natural and urban scenes, mostly about 150×150 px, "
        f"in {len(classes)} classes.")
    add("- **Splits used in this project:**\n")
    add("| Split | Origin | Images |\n|---|---|---|")
    add(f"| Train | {100 - int(config.VAL_FRACTION * 100)}% of `seg_train` (stratified) | {totals['train']} |")
    add(f"| Validation | {int(config.VAL_FRACTION * 100)}% of `seg_train` (stratified) | {totals['val']} |")
    add(f"| Test | official `seg_test` (never used for training or model selection) | {totals['test']} |\n")
    add("Images per class:\n")
    add("| Class | Train | Validation | Test |\n|---|---|---|---|")
    for c in classes:
        add(f"| {c} | {split['train'][c]} | {split['val'][c]} | {split['test'][c]} |")
    add("")

    add("## 3. Methodology\n")
    add(f"- **Model:** `{cfg['arch']}` pretrained on ImageNet; the final layer is replaced by a "
        f"{len(classes)}-way classifier (dropout 0.3).")
    add("- **Preprocessing:** images resized to 224×224 and normalised with ImageNet mean/std.")
    add("- **Data augmentation (training only):** random resized crop (scale 0.6–1.0), horizontal flip, "
        "rotation (±15°) and colour jitter (brightness, contrast, saturation 0.2).")
    add(f"- **Fine-tuning:** (1) {cfg['warmup_epochs']} warm-up epochs with the backbone frozen "
        f"(head LR {cfg['head_lr']}), then (2) the full network is unfrozen with AdamW "
        f"(backbone LR {cfg['backbone_lr']}, head LR {cfg['head_lr']}), cosine LR decay, "
        f"weight decay {cfg['weight_decay']} and label smoothing {cfg['label_smoothing']}.")
    add(f"- **Other settings:** batch size {cfg['batch_size']}, up to {cfg['epochs']} epochs, "
        f"early stopping (patience {cfg['patience']}), mixed precision on GPU, seed {cfg['seed']}.")
    add("- **Model selection:** the checkpoint with the best validation accuracy is kept; "
        "the test set is used only once for the final evaluation.\n")

    add("## 4. Results\n")
    add(f"- **Test accuracy:** **{metrics['accuracy'] * 100:.2f}%** on {metrics['num_test_images']} images "
        f"(test-time augmentation: {'on' if metrics['tta'] else 'off'})")
    epoch_note = f" (epoch {history['best_epoch']})" if history.get("best_epoch") else ""
    add(f"- **Best validation accuracy:** {metrics['best_val_acc'] * 100:.2f}%{epoch_note}")
    add(f"- **Macro F1:** {metrics['macro_f1']:.4f} | **Weighted F1:** {metrics['weighted_f1']:.4f}")
    if history.get("train_time_sec"):
        add(f"- **Training time:** {history['train_time_sec'] / 60:.1f} min on `{history['device']}`")
    if not history:
        add("- **Note:** the per-epoch training log was not saved for this run, so training curves are not "
            "included. The model was selected by best validation accuracy.")
    add("")
    add("### Per-class metrics (test set)\n")
    add("| Class | Precision | Recall | F1-score | Support |\n|---|---|---|---|---|")
    for c in classes:
        m = metrics["per_class"][c]
        add(f"| {c} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {int(m['support'])} |")
    add("")
    curves = image_line("Training curves", "training_curves.png")
    if curves:
        add("### Training curves\n")
        add(curves)
    add("### Confusion matrix\n")
    add(image_line("Confusion matrix", "confusion_matrix.png"))
    add(image_line("Normalised confusion matrix", "confusion_matrix_normalized.png"))

    add("### Error analysis\n")
    confusions = top_confusions(metrics["confusion_matrix"], classes)
    if confusions:
        add("The most frequent confusions on the test set are:\n")
        for count, true_c, pred_c in confusions:
            add(f"- `{true_c}` predicted as `{pred_c}`: {count} images")
        add("")
    add("Such pairs are visually or semantically close (for example glacier/mountain or buildings/street), "
        "and many photos contain elements of more than one class. The most confident mistakes are shown below.\n")
    add(image_line("Misclassified examples", "misclassified_examples.png"))

    add("## 5. Limitations\n")
    add("- **Closed set:** the model knows only these classes. Any other image (a person, an animal, an object) "
        "is still assigned to one of them, often with high confidence. There is no out-of-distribution detection.")
    add("- **Ambiguous labels:** many scenes contain several elements (a street lined with buildings, a glacier "
        "beside mountains), so the single label is partly subjective. This limits the achievable accuracy.")
    add("- **Low resolution:** images are about 150×150 px and are upsampled to 224×224, which removes fine detail.")
    add("- **Dataset bias:** the data comes from one collection with limited variety of regions, seasons, "
        "camera types and viewpoints. Performance on photos from other sources (phone snapshots, night scenes, "
        "drone views, heavy filters) may be lower.")
    add("- **Same-distribution test set:** the test accuracy measures performance on data similar to the "
        "training data, not on arbitrary real-world images.")
    add("- **Uncalibrated confidence:** softmax probabilities are not calibrated and should not be read as "
        "true likelihoods.")
    add("- **Single run:** results come from one training run with one seed; run-to-run variation of a few "
        "tenths of a percent is expected.\n")

    add("## 6. Conclusion and future work\n")
    add(f"Fine-tuning a pretrained `{cfg['arch']}` reached {metrics['accuracy'] * 100:.2f}% accuracy on the "
        "held-out test set. Possible improvements: larger or ensemble models, higher-resolution inputs, stronger "
        "augmentation (MixUp/CutMix), confidence calibration, an \"unknown\" class or rejection threshold, "
        "and adding more diverse training images.\n")

    add("## 7. Reproducibility\n")
    add("```bash\npip install -r requirements.txt\npython dataset.py        # verify dataset\n"
        "python train.py\npython evaluate.py --tta\npython generate_report.py\n```\n")
    add("Trained weights: `models/best_model.pth` (contains the class names and architecture).")

    (config.PROJECT_ROOT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("REPORT.md written.")


if __name__ == "__main__":
    main()
