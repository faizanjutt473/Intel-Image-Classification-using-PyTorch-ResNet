"""Model construction, freezing helpers and checkpoint loading."""
import torch
from torch import nn
from torchvision import models

SUPPORTED_ARCHS = ("resnet18", "resnet50", "mobilenet_v3_large", "efficientnet_b0")


def build_model(arch, num_classes, pretrained=True, dropout=0.3):
    """Load an ImageNet-pretrained backbone and replace its classification head."""
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose from {SUPPORTED_ARCHS}.")
    model = models.get_model(arch, weights="DEFAULT" if pretrained else None)

    if arch.startswith("resnet"):
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif arch == "efficientnet_b0":
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(model.classifier[1].in_features, num_classes)
        )
    else:  # mobilenet_v3_large
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def get_head(model):
    return model.fc if hasattr(model, "fc") else model.classifier


def split_parameters(model):
    """Return (backbone_params, head_params)."""
    head_ids = {id(p) for p in get_head(model).parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    head = [p for p in model.parameters() if id(p) in head_ids]
    return backbone, head


def set_backbone_trainable(model, trainable):
    for param in split_parameters(model)[0]:
        param.requires_grad = trainable


def load_checkpoint(path, device):
    """Rebuild a model from a checkpoint written by train.py. Returns (model, checkpoint)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(checkpoint["arch"], len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint
