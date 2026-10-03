"""ResNet18 untuk eksperimen 3 mode (feature / partial / scratch).

Folder ini khusus ResNet18. Arsitektur tidak bisa diganti lewat flag supaya
hasil eksperimen selalu memakai backbone yang sama.
"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

ARCH = "resnet18"
DEFAULT_ARCH = ARCH
MODES = ("feature", "partial", "scratch")


def build_model(num_classes: int, mode: str = "feature", dropout: float = 0.0):
    """Bangun ResNet18 sesuai mode eksperimen.

    mode:
      feature  - bobot pretrained dibekukan, hanya fc yang dilatih
      partial  - layer4 + fc dilatih, bagian awal (conv1..layer3) beku
      scratch  - bobot pretrained dibuang, semua layer dilatih dari nol
    """
    if mode not in MODES:
        raise ValueError(f"mode harus salah satu dari {MODES}, bukan {mode!r}")

    if mode == "scratch":
        model = resnet18(weights=None)
    else:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # beku dulu semuanya, lalu buka sesuai mode
    for param in model.parameters():
        param.requires_grad = False

    if mode in ("partial", "scratch"):
        for param in model.parameters():
            param.requires_grad = True
    if mode == "partial":
        for module in (model.conv1, model.bn1, model.maxpool,
                       model.layer1, model.layer2, model.layer3):
            for param in module.parameters():
                param.requires_grad = False

    in_features = model.fc.in_features
    layers = [nn.Dropout(dropout)] if dropout > 0 else []
    layers.append(nn.Linear(in_features, num_classes))
    model.fc = nn.Sequential(*layers)

    if mode == "feature":
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


def param_groups(model: nn.Module, lr: float, lr_mult: dict):
    """Kelompokkan parameter sesuai learning rate tiap bagian.

    lr_mult punya key: "stem" (conv1/bn1/layer1-3), "layer4", "fc".
    Parameter yang tidak boleh dilatih (feature/partial) di-lewat.
    """
    groups = {"stem": [], "layer4": [], "fc": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            key = "fc"
        elif name.startswith("layer4."):
            key = "layer4"
        else:
            key = "stem"
        groups[key].append(param)

    out = []
    for key, params in groups.items():
        if params:
            out.append({"params": params, "lr": lr * lr_mult.get(key, 1.0), "name": key})
    if not out:
        raise ValueError("Tidak ada parameter yang boleh dilatih.")
    return out


def count_params(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
