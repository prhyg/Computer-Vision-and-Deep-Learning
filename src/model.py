import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)

MODES = ("feature", "partial", "scratch")

# nama arsitektur -> (builder, enum weights pretrained ImageNet)
ARCHS = {
    "resnet18": (resnet18, ResNet18_Weights),
    "resnet50": (resnet50, ResNet50_Weights),
}
DEFAULT_ARCH = "resnet50"


def build_model(
    num_classes: int,
    mode: str = "feature",
    dropout: float = 0.0,
    freeze_backbone: bool | None = None,
    arch: str = DEFAULT_ARCH,
):
    """Bangun ResNet sesuai mode eksperimen (default ResNet50).

    mode:
      feature  - bobot pretrained dibekukan, hanya fc yang dilatih
      partial  - layer4 + fc dilatih, bagian awal (conv1..layer3) beku
      scratch  - bobot pretrained dibuang, semua layer dilatih dari nol

    freeze_backbone (legacy) = True dipetakan ke mode "feature".
    """
    if freeze_backbone:
        mode = "feature"
    if mode not in MODES:
        raise ValueError(f"mode harus salah satu dari {MODES}, bukan {mode!r}")
    if arch not in ARCHS:
        raise ValueError(f"arch harus salah satu dari {tuple(ARCHS)}, bukan {arch!r}")

    builder, weights_enum = ARCHS[arch]
    if mode == "scratch":
        model = builder(weights=None)
    else:
        model = builder(weights=weights_enum.DEFAULT)

    # beku dulu semuanya, lalu buka sesuai mode
    for param in model.parameters():
        param.requires_grad = False

    if mode in ("partial", "scratch"):
        for param in model.parameters():
            param.requires_grad = True
    if mode == "partial":
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.bn1.parameters():
            param.requires_grad = False
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.maxpool.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    layers = [nn.Dropout(dropout)] if dropout > 0 else []
    layers.append(nn.Linear(in_features, num_classes))
    model.fc = nn.Sequential(*layers)

    if mode == "feature":
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


def param_groups(model: nn.Module, mode: str, lr: float, lr_mult: dict):
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
