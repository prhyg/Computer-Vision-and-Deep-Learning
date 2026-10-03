"""ResNet50 mode feature: bobot pretrained beku, hanya head fc yang dilatih.

Folder ini khusus ResNet50 dengan satu mode training. Arsitektur dan mode sudah
dipatok supaya tidak ada eksperimen lain yang tercampur ke dalam hasilnya.
"""

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

ARCH = "resnet50"
DEFAULT_ARCH = ARCH
MODE = "feature"


def build_model(num_classes: int, dropout: float = 0.0):
    """Bangun ResNet50 dengan head baru di atas backbone pretrained yang beku."""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    layers = [nn.Dropout(dropout)] if dropout > 0 else []
    layers.append(nn.Linear(in_features, num_classes))
    model.fc = nn.Sequential(*layers)

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def param_groups(model: nn.Module, lr: float, lr_mult: dict | None = None):
    """Mode feature cuma punya satu grup yang boleh dilatih: fc.

    lr_mult tidak dipakai di mode ini tapi tetap diterima agar signature sama
    dengan train.py di folder resnet18_3mode.
    """
    params = [p for p in model.fc.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Head fc kosong, tidak ada parameter yang boleh dilatih.")
    return [{"params": params, "lr": lr, "name": "fc"}]


def count_params(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
