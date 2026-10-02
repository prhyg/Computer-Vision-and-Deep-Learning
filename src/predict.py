import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from dataset import IMAGENET_MEAN, IMAGENET_STD
from model import DEFAULT_ARCH, build_model

# ukuran feature sebelum fc: 512 untuk resnet18, 2048 untuk resnet50
FC_IN_FEATURES = {"resnet18": 512, "resnet50": 2048}


def infer_arch(ckpt, train_args, override=None):
    """Tentukan arsitektur: flag > args checkpoint > key ckpt > deduce dari bentuk fc."""
    if override:
        return override
    for src in (train_args, ckpt):
        arch = src.get("arch")
        if arch in FC_IN_FEATURES:
            return arch
    # checkpoint lama tidak menyimpan arch, nebak dari fc.in_features
    for key, val in ckpt["model"].items():
        if key.startswith("fc.") and key.endswith("weight") and val.dim() == 2:
            n = val.shape[1]
            for arch, feat in FC_IN_FEATURES.items():
                if n == feat:
                    return arch
    return DEFAULT_ARCH


def infer_dropout(ckpt, train_args):
    """Dropout tidak selalu tersimpan di checkpoint, deduce dari fc.1.* yang ada."""
    dropout = train_args.get("dropout")
    if dropout:
        return float(dropout)
    if any(k.startswith("fc.1.") for k in ckpt["model"]):
        print("Catatan: checkpoint pakai dropout di head, menebak --dropout dari struktur fc.")
        return 0.2
    return 0.0


def parse_args():
    p = argparse.ArgumentParser(description="Prediksi dengan checkpoint ResNet")
    p.add_argument("--checkpoint", default="outputs/best.pt")
    p.add_argument("--arch", choices=["resnet18", "resnet50"], default=None,
                   help="Override arsitektur; default diambil dari checkpoint")
    p.add_argument("images", nargs="+", help="path gambar atau folder")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def collect_images(inputs):
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(
                f for f in p.rglob("*")
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ))
        else:
            files.append(p)
    return files


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    arch = infer_arch(ckpt, train_args, args.arch)
    dropout = infer_dropout(ckpt, train_args)
    model = build_model(len(class_to_idx), dropout=dropout, arch=arch)
    model.load_state_dict(ckpt["model"])
    print(f"Arsitektur {arch} (dropout {dropout}) dimuat dari {args.checkpoint}")
    model.to(device).eval()

    size = train_args.get("image_size") or ckpt.get("image_size") or args.image_size
    tf = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    files = collect_images(args.images)
    if not files:
        raise SystemExit("Tidak ada gambar ditemukan.")

    for f in files:
        img = tf(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(img).softmax(1)[0]
        top = probs.topk(min(args.top_k, len(idx_to_class)))
        best = [(idx_to_class[i.item()], p.item()) for p, i in zip(top.values, top.indices)]
        label, conf = best[0]
        detail = ", ".join(f"{n}={c:.3f}" for n, c in best)
        print(f"{f}: {label} ({conf:.4f})  [{detail}]")


if __name__ == "__main__":
    main()