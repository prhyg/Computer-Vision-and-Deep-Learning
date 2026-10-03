"""Prediksi pakai checkpoint ResNet50 dari folder resnet50_1mode.

    python predict.py --checkpoint outputs/best_feature.pt ../data/test
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from dataset import IMAGENET_MEAN, IMAGENET_STD
from model import ARCH, build_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FOLDER = Path(__file__).resolve().parent


def infer_dropout(ckpt):
    """Dropout tidak selalu tersimpan di checkpoint, deduce dari fc.1.* yang ada."""
    dropout = ckpt.get("args", {}).get("dropout") or ckpt.get("dropout")
    if dropout:
        return float(dropout)
    if any(k.startswith("fc.1.") for k in ckpt["model"]):
        print("Catatan: checkpoint pakai dropout di head, menebak 0.2 dari struktur fc.")
        return 0.2
    return 0.0


def parse_args():
    p = argparse.ArgumentParser(description="Prediksi dengan checkpoint ResNet50")
    p.add_argument("--checkpoint", default=str(FOLDER / "outputs/best_feature.pt"))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("images", nargs="+", help="path gambar atau folder")
    return p.parse_args()


def collect_images(inputs):
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS))
        else:
            files.append(p)
    return files


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    dropout = infer_dropout(ckpt)
    model = build_model(len(class_to_idx), dropout=dropout)
    model.load_state_dict(ckpt["model"])
    print(f"{ARCH} (dropout {dropout}, mode {ckpt.get('mode', 'feature')}) "
          f"dimuat dari {args.checkpoint}")
    model.to(device).eval()

    size = ckpt.get("image_size") or ckpt.get("args", {}).get("image_size") or args.image_size
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
