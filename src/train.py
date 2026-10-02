import argparse
import json
import time
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_datasets
from model import ARCHS, DEFAULT_ARCH, build_model, count_params, param_groups


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ResNet (ImageFolder)")
    p.add_argument("--arch", choices=sorted(ARCHS), default=DEFAULT_ARCH,
                   help="arsitektur backbone (default: resnet50)")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="outputs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mode", choices=["feature", "partial", "scratch"], default="feature",
                   help="feature=fc saja, partial=layer4+fc, scratch=semua dari nol")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1,
                   help="LR dikali nilai ini untuk stem (conv1..layer3)")
    p.add_argument("--layer4-lr-mult", type=float, default=1.0,
                   help="LR dikali nilai ini untuk layer4")
    p.add_argument("--fc-lr-mult", type=float, default=1.0,
                   help="LR dikali nilai ini untuk fc/head")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Deprecated, pakai --mode feature")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--tag", default=None,
                   help="Suffix untuk nama file output, mis. 'partial' -> best_partial.pt")
    p.add_argument("--patience", type=int, default=5, help="early stop patience (epoch)")
    p.add_argument("--amp", action="store_true", help="mixed precision di CUDA")
    p.add_argument("--resume", default=None)
    p.add_argument("--plot-only", default=None,
                   help="Hanya buat grafik dari history.json yang ada, tanpa training")
    p.add_argument("--no-plot", action="store_true", help="Matikan pembuatan grafik")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast("cuda", enabled=amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def plot_history(history, out_path, title=None):
    if not HAS_MPL:
        print("matplotlib belum terpasang, grafik dilewati. `pip install matplotlib`")
        return None
    epochs = [h["epoch"] for h in history if "epoch" in h]
    if not epochs:
        print("History kosong, tidak ada grafik.")
        return None

    def series(key):
        return [h.get(key, float("nan")) for h in history if "epoch" in h]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    tr_loss, va_loss = series("train_loss"), series("val_loss")
    axes[0][0].plot(epochs, tr_loss, marker="o", label="train")
    axes[0][0].plot(epochs, va_loss, marker="o", label="val")
    axes[0][0].set_title("Loss per epoch")
    axes[0][0].set_ylabel("loss")
    # celah train-val = tanda overfitting
    axes[0][0].fill_between(epochs, tr_loss, va_loss, color="red", alpha=0.12,
                            label="selisih train-val")
    axes[0][0].legend()
    axes[0][0].grid(alpha=0.3)

    tr_acc, va_acc = series("train_acc"), series("val_acc")
    axes[0][1].plot(epochs, tr_acc, marker="o", label="train")
    axes[0][1].plot(epochs, va_acc, marker="o", label="val")
    axes[0][1].set_title("Accuracy per epoch")
    axes[0][1].set_ylabel("accuracy")
    axes[0][1].set_ylim(0, 1.05)
    axes[0][1].legend()
    axes[0][1].grid(alpha=0.3)
    # epoch dengan val_acc tertinggi
    best_i = max(range(len(va_acc)), key=lambda i: va_acc[i])
    axes[0][1].axvline(epochs[best_i], ls="--", color="green", alpha=0.7,
                       label=f"best epoch {epochs[best_i]} ({va_acc[best_i]:.3f})")
    axes[0][1].legend(loc="lower right")

    axes[1][0].bar(epochs, series("time"), color="tab:green", alpha=0.8)
    axes[1][0].set_title("Durasi per epoch (detik)")
    axes[1][0].set_ylabel("detik")

    axes[1][1].plot(epochs, series("lr"), marker="o", color="tab:orange")
    axes[1][1].set_title("Learning rate per epoch")
    axes[1][1].set_ylabel("lr")
    axes[1][1].set_yscale("log")

    for ax in axes.flat:
        ax.set_xlabel("epoch")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Kurva training{f' - {title}' if title else ''} ({len(epochs)} epoch)",
                 fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Grafik disimpan ke {out_path}")
    return out_path


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or args.mode

    if args.plot_only:
        history = json.loads(Path(args.plot_only).read_text())
        plot_history(history, out_dir / f"curves_{tag}.png", title=f"mode={tag}")
        return

    device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"

    train_ds, val_ds, test_ds = build_datasets(args.data, args.image_size)
    if not train_ds.class_to_idx:
        raise SystemExit(f"Tidak ada kelas di {args.data}/train. Isi folder dengan <nama_kelas>/")

    if val_ds.class_to_idx != train_ds.class_to_idx:
        raise SystemExit("Kelas train dan val tidak sama. Pastikan tiap split punya semua kelas.")

    print(f"Jumlah kelas : {len(train_ds.classes)} -> {train_ds.classes}")
    print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds) if test_ds else 0}")
    print(f"Device       : {device} (amp={amp})")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers, pin_memory=device.type == "cuda")
        if test_ds else None
    )

    model = build_model(
        len(train_ds.classes),
        mode=args.mode,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        arch=args.arch,
    ).to(device)

    # mode scratch butuh LR jauh lebih besar: bobot acak, bukan fitur pretrained
    lr_mult = {
        "stem": args.backbone_lr_mult,
        "layer4": args.layer4_lr_mult,
        "fc": args.fc_lr_mult,
    }
    groups = param_groups(model, args.mode, args.lr, lr_mult)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    n_train, n_total = count_params(model)
    print(f"Arsitektur   : {args.arch}")
    print(f"Mode         : {args.mode} (pretrain={'tidak' if args.mode == 'scratch' else 'ya'})")
    print(f"Param dilatih: {n_train:,}/{n_total:,} ({100 * n_train / n_total:.1f}%)")
    for g in groups:
        print(f"  {g['name']:7s} lr={g['lr']:.2e} ({len(g['params'])} tensor)")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch = 0
    best_acc, bad_epochs = 0.0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"Resume dari epoch {start_epoch} (best_acc={best_acc:.4f})")

    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, seen, correct = 0.0, 0, 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            pbar.set_postfix(loss=f"{running / seen:.4f}",
                             acc=f"{correct / seen:.4f}")

        train_loss = running / max(seen, 1)
        train_acc = correct / max(seen, 1)
        val_loss, val_acc = evaluate(model, val_loader, device, amp)
        scheduler.step(val_acc)

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc,
            "lr": optimizer.param_groups[-1]["lr"],
            "time": time.time() - t0, "mode": args.mode,
        })
        improved = val_acc > best_acc
        if improved:
            best_acc, bad_epochs = val_acc, 0
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "val_acc": val_acc, "args": vars(args), "mode": args.mode,
                "class_to_idx": train_ds.class_to_idx,
            }, out_dir / f"best_{tag}.pt")
        else:
            bad_epochs += 1

        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch,
            "val_acc": val_acc, "args": vars(args), "mode": args.mode,
            "class_to_idx": train_ds.class_to_idx,
        }, out_dir / f"last_{tag}.pt")

        print(f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} best={best_acc:.4f} "
              f"{'*' if improved else ''} ({time.time() - t0:.1f}s)")

        if bad_epochs >= args.patience:
            print(f"Early stop: tidak ada perbaikan selama {args.patience} epoch.")
            break

    (out_dir / f"history_{tag}.json").write_text(json.dumps(history, indent=2))
    if not args.no_plot:
        plot_history(history, out_dir / f"curves_{tag}.png", title=f"mode={args.mode}")

    if test_loader:
        ckpt = torch.load(out_dir / f"best_{tag}.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        test_loss, test_acc = evaluate(model, test_loader, device, amp)
        print(f"Test  -> loss={test_loss:.4f} acc={test_acc:.4f}")
        history.append({"test_loss": test_loss, "test_acc": test_acc})
        (out_dir / f"history_{tag}.json").write_text(json.dumps(history, indent=2))

    print(f"Selesai [{args.mode}]. Best val_acc={best_acc:.4f}. "
          f"Checkpoint: {out_dir / f'best_{tag}.pt'}")


if __name__ == "__main__":
    main()