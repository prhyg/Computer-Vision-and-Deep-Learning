"""Eksperimen ResNet18 3 mode: feature vs partial vs scratch, lalu bandingkan.

Semua mode memakai data, seed, dan hyperparameter yang sama supaya perbandingan
adil. Bedanya cuma bagian mana yang dilatih:

  feature  - hanya fc (head), backbone pretrained beku
  partial  - layer4 + fc, stem (conv1..layer3) beku
  scratch  - semua layer, bobot pretrained dibuang

Catatan penting untuk mode scratch: bobot acak butuh learning rate jauh lebih
besar dari mode yang pakai pretrained. Kalau --lr dibiarkan sama, scratch akan
nyaris tidak bisa converge. Default script ini memakai LR berbeda per mode.

Output ditulis ke outputs/:
  best_<mode>.pt, history_<mode>.json, compare.json, results.md,
  compare_modes.png, compare_convergence.png
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_datasets
from model import ARCH, build_model, count_params, param_groups

MODES = ("feature", "partial", "scratch")

# Path default diselesaikan relatif ke file ini, jadi script bisa dijalankan dari
# mana saja (dari root repo atau dari dalam folder ini) tanpa adjust flag.
FOLDER = Path(__file__).resolve().parent
DEFAULT_DATA = FOLDER.parent / "data"
DEFAULT_OUT = FOLDER / "outputs"

# Learning rate per mode.
# feature  - hanya fc dilatih, LR besar aman karena bobot pretrained sudah bagus
# partial  - layer4 ikut berubah, LR lebih kecil supaya fitur preset tidak rusak
# scratch  - bobot acak, butuh LR terbesar supaya bisa converge dari nol
DEFAULT_LR = {"feature": 1e-3, "partial": 1e-4, "scratch": 1e-2}


def parse_args():
    p = argparse.ArgumentParser(
        description=f"Bandingkan 3 mode training pada {ARCH} (ImageFolder)")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES),
                   help="mode yang dijalankan (default: semua 3)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--layer4-lr-mult", type=float, default=1.0)
    p.add_argument("--fc-lr-mult", type=float, default=1.0)
    p.add_argument("--lr", metavar="MODE=LR", action="append", default=[],
                   help="Override LR satu mode, bisa diulang. Mis. --lr feature=5e-4")
    p.add_argument("--no-plot", action="store_true", help="Matikan pembuatan grafik")
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def run_mode(args, mode, train_ds, val_ds, test_ds, device):
    set_seed(args.seed)
    lr = args.lr_override.get(mode, DEFAULT_LR[mode])
    print(f"\n{'=' * 70}\nMODE {mode.upper()}  ({ARCH}, lr={lr:.0e}, seed={args.seed})\n{'=' * 70}")

    model = build_model(len(train_ds.classes), mode=mode, dropout=args.dropout).to(device)
    n_train, n_total = count_params(model)
    print(f"Param dilatih: {n_train:,}/{n_total:,} ({100 * n_train / n_total:.1f}%)")

    lr_mult = {
        "stem": args.backbone_lr_mult,
        "layer4": args.layer4_lr_mult,
        "fc": args.fc_lr_mult,
    }
    groups = param_groups(model, lr, lr_mult)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    for g in groups:
        print(f"  {g['name']:7s} lr={g['lr']:.2e} ({len(g['params'])} tensor)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers, pin_memory=device.type == "cuda")
        if test_ds else None
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=2)
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_acc, best_state, bad = 0.0, None, 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running = seen = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"[{mode}] {epoch + 1}/{args.epochs}")
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
            pbar.set_postfix(loss=f"{running / seen:.4f}")

        train_loss = running / max(seen, 1)
        # train_acc diambil dari train_loader dalam mode eval, bukan dari batch
        # training (argmax training sedang berubah tiap step, jadi tidak fair)
        model.eval()
        tcorr = tseen = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                tcorr += (model(images).argmax(1) == labels).sum().item()
                tseen += labels.size(0)
        train_acc = tcorr / max(tseen, 1)

        val_loss, val_acc = evaluate(model, val_loader, device, amp)
        scheduler.step(val_acc)
        epoch_time = time.time() - t0
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc,
            "lr": optimizer.param_groups[-1]["lr"], "time": epoch_time,
            "mode": mode, "arch": ARCH,
        })

        improved = val_acc > best_acc
        if improved:
            best_acc, bad = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  [{mode}] epoch {epoch + 1}: train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"best={best_acc:.4f} {'*' if improved else ''} ({epoch_time:.1f}s)")
        if bad >= args.patience:
            print(f"  [{mode}] early stop")
            break

    result = {
        "arch": ARCH, "mode": mode, "lr": lr, "seed": args.seed,
        "best_val_acc": best_acc, "epochs_run": len(history),
        "trainable_params": n_train, "total_params": n_total,
        "history": history,
    }
    if test_loader and best_state:
        model.load_state_dict(best_state)
        model.to(device)
        test_loss, test_acc = evaluate(model, test_loader, device, amp)
        result["test_loss"], result["test_acc"] = test_loss, test_acc
        print(f"  [{mode}] TEST: acc={test_acc:.4f} loss={test_loss:.4f}")

    return result, best_state


def epochs_to_best(history, threshold=1.0):
    """Epoch pertama yang mencapai threshold, atau None."""
    for h in history:
        if h["val_acc"] >= threshold:
            return h["epoch"]
    return None


def make_table(results):
    lines = [
        f"{'mode':9s} {'lr':>7s} {'param dilatih':>16s} {'val':>7s} {'test':>7s} "
        f"{'epoch':>6s} {'ep@1.0':>8s}",
        "-" * 70,
    ]
    for r in results:
        n, tot = r["trainable_params"], r["total_params"]
        pct = 100 * n / tot
        ta = f"{r['test_acc']:.4f}" if "test_acc" in r else "-"
        e1 = epochs_to_best(r["history"])
        lines.append(
            f"{r['mode']:9s} {r['lr']:>7.0e} {n:>10,} ({pct:>3.0f}%) "
            f"{r['best_val_acc']:>7.4f} {ta:>7s} {r['epochs_run']:>6d} "
            f"{str(e1 if e1 else '-'):>8s}"
        )
    return "\n".join(lines)


def markdown_table(results, classes, n_train, n_val, n_test, args):
    lines = [
        f"# Hasil perbandingan 3 mode — {ARCH}",
        "",
        f"Data: {n_train} train / {n_val} val / {n_test} test, "
        f"kelas `{', '.join(classes)}`, backbone `{ARCH}`, seed {args.seed}, "
        f"epochs maks {args.epochs}, patience {args.patience}, "
        f"batch {args.batch_size}, dropout {args.dropout}.",
        "",
        "| mode | learning rate | param dilatih | % | best val_acc | test_acc | test_loss | epoch ke val=1.0 | epoch |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        n, tot = r["trainable_params"], r["total_params"]
        e1 = epochs_to_best(r["history"])
        ta = f"{r['test_acc']:.4f}" if "test_acc" in r else "-"
        tl = f"{r['test_loss']:.4f}" if "test_loss" in r else "-"
        lines.append(
            f"| `{r['mode']}` | {r['lr']:.0e} | {n:,} | {100 * n / tot:.0f}% | "
            f"{r['best_val_acc']:.4f} | {ta} | {tl} | "
            f"{e1 if e1 else '-'} | {r['epochs_run']} |"
        )
    lines += [
        "",
        "## Arti kolom",
        "",
        "- **param dilatih** — jumlah bobot yang di-update, bukan total bobot model.",
        "- **epoch ke val=1.0** — epoch pertama val_acc mencapai 1.0, proxy kecepatan konvergensi.",
        "- **test_loss** — nilai loss, bukan cuma akurasi. Acc 1.0 dengan loss besar",
        "  berarti model yakin tapi belum presisi.",
        "",
    ]
    return "\n".join(lines)


def make_plot(results, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib belum ada, grafik dilewati")
        return

    colors = {"feature": "tab:blue", "partial": "tab:orange", "scratch": "tab:red"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for r in results:
        ep = [h["epoch"] for h in r["history"]]
        c = colors.get(r["mode"], "gray")
        axes[0].plot(ep, [h["train_loss"] for h in r["history"]], color=c,
                     label=f"{r['mode']} train")
        axes[0].plot(ep, [h["val_loss"] for h in r["history"]], color=c, ls="--",
                     alpha=0.6, label=f"{r['mode']} val")
        axes[1].plot(ep, [h["train_acc"] for h in r["history"]], color=c,
                     label=f"{r['mode']} train")
        axes[1].plot(ep, [h["val_acc"] for h in r["history"]], color=c, ls="--",
                     alpha=0.6, label=f"{r['mode']} val")

    axes[0].set_yscale("log")
    axes[0].set_title("Loss per epoch (log)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")

    axes[1].set_title(f"Akurasi per epoch — {ARCH}")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].set_ylim(0, 1.05)
    axes[1].axhline(1.0, ls="-", color="green", alpha=0.25, lw=1)
    for r in results:
        e1 = epochs_to_best(r["history"])
        if e1:
            axes[1].axvline(e1, ls=":", color=colors.get(r["mode"], "gray"), alpha=0.45)

    # panel 3: param dilatih per mode, satu-satunya metrik yang selalu membedakan
    x = range(len(results))
    w = 0.38
    tests = [r.get("test_acc", float("nan")) for r in results]
    vals = [r["best_val_acc"] for r in results]
    axes[2].bar([i - w / 2 for i in x], vals, w, label="val_acc", color="tab:blue")
    axes[2].bar([i + w / 2 for i in x], tests, w, label="test_acc", color="tab:cyan")
    for i, v in enumerate(vals):
        axes[2].text(i - w / 2, v + .01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(tests):
        if v == v:
            axes[2].text(i + w / 2, v + .01, f"{v:.3f}", ha="center", fontsize=8)
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels([r["mode"] for r in results])
    axes[2].set_ylim(0, 1.06)
    axes[2].set_title("Akurasi akhir")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    # panel terpisah: konvergensi, biasanya ini yang jadi pembeda nyata
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    for r in results:
        ep = [h["epoch"] for h in r["history"]]
        ax.plot(ep, [h["val_acc"] for h in r["history"]], marker="o",
                color=colors.get(r["mode"], "gray"), label=f"{r['mode']}")
        e1 = epochs_to_best(r["history"])
        if e1:
            ax.axvline(e1, ls=":", color=colors.get(r["mode"], "gray"), alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("epoch"); ax.set_ylabel("val accuracy")
    ax.set_title("Kecepatan konvergensi (garis putus = epoch pertama val=1.0)")
    ax.legend(); ax.grid(alpha=0.3)
    fig2.tight_layout()
    conv_path = out_path.with_name("compare_convergence.png")
    fig2.savefig(conv_path, dpi=120)
    plt.close(fig2)
    print(f"Grafik perbandingan: {out_path}")
    print(f"Grafik konvergensi: {conv_path}")


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    args.lr_override = {}
    for item in args.lr:
        if "=" not in item:
            raise SystemExit(f"--lr harus MODE=LR, dapat {item!r}")
        key, _, val = item.partition("=")
        key = key.strip()
        if key not in DEFAULT_LR:
            raise SystemExit(f"Mode {key!r} tidak dikenal. Pilihan: {list(DEFAULT_LR)}")
        try:
            args.lr_override[key] = float(val)
        except ValueError:
            raise SystemExit(f"Nilai LR tidak valid: {val!r}")
    if args.lr_override:
        print(f"LR override: "
              f"{ {k: f'{v:.0e}' for k, v in args.lr_override.items()} }")

    train_ds, val_ds, test_ds = build_datasets(args.data, args.image_size)
    if not train_ds.class_to_idx:
        raise SystemExit(f"Tidak ada kelas di {args.data}/train. Isi folder dengan <nama_kelas>/")
    if val_ds.class_to_idx != train_ds.class_to_idx:
        raise SystemExit("Kelas train dan val tidak sama. Pastikan tiap split punya semua kelas.")

    print(f"Arsitektur : {ARCH}")
    print(f"Kelas      : {train_ds.classes}")
    print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds) if test_ds else 0}")
    print(f"Device     : {device}")

    results, bests = [], {}
    for mode in args.modes:
        res, state = run_mode(args, mode, train_ds, val_ds, test_ds, device)
        results.append(res)
        (out / f"history_{mode}.json").write_text(json.dumps(res["history"], indent=2))
        if state is not None:
            bests[mode] = state
            torch.save({"model": state, "mode": mode, "arch": ARCH,
                        "dropout": args.dropout, "image_size": args.image_size,
                        "lr": res["lr"], "val_acc": res["best_val_acc"],
                        "class_to_idx": train_ds.class_to_idx}, out / f"best_{mode}.pt")

    (out / "compare.json").write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print(f"HASIL PERBANDINGAN {ARCH}")
    print("=" * 70)
    print(make_table(results))

    if not args.no_plot:
        make_plot(results, out / "compare_modes.png")

    n_test = len(test_ds) if test_ds else 0
    (out / "results.md").write_text(
        markdown_table(results, train_ds.classes, len(train_ds), len(val_ds), n_test, args)
    )
    print(f"\nTabel markdown: {out / 'results.md'}")
    print(f"Detail json   : {out / 'compare.json'}")
    print(f"Checkpoint    : {out}/best_<mode>.pt")


if __name__ == "__main__":
    main()
