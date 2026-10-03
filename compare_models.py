"""Bandingkan ResNet18 vs ResNet50 dari hasil training kedua folder.

Script ini tidak training apa pun. Dia membaca output yang sudah ada:

  resnet18_3mode/outputs/compare.json          (3 mode)
  resnet50_1mode/outputs/history_feature.json  (1 mode)

 lalu menggambar perbandingan feature-vs-feature. Mode itu sama di kedua sisi
(feature), jadi yang diukur hanya pengaruh arsitektur: ResNet18 (11,7jt param)
vs ResNet50 (23,5jt param). Mode ResNet18 partial/scratch tetap digambar sebagai
garis konteks tipis, tapi bukan pembanding utama.

Jalankan:
  python compare_models.py
  python compare_models.py --out outputs
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R18_OUT = ROOT / "resnet18_3mode" / "outputs"
R50_OUT = ROOT / "resnet50_1mode" / "outputs"

MAIN_MODE = "feature"


def parse_args():
    p = argparse.ArgumentParser(description="Grafik perbandingan ResNet18 vs ResNet50")
    p.add_argument("--resnet18", default=str(R18_OUT), help="folder output resnet18_3mode")
    p.add_argument("--resnet50", default=str(R50_OUT), help="folder output resnet50_1mode")
    p.add_argument("--out", default=str(ROOT / "outputs"),
                   help="folder untuk compare_models.png dan compare_models.md")
    p.add_argument("--no-context", action="store_true",
                   help="jangan gambar mode partial/scratch sebagai konteks")
    return p.parse_args()


def load_resnet18(folder):
    """compare.json -> dict mode -> result."""
    path = Path(folder) / "compare.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {r["mode"]: r for r in data}


def load_resnet50(folder):
    """history_feature.json -> result shaped like the resnet18 ones.

    File history ditulis ulang sekali lagi di akhir training dengan satu dict
    tambahan {test_loss, test_acc} tanpa key "epoch", jadi itu disaring keluar.
    """
    path = Path(folder) / "history_feature.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    history = [h for h in raw if "epoch" in h]
    if not history:
        return None
    res = {"mode": MAIN_MODE, "arch": "resnet50", "history": history}
    res["best_val_acc"] = max(h["val_acc"] for h in history)
    res["epochs_run"] = len(history)
    for h in raw:
        if "test_acc" in h:
            res["test_acc"], res["test_loss"] = h["test_acc"], h.get("test_loss")
    # jumlah parameter dari checkpoint, biar tidak perlu model
    ckpt = Path(folder) / "best_feature.pt"
    if ckpt.exists():
        try:
            import torch
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
            res["total_params"] = sum(v.numel() for v in sd.values())
            head = sum(v.numel() for k, v in sd.items() if k.startswith("fc."))
            res["trainable_params"] = head
        except Exception as e:  # noqa: BLE001
            print(f"Catatan: gagal baca parameter dari checkpoint ({e})")
    return res


def mean_epoch_time(result):
    times = [h["time"] for h in result["history"] if h.get("time")]
    return sum(times) / len(times) if times else None


def epochs_to_best(result, threshold=1.0):
    for h in result["history"]:
        if h["val_acc"] >= threshold:
            return h["epoch"]
    return None


def make_plot(r18, r18_all, r50, out_path, show_context=True):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib belum ada, grafik dilewati")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def e1(r):
        return epochs_to_best(r) or 0

    main = [("ResNet18 " + MAIN_MODE, r18, "tab:blue"),
            ("ResNet50 " + MAIN_MODE, r50, "tab:red")]
    ctx = []
    if show_context:
        for mode, color in (("partial", "tab:orange"), ("scratch", "tab:green")):
            if r18_all.get(mode):
                ctx.append((f"ResNet18 {mode} (konteks)", r18_all[mode], color))

    # panel 1: val accuracy
    ax = axes[0][0]
    for label, r, c in ctx:
        ax.plot([h["epoch"] for h in r["history"]],
                [h["val_acc"] for h in r["history"]],
                ls=":", lw=1.3, alpha=0.7, color=c, label=label)
    for label, r, c in main:
        ep = [h["epoch"] for h in r["history"]]
        ax.plot(ep, [h["val_acc"] for h in r["history"]], marker="o", ms=3,
                color=c, label=label, lw=2)
        if e1(r):
            ax.axvline(e1(r), ls="--", color=c, alpha=0.4, lw=1)
    ax.axhline(1.0, ls="-", color="black", alpha=0.15, lw=1)
    ax.set_title("Val accuracy per epoch\n(garis putus = epoch pertama val=1.0)")
    ax.set_xlabel("epoch"); ax.set_ylabel("val accuracy"); ax.set_ylim(0, 1.06)
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
    if ctx:
        ax.text(0.02, 0.55, "akurasi mentok di 1.0\n=> pakai loss",
                transform=ax.transAxes, fontsize=9, color="dimgray")

    # panel 2: validation loss
    ax = axes[0][1]
    for label, r, c in ctx:
        ax.plot([h["epoch"] for h in r["history"]],
                [h["val_loss"] for h in r["history"]],
                ls=":", lw=1.3, alpha=0.7, color=c, label=label)
    for label, r, c in main:
        ax.plot([h["epoch"] for h in r["history"]],
                [h["val_loss"] for h in r["history"]],
                marker="o", ms=3, color=c, label=label, lw=2)
    ax.set_yscale("log")
    ax.set_title("Validation loss per epoch (log, makin rendah makin baik)")
    ax.set_xlabel("epoch"); ax.set_ylabel("val loss")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # panel 3: kecepatan, detik per epoch
    ax = axes[1][0]
    labels, times, colors = [], [], []
    for label, r, c in main:
        t = mean_epoch_time(r)
        if t:
            labels.append(label.split()[0]); times.append(t); colors.append(c)
    if show_context:
        for mode in ("partial", "scratch"):
            r = r18.get(mode)
            t = mean_epoch_time(r) if r else None
            if t:
                labels.append(f"r18\n{mode}")
                times.append(t)
                colors.append("tab:orange" if mode == "partial" else "tab:green")
    if times:
        bars = ax.bar(labels, times, color=colors, alpha=0.85)
        for b, v in zip(bars, times):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}s",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(times) * 1.25)
    ax.set_title("Durasi rata-rata per epoch\n(lebih rendah = lebih cepat)")
    ax.set_ylabel("detik"); ax.grid(alpha=0.3, axis="y")

    # panel 4: hasil akhir.
    # Akurasi semuanya mentok di 1.0 jadi tidak membedakan sama sekali. Yang
    # masih membedakan adalah loss (selisih 4 ordeks magnituda) dan jumlah
    # parameter, jadi dua hal itu yang digambar.
    ax = axes[1][1]
    series = main + (ctx if show_context else [])
    names = [label.replace(" (konteks)", "\n(konteks)").replace("ResNet18 ", "r18 ")
             .replace("ResNet50 ", "r50 ") for label, _, _ in series]
    final_val = [r["history"][-1]["val_loss"] for _, r, _ in series]
    tests = [r.get("test_loss", float("nan")) for _, r, _ in series]
    x = range(len(series))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], final_val, w, label="val loss (epoch akhir)",
                color="tab:blue")
    b2 = ax.bar([i + w / 2 for i in x], tests, w, label="test loss", color="tab:cyan")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if h == h:
                ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.4f}",
                        ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(top=max(t for t in final_val + tests if t == t) * 30)
    ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("loss (log, makin rendah makin baik)")
    ax.set_title("Loss akhir + jumlah parameter\n(akurasi semuanya 1.0, jadi loss pembeda)")
    for i, (_, r, _) in enumerate(series):
        tot = r.get("total_params")
        tr = r.get("trainable_params")
        if tot:
            ax.text(i, 0.012, f"{tot/1e6:.1f}M total\n{(tr/1e3 if tr else 0):.1f}k dilatih",
                    ha="center", va="bottom", fontsize=7, transform=ax.get_xaxis_transform())
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("ResNet18 vs ResNet50 (mode feature, data & seed sama)", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Grafik: {out_path}")


def make_table(r18, r50):
    rows = []
    for label, r in (("ResNet18", r18), ("ResNet50", r50)):
        t = mean_epoch_time(r)
        rows.append({
            "model": label, "mode": r["mode"],
            "params": r.get("total_params"), "trainable": r.get("trainable_params"),
            "val": r["best_val_acc"], "test": r.get("test_acc"),
            "val_loss": r["history"][-1]["val_loss"], "test_loss": r.get("test_loss"),
            "epochs": r["epochs_run"], "ep@1.0": epochs_to_best(r),
            "sec_ep": t, "total_min": (t * r["epochs_run"] / 60) if t else None,
        })
    return rows


def render_markdown(rows, extra=()):
    lines = [
        "# Perbandingan ResNet18 vs ResNet50",
        "",
        "Kedua model dilatih dengan data, seed, dan mode yang sama persis",
        f"(mode `{MAIN_MODE}`), jadi selisihnya mencerminkan arsitektur saja.",
        "",
        "**Akurasi tidak dipakai sebagai pembanding utama**: semua konfigurasi",
        "mentok di `val_acc`/`test_acc` 1.0000, jadi angkanya tidak membedakan.",
        "Yang masih membedakan adalah `loss` dan waktu training.",
        "",
        "| model | mode | param total | param dilatih | best val_acc | test_acc | val_loss (akhir) | test_loss | epoch | epoch ke val=1.0 | detik/epoch | total menit |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        def fmt(v, spec="{:.4f}"):
            return spec.format(v) if isinstance(v, (int, float)) and v == v else "-"
        lines.append(
            f"| **{r['model']}** | `{r['mode']}` | {r['params']:,} | "
            f"{r['trainable']:,} | {fmt(r['val'])} | {fmt(r['test'])} | "
            f"{fmt(r['val_loss'])} | {fmt(r['test_loss'])} | {r['epochs']} | "
            f"{r['ep@1.0'] if r['ep@1.0'] else '-'} | "
            f"{fmt(r['sec_ep'], '{:.1f}')} | {fmt(r['total_min'], '{:.1f}')} |"
        )
    for line in extra:
        lines.append(line)
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    r18_all = load_resnet18(args.resnet18)
    r50 = load_resnet50(args.resnet50)

    missing = []
    if r18_all is None:
        missing.append(f"{Path(args.resnet18)}/compare.json "
                       "(jalankan: python resnet18_3mode/compare.py)")
    if r50 is None:
        missing.append(f"{Path(args.resnet50)}/history_feature.json "
                       "(jalankan: python resnet50_1mode/train.py)")
    if missing:
        raise SystemExit("History belum ada:\n  - " + "\n  - ".join(missing))

    if MAIN_MODE not in r18_all:
        raise SystemExit(
            f"Mode {MAIN_MODE!r} tidak ada di compare.json. "
            f"Yang tersedia: {sorted(r18_all)}. "
            f"Ulangi dengan --modes {MAIN_MODE} partial scratch."
        )

    r18 = r18_all[MAIN_MODE]
    print(f"ResNet18 {r18['mode']:8s} val={r18['best_val_acc']:.4f} "
          f"epoch={r18['epochs_run']}")
    print(f"ResNet50 {r50['mode']:8s} val={r50['best_val_acc']:.4f} "
          f"epoch={r50['epochs_run']}")

    make_plot(r18, r18_all, r50, out_dir / "compare_models.png",
              show_context=not args.no_context)

    rows = make_table(r18, r50)
    extra = []
    for mode in ("partial", "scratch"):
        r = r18_all.get(mode)
        if not r:
            continue
        t = mean_epoch_time(r)
        extra.append("")
        extra.append(f"## Konteks: ResNet18 mode `{mode}`")
        extra.append("")
        extra.append(f"- best val_acc {r['best_val_acc']:.4f}, "
                     f"test_acc {r.get('test_acc', float('nan')):.4f}")
        extra.append(f"- val_loss akhir {r['history'][-1]['val_loss']:.4f}, "
                     f"test_loss {r.get('test_loss', float('nan')):.4f}")
        extra.append(f"- {r['epochs_run']} epoch, "
                     f"{f'{t:.1f}' if t else '-'} detik/epoch")
    md = render_markdown(rows, extra)
    (out_dir / "compare_models.md").write_text(md)
    print(f"Tabel  : {out_dir / 'compare_models.md'}")
    print(md)


if __name__ == "__main__":
    main()
