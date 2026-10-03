"""Mengatur data/raw/<kelas> menjadi data/{train,val,test}/<kelas>/.

Penting: frame berasal dari video yang sama dan berurutan. Split dilakukan
dalam blok waktu berurutan (bukan acak) supaya frame train dan val tidak
berdekatan -- kalau acak, val hanya mengukur kemiripan frame, bukan kemampuan
generalisasi.

Sumber kelas dibaca dari <data>/raw/ kalau folder itu ada, kalau tidak dari
<data>/ langsung. Hasil split tetap ditulis ke <data>/{train,val,test}/.
"""

import argparse
import random
import re
import shutil
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}
FRAME_RE = re.compile(r"frame_(\d+)")


def parse_args():
    p = argparse.ArgumentParser(description="Split data per kelas menjadi train/val/test")
    p.add_argument("--data", default="../data",
                   help="root data; kelas dibaca dari <data>/raw/ kalau ada")
    p.add_argument("--source", default=None,
                   help="folder sumber kelas, default <data>/raw kalau ada, "
                        "kalau tidak <data> sendiri")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                   metavar=("TRAIN", "VAL", "TEST"), help="Jumlahnya harus 1.0")
    p.add_argument("--copy", action="store_true",
                   help="Salin file (default: pindahkan/link dari folder sumber)")
    p.add_argument("--link", action="store_true",
                   help="Buat symlink, hemat kuota disk (tidak bisa dipakai di Windows tanpa developer mode)")
    p.add_argument("--keep-src", action="store_true",
                   help="Folder sumber tidak dihapus setelah disalin. Master di "
                        "data/raw/ sebaiknya selalu dijaga")
    p.add_argument("--block", type=int, default=5,
                   help="Jumlah frame berurutan per blok (Semakin besar,-Isolation antar split makin kuat)")
    p.add_argument("--classes", nargs="+", default=None, metavar="NAMA",
                   help="Hanya proses kelas tertentu. Pakai ini kalau menambah kelas baru, "
                        "supaya split kelas lama tidak ikut diacak ulang.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def frame_index(path: Path) -> int:
    m = FRAME_RE.search(path.name)
    return int(m.group(1)) if m else 0


def contiguous_blocks(items, rng, chunk=5):
    """Potong list terurut jadi potongan waktu berurutan, lalu acak urutan potongan.

    Setiap potongan berisi frame yang bersebelahan (statis), sehingga frame dalam
    satu split tidak bersebelahan dengan frame di split lain.
    """
    blocks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    rng.shuffle(blocks)
    return blocks


def place(src: Path, dst: Path, args):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if args.link:
        dst.symlink_to(src.resolve())
    elif args.copy or args.keep_src:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def main():
    args = parse_args()
    data = Path(args.data).resolve()
    if args.source:
        src_root = Path(args.source).resolve()
    elif (data / "raw").is_dir():
        src_root = data / "raw"
    else:
        src_root = data
    out = data
    total = sum(args.ratios)
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"--ratios harus berjumlah 1.0, dapat {total}")

    class_dirs = sorted(
        d for d in src_root.iterdir()
        if d.is_dir() and d.name not in {"train", "val", "test"}
    )
    if not class_dirs:
        raise SystemExit(f"Tidak ada folder kelas di {src_root}.")

    if args.classes:
        wanted = set(args.classes)
        known = {d.name for d in class_dirs}
        missing = wanted - known
        if missing:
            raise SystemExit(
                f"Kelas tidak ada di {src_root}: {sorted(missing)}. "
                f"Yang tersedia: {sorted(known)}"
            )
        class_dirs = [d for d in class_dirs if d.name in wanted]
        if not class_dirs:
            raise SystemExit("--classes tidak menyeleksi apa pun.")

    # Split per kelas diulang dari file yang sama persis akan menghasilkan nama
    # file yang sama, jadi menjalankan ulang untuk kelas lama itu idempoten
    # (selama --seed dan --block tidak diubah). Tetap pakai --classes untuk
    # menambah kelas baru supaya tidak ada yang ikut teracak.
    rng = random.Random(args.seed)
    print(f"Sumber: {[d.name for d in class_dirs]}")
    print(f"Rasio : train={args.ratios[0]:.0%} val={args.ratios[1]:.0%} test={args.ratios[2]:.0%}")

    counts = {}
    for class_dir in class_dirs:
        images = sorted(
            (p for p in class_dir.rglob("*")
             if p.is_file() and p.suffix.lower() in IMAGE_EXT),
            key=frame_index,
        )
        if not images:
            print(f"  {class_dir.name}: tidak ada gambar, dilewati")
            continue

        n_val = round(len(images) * args.ratios[1])
        n_test = round(len(images) * args.ratios[2])
        n_train = len(images) - n_val - n_test

        blocks = contiguous_blocks(images, rng, chunk=args.block)

        buckets = {"train": [], "val": [], "test": []}
        for part in blocks:
            if len(buckets["train"]) < n_train:
                buckets["train"].extend(part)
            elif len(buckets["val"]) < n_val:
                buckets["val"].extend(part)
            else:
                buckets["test"].extend(part)

        print(f"  {class_dir.name}: {len(images)} gambar "
              f"(train={len(buckets['train'])} val={len(buckets['val'])} test={len(buckets['test'])})")

        for split, files in buckets.items():
            for i, src in enumerate(sorted(files, key=frame_index)):
                # nama file menyimpan frame index asal, supaya mudah dilacak
                # ke frames.csv / dicek tidak ada frame kembar antar split
                dst = out / split / class_dir.name / f"{src.stem}__{i:03d}.jpg"
                place(src, dst, args)

        counts[class_dir.name] = {k: len(v) for k, v in buckets.items()}

        if not (args.copy or args.keep_src or args.link):
            leftovers = [p for p in class_dir.rglob("*")
                         if p.is_file() and p.suffix.lower() in IMAGE_EXT]
            for p in leftovers:
                p.unlink()
            for f in ("frames.csv", "extraction.json"):
                (class_dir / f).unlink(missing_ok=True)
            if not any(class_dir.iterdir()):
                class_dir.rmdir()

    print("\nSelesai. Struktur:")
    for split in ("train", "val", "test"):
        for cdir in sorted((out / split).iterdir()) if (out / split).exists() else []:
            n = len(list(cdir.iterdir()))
            print(f"  data/{split}/{cdir.name}  ({n} gambar)")


if __name__ == "__main__":
    main()
