"""Bangun ulang data/raw/ dan data/metadata.csv dari hasil split.

Split menaruh file di data/{train,val,test}/<kelas>/ dengan nama
<stem_asal>__<urutan>.jpg, jadi frame index asal masih bisa dibaca dari nama
file. Script ini roadblocks file itu kembali ke data/raw/<kelas>/ dengan nama
asli, lalu menulis satu metadata.csv yang menggabungkan info video, posisi
waktu, dan splitassignment-nya.

Hardlink dipakai supaya 200 gambar tidak tergandakan di disk.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}
FRAME_RE = re.compile(r"frame_(\d+)")
# suffix urutan yang ditambahkan split, mis. "...__000.jpg" -> "...jpg"
ORDER_RE = re.compile(r"__\d+(?=\.[^.]+$)")
SPLITS = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(description="Rekonstruksi data/raw + metadata.csv")
    p.add_argument("--data", default="../data")
    p.add_argument("--mode", choices=["hardlink", "copy", "move"], default="hardlink",
                   help="cara isi raw/: hardlink (default), copy, atau move")
    p.add_argument("--fps", type=float, default=29.96414341760006,
                   help="FPS video untuk mengukur detik dari frame index")
    p.add_argument("--force", action="store_true",
                   help="tulis ulang Though file yang sudah ada di raw/")
    return p.parse_args()


def place(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and mode == "hardlink":
        dst.unlink()
    if mode == "hardlink":
        dst.hardlink_to(src)
    elif mode == "copy":
        dst.write_bytes(src.read_bytes())
    else:
        src.rename(dst)


def main():
    args = parse_args()
    data = Path(args.data).resolve()
    raw = data / "raw"

    # kumpulkan: kelas -> frame_index -> (path, split)
    entries = defaultdict(dict)
    for split in SPLITS:
        sdir = data / split
        if not sdir.exists():
            continue
        for class_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() not in IMAGE_EXT:
                    continue
                m = FRAME_RE.search(f.name)
                if not m:
                    continue
                idx = int(m.group(1))
                if idx in entries[class_dir.name]:
                    raise SystemExit(
                        f"Frame {idx} duplikat di kelas {class_dir.name}: "
                        f"{entries[class_dir.name][idx][0].name} dan {f.name}. "
                        "Jangan rebuild dua kali tanpa --force."
                    )
                entries[class_dir.name][idx] = (f, split)

    if not entries:
        raise SystemExit(f"Tidak ada gambar di {data}/train|val|test")

    total = sum(len(v) for v in entries.values())
    print(f"{total} gambar dari {len(entries)} kelas -> {raw}")

    rows = []
    for cls in sorted(entries):
        for idx in sorted(entries[cls]):
            src, split = entries[cls][idx]
            # kembalikan ke nama asal: split menambah suffix "__000" untuk
            # menjaga urutan, dan itu harus dibuang supaya raw/ punya nama
            # yang sama dengan file asli hasil ekstraksi video
            orig = ORDER_RE.sub("", src.name)
            if FRAME_RE.search(orig) is None:
                continue
            dst = raw / cls / orig
            if dst.exists() and not args.force:
                continue
            place(src, dst, args.mode)
            rows.append({
                "class": cls,
                "image": f"raw/{cls}/{orig}",
                "split": split,
                "frame_index": idx,
                "seconds": round(idx / args.fps, 2),
            })

    meta = data / "metadata.csv"
    fieldnames = ["class", "image", "split", "frame_index", "seconds"]
    with meta.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (r["class"], r["frame_index"])):
            writer.writerow(r)

    print(f"Metadata: {meta} ({len(rows)} baris)")
    print("\nRingkasan per kelas x split:")
    tally = defaultdict(int)
    for r in rows:
        tally[(r["class"], r["split"])] += 1
    for cls in sorted(entries):
        parts = "  ".join(f"{s}={tally[(cls, s)]}" for s in SPLITS)
        print(f"  {cls:14s} {parts}  total={len(entries[cls])}")
    print(f"\nTotal: {len(rows)} gambar | {raw}")


if __name__ == "__main__":
    main()
