"""Menyiapkan dataset kustom (mis. data/raw) menjadi struktur dataset_yolo/.

Membaca metadata.csv untuk tahu nama kelas, menyalin gambar ke images/,
membuat classes.txt, dan membuat file label .txt kosong (siap diisi anotasi).

Pemakaian : python 04_prepare_dataset.py [folder_data] [folder_dataset_yolo]
Default   : ~/Documents/transfer_learning/data  ->  .../data/dataset_yolo
Langkah   : 1) anotasi (LabelImg/Labelme) pada folder images/ lalu simpan .txt ke labels/
            2) python 01_cek_anotasi.py <folder_dataset_yolo>
            3) python 05_split_dataset_yolo.py
"""
import csv
import shutil
import sys
from pathlib import Path

BASE = Path.home() / "Documents" / "transfer_learning" / "data"
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE / "dataset_yolo"

META = DATA / "metadata.csv"
RAW = DATA / "raw"
if not META.exists() or not RAW.exists():
    sys.exit(f"Dibutuhkan {META} dan {RAW} (struktur dataset kustom). "
             f"Contoh: python 04_prepare_dataset.py <folder_data>")

kelas = sorted({r["class"].strip() for r in csv.DictReader(META.open(encoding="utf-8")) if r["class"].strip()})
(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "labels").mkdir(parents=True, exist_ok=True)
(OUT / "classes.txt").write_text("\n".join(kelas) + "\n")

jumlah = 0
for baris in csv.DictReader(META.open(encoding="utf-8")):
    src = DATA / baris["image"]
    if not src.exists():
        src = RAW / baris["class"] / Path(baris["image"]).name      # fallback: cari di raw/<kelas>/
    if not src.exists():
        print(f"  [!] gambar tidak ditemukan: {baris['image']}")
        continue
    shutil.copy(src, OUT / "images" / src.name)
    label = OUT / "labels" / f"{src.stem}.txt"
    if not label.exists():
        label.write_text("")                                        # kosong = belum dianotasi
    jumlah += 1

print(f"Kelas     : {', '.join(kelas)} (class_id 0..{len(kelas) - 1})")
print(f"Gambar    : {jumlah} disalin ke {OUT / 'images'}")
print(f"Label     : {OUT / 'labels'} (kosong, diisi saat anotasi)")
print(f"\nBerikutnya: anotasi dulu (LabelImg/Labelme), lalu jalankan 01_cek_anotasi.py {OUT}")