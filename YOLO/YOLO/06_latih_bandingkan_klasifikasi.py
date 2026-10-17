"""Praktik terpadu minggu 4-7, tahap 3-5 KLASIFIKASI: melatih, tuning, memilih, dan menguji.

Eksperimen yang dijalankan (pengaturan lain SAMA untuk semua):
  E1 pretrained : model pretrained ImageNet (transfer learning)          -> baseline
  E2 scratch    : arsitektur sama, bobot acak (tanpa transfer)           -> pembanding
  E3 tuning     : E1 + SATU perubahan (default imgsz=224)                -> eksperimen tuning
Model terbaik dipilih dengan top1 pada data VAL, lalu diuji SEKALI pada data TEST.

Struktur data (format ImageFolder, sudah ada di repo):
  data/train/<kelas>/*.jpg, data/val/<kelas>/*.jpg, data/test/<kelas>/*.jpg

Pemakaian:
  python 06_latih_bandingkan_klasifikasi.py --nim 4222101001 --nama "Budi Santoso"
  python 06_latih_bandingkan_klasifikasi.py --nim ... --nama "..." --epochs 30 --ubah epochs=60
  python 06_latih_bandingkan_klasifikasi.py --nim ... --nama "..." --model yolov8n-cls

Membutuhkan: pip install ultralytics, dan folder data/{train,val,test} berisi subdirektori kelas.
"""
import argparse
import csv
import getpass
import platform
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--nim", required=True)
p.add_argument("--nama", required=True)
p.add_argument("--data", default="data", help="root data dengan train/ val/ test/ per kelas")
p.add_argument("--model", default="yolo26n-cls", help="nama model tanpa ekstensi, misal yolo26n-cls atau yolov8n-cls")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--imgsz", type=int, default=224)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--device", default=None, help="0 untuk GPU pertama, cpu untuk CPU (default: otomatis)")
p.add_argument("--ubah", default="imgsz=288", help="SATU perubahan untuk E3, format kunci=nilai")
args = p.parse_args()

# --- Satu perubahan untuk eksperimen tuning (E3) ---
kunci, nilai = args.ubah.split("=", 1)
try:
    nilai = int(nilai)
except ValueError:
    try:
        nilai = float(nilai)
    except ValueError:
        pass                                   # biarkan sebagai teks, misal optimizer=SGD

DATA = Path(args.data)
for split in ("train", "val", "test"):
    if not (DATA / split).is_dir():
        raise SystemExit(f"{DATA / split} tidak ditemukan. Jalankan dulu prepare/split.py --data {DATA}")
if not any(d.is_dir() for d in (DATA / "train").iterdir()):
    raise SystemExit(f"{DATA / 'train'} kosong, tidak ada subdirektori kelas.")
PROJECT = Path("hasil_praktik_klasifikasi").resolve()


def val_loss_akhir(save_dir: Path) -> float:
    """Val loss epoch terakhir dibaca dari results.csv (ClassifyMetrics tidak mengekspos loss)."""
    import csv as _csv
    with (save_dir / "results.csv").open(newline="") as f:
        baris = list(_csv.DictReader(f))
    return float(baris[-1]["val/loss"]) if baris else float("nan")


def latih(nama, model_file, pretrained, **ubah):
    """Melatih satu eksperimen, lalu mengevaluasi best.pt di data val."""
    cfg = dict(data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
               seed=0, deterministic=True, project=str(PROJECT), name=nama,
               exist_ok=True, pretrained=pretrained, plots=True, verbose=False)
    if args.device is not None:
        cfg["device"] = args.device
    cfg.update(ubah)
    print(f"\n=== {nama}: {model_file} | pretrained={pretrained} | {ubah or 'pengaturan dasar'} ===")
    model = YOLO(model_file)
    t0 = time.time()
    model.train(**cfg)
    menit = (time.time() - t0) / 60
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(DATA), split="val", imgsz=cfg["imgsz"], batch=args.batch,
                            plots=False, verbose=False)
    return {"eksperimen": nama, "perubahan": ", ".join(f"{k}={v}" for k, v in ubah.items()) or "-",
            "pretrained": pretrained, "top1": round(m.top1, 4), "top5": round(m.top5, 4),
            "val_loss": round(val_loss_akhir(Path(model.trainer.save_dir)), 4),
            "inferensi_ms": round(m.speed["inference"], 2),
            "waktu_latih_menit": round(menit, 2), "imgsz": cfg["imgsz"], "bobot": str(best)}


hasil = [
    latih("E1_pretrained", f"{args.model}.pt", True),
    latih("E2_scratch", f"{args.model}.yaml", False),
    latih("E3_tuning", f"{args.model}.pt", True, **{kunci: nilai}),
]

# --- Pilih model terbaik dengan data VAL, lalu uji SEKALI di data TEST ---
terbaik = max(hasil, key=lambda r: (r["top1"], -r["val_loss"]))
uji = YOLO(terbaik["bobot"]).val(data=str(DATA), split="test", imgsz=terbaik["imgsz"],
                                  batch=args.batch, plots=False, verbose=False)

garis = "=" * 78
baris = [garis, "  PRAKTIK TERPADU MINGGU 4-7 - KLASIFIKASI", garis,
         f"  NIM / Nama    : {args.nim} / {args.nama}",
         f"  Waktu         : {datetime.now():%d-%m-%Y %H:%M:%S}",
         f"  Komputer/user : {platform.node()} / {getpass.getuser()}",
         f"  Dataset       : {DATA.resolve()}",
         f"  Pengaturan    : model {args.model}, epochs {args.epochs}, imgsz {args.imgsz}, batch {args.batch}",
         garis,
         f"  {'Eksperimen':15s}{'Perubahan':14s}{'top1':>8s}{'top5':>8s}{'val_loss':>10s}{'ms':>7s}{'menit':>8s}"]
for r in hasil:
    baris.append(f"  {r['eksperimen']:15s}{r['perubahan']:14s}{r['top1']:8.3f}{r['top5']:8.3f}"
                 f"{r['val_loss']:10.4f}{r['inferensi_ms']:7.1f}{r['waktu_latih_menit']:8.1f}")
baris += [garis,
          f"  Model terpilih (top1 val tertinggi): {terbaik['eksperimen']}",
          f"  Hasil di data TEST (sekali saja)    : top1 {uji.top1:.3f} | top5 {uji.top5:.3f}",
          garis]
laporan = "\n".join(baris)
print("\n" + laporan)

Path(f"hasil_praktik_klasifikasi_{args.nim}.txt").write_text(laporan + "\n", encoding="utf-8")
with open(f"hasil_praktik_klasifikasi_{args.nim}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(hasil[0].keys()))
    w.writeheader()
    w.writerows(hasil)
print(f"Tersimpan: hasil_praktik_klasifikasi_{args.nim}.txt dan .csv | grafik tiap eksperimen di {PROJECT}/")