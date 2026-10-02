# Transfer Learning — ResNet18

Fine-tune ResNet18 pretrained ImageNet untuk klasifikasi citra memakai PyTorch.

## Eksperimen 3 mode

Bandingkan `feature` vs `partial` vs `scratch` dengan data, seed, dan
hyperparameter yang sama:

```bash
cd src
python compare.py --data ../data --epochs 20 --patience 6
```

Output: `outputs/compare.json`, `outputs/results.md` (tabel markdown),
`outputs/compare_modes.png` (loss + akurasi per epoch), 
`outputs/compare_convergence.png`, dan `best_<mode>.pt` per mode.

Learning rate default per mode (`DEFAULT_LR` di `compare.py`):
`feature 3e-4`, `partial 3e-4`, `scratch 1e-2`. Mode scratch butuh LR jauh
lebih besar karena bobotnya acak, bukan fitur pretrained.

## Menyusun dataset

```bash
python split.py --data ../data --keep-src
```

Split dilakukan per blok waktu berurutan, bukan acak. Frame berasal dari video
yang sama, jadi split acak membuat frame train dan val nyaris identik dan
val_acc jadi tidak bermakna. Flag penting: `--ratios 0.7 0.15 0.15`, `--block 5`,
`--keep-src` (folder sumber tidak dihapus), `--copy`, `--link` (symlink).

## Metadata dataset

`data/metadata.csv` mencatat setiap gambar: `class`, `image`, `split`,
`frame_index`, `seconds`. `frame_index` penting karena berasal dari video yang
sama — dipakai untuk memastikan split tidak memisahkan frame bersebelahan.

Kalau `data/raw/` atau `metadata.csv` hilang, bangun ulang dari file hasil
split (nama file masih menyimpan frame index aslinya):

```bash
python build_metadata.py --data ../data          # hardlink, tidak duplikat disk
python build_metadata.py --data ../data --force  # tulis ulang
```

## Struktur

```
data/
  raw/<kelas>/*.jpg     # Master, tidak ikut training
  train/<kelas>/*.jpg
  val/<kelas>/*.jpg
  test/<kelas>/*.jpg
  metadata.csv          # class, image, split, frame_index, seconds
src/
  dataset.py   # ImageFolder + augmentasi
  model.py     # ResNet18 + head baru
  train.py     # loop training
  predict.py   # inferensi
outputs/
  best.pt last.pt history.json curves.png
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
cd src
python train.py --data ../data --epochs 20 --batch-size 32 --amp
```

Opsi berguna:

| Flag | Arti |
|---|---|
| `--freeze-backbone` | deprecated, pakai `--mode feature` |
| `--mode feature` | hanya `fc` yang dilatih (1.026 param) |
| `--mode partial` | `layer4` + `fc` dilatih, stem beku (75% param) |
| `--mode scratch` | semua layer, bobot pretrained dibuang (100% param) |
| `--layer4-lr-mult 0.5` | LR dikali ini untuk layer4 |
| `--fc-lr-mult 1.0` | LR dikali ini untuk head |
| `--tag Nama` | suffix file output, mis. `best_Nama.pt` |
| `--patience 5` | early stop |
| `--resume outputs/last.pt` | lanjutkan training |
| `--dropout 0.2` | regularisasi head |
| `--device cpu` | paksa CPU |
| `--no-plot` | jangan buat grafik |
| `--plot-only ../outputs/history.json` | buat grafik dari history lama, tanpa training |

Perhatikan: `cd src` dulu karena `train.py` mengimpor modul di foldernya sendiri.

## Prediksi

```bash
cd src
python predict.py --checkpoint ../outputs/best.pt ../data/test
```

## Tips

- Rasio split yang umum: train 70%, val 15%, test 15%.
- Setiap kelas wajib ada di train dan val, kalau tidak script berhenti dengan pesan error.
- Kalau dataset kecil (<2000 gambar), mulai dari `--epochs 10 --lr 1e-4 --dropout 0.2`.
- Kalau dataset besar, `--lr 3e-4 --batch-size 64` dan `--amp` membantu.
- `outputs/history.json` berisi akurasi per epoch untuk lihat overfitting.
- `outputs/curves.png` menampilkan kurva train vs val (loss, accuracy, learning rate) otomatis setiap training selesai.
