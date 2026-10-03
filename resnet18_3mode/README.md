# ResNet18 — Eksperimen 3 Mode

Bandingkan `feature` vs `partial` vs `scratch` pada **ResNet18**. Semua mode memakai
data, seed, dan hyperparameter yang sama, jadi yang dibandingkan hanya bagian mana
yang dilatih:

| mode | yang dilatih | param |
|---|---|---|
| `feature` | hanya head `fc`, backbone pretrained beku | 1.026 |
| `partial` | `layer4` + `fc`, stem (`conv1`..`layer3`) beku | 8.394.754 (75%) |
| `scratch` | semua layer, bobot pretrained dibuang | 11.177.538 (100%) |

Folder ini mandiri — `dataset.py` dan `model.py` di sini adalah salinan, tidak
mengimpor dari folder lain. Arsitektur dipatok ResNet18, tidak ada flag `--arch`.

## Jalankan

Bisa dari root repo atau dari dalam folder ini, path defaultnya sudah relatif ke file:

```bash
python resnet18_3mode/compare.py --epochs 20 --patience 5
```

Semua 3 mode:

```bash
python resnet18_3mode/compare.py --epochs 20 --batch-size 32 --patience 5
```

Cuma sebagian mode:

```bash
python resnet18_3mode/compare.py --modes feature partial --epochs 20
```

Learning rate default per mode (`DEFAULT_LR` di `compare.py`):
`feature 1e-3`, `partial 1e-4`, `scratch 1e-2`. Mode `scratch` butuh LR jauh lebih
besar karena bobotnya acak, bukan fitur pretrained — kalau dipaksa sama, dia nyaris
tidak bisa converge. Override per mode:

```bash
python resnet18_3mode/compare.py --lr feature=5e-4 --lr scratch=3e-3
```

## Output

Semua ditulis ke `resnet18_3mode/outputs/`:

| File | Isi |
|---|---|
| `best_<mode>.pt` | checkpoint terbaik per mode (pilih sesuai val_acc) |
| `history_<mode>.json` | loss/akurasi per epoch mode itu |
| `compare.json` | ringkasan semua mode |
| `results.md` | tabel markdown perbandingan |
| `compare_modes.png` | loss + akurasi + akurasi akhir |
| `compare_convergence.png` | kecepatan konvergensi |

Perbandingan dengan ResNet50 (mode feature) dibuat terpisah dari root:

```bash
python compare_models.py
```

## Prediksi

```bash
python resnet18_3mode/predict.py --checkpoint resnet18_3mode/outputs/best_partial.pt data/test
```

Mode dan dropout diambil otomatis dari checkpoint.

## Flag berguna

| Flag | Arti |
|---|---|
| `--modes feature partial` | jalankan mode tertentu saja |
| `--lr MODE=LR` | override LR satu mode, bisa diulang |
| `--epochs 20` | batas epoch |
| `--patience 5` | early stop |
| `--batch-size 32` | batch size |
| `--dropout 0.2` | regularisasi head |
| `--seed 42` | seed, sama untuk semua mode |
| `--amp` | mixed precision di CUDA |
| `--data ../data` | ganti lokasi dataset |
| `--no-plot` | jangan buat grafik |
| `--device cpu` | paksa CPU |
