# ResNet50 — Mode Feature

Fine-tune **ResNet50** pretrained ImageNet untuk klasifikasi citra dengan mode
`feature`: seluruh backbone dibekukan, hanya head `fc` yang dilatih
(4.098 dari 23.512.130 parameter, 0.02%).

Pilih mode ini karena paling murah: tidak ada gradien yang perlu dihitung sampai
backbone, jadi satu epoch jauh lebih cepat dan tidak ada risiko merusak fitur
pretrained. Eksperimen `partial` / `scratch` ada di folder `resnet18_3mode/`.

Folder ini mandiri — `dataset.py` dan `model.py` di sini adalah salinan, tidak
mengimpor dari folder lain. Arsitektur dan mode dipatok, tidak ada flag
`--arch` / `--mode`.

## Jalankan

Bisa dari root repo atau dari dalam folder ini, path defaultnya sudah relatif ke file:

```bash
python resnet50_1mode/train.py --epochs 20 --batch-size 32 --amp
```

## Output

Semua ditulis ke `resnet50_1mode/outputs/`:

| File | Isi |
|---|---|
| `best_feature.pt` | checkpoint dengan val_acc tertinggi |
| `last_feature.pt` | checkpoint epoch terakhir, untuk `--resume` |
| `history_feature.json` | loss/akurasi/lr per epoch + hasil test |
| `curves_feature.png` | kurva train vs val (loss, akurasi, durasi, lr) |

Lanjutin training yang terputus:

```bash
python resnet50_1mode/train.py --resume resnet50_1mode/outputs/last_feature.pt --epochs 30
```

Buat ulang grafik dari history yang sudah ada, tanpa training:

```bash
python resnet50_1mode/train.py --plot-only resnet50_1mode/outputs/history_feature.json
```

## Prediksi

```bash
python resnet50_1mode/predict.py --checkpoint resnet50_1mode/outputs/best_feature.pt data/test
```

Dropout dan ukuran gambar diambil otomatis dari checkpoint.

## Perbandingan dengan ResNet18

`resnet18_3mode/` punya 3 mode; perbandingan dengan mode `feature` di sini
dibuat dari root:

```bash
python compare_models.py
```

## Flag berguna

| Flag | Arti |
|---|---|
| `--lr 1e-3` | learning rate head |
| `--epochs 20` | batas epoch |
| `--patience 5` | early stop |
| `--batch-size 32` | batch size |
| `--dropout 0.2` | dropout di head, regularisasi untuk dataset kecil |
| `--label-smoothing 0.1` | label smoothing |
| `--weight-decay 1e-4` | weight decay AdamW |
| `--amp` | mixed precision di CUDA |
| `--tag Nama` | suffix file output, mis. `best_Nama.pt` |
| `--resume .../last_feature.pt` | lanjutkan training |
| `--data ../data` | ganti lokasi dataset |
| `--no-plot` | jangan buat grafik |
| `--device cpu` | paksa CPU |
