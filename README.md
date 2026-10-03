# Klasifikasi Kotak dan Sampah — Transfer Learning

Klasifikasi citra 3 kelas (`box_cokelat`, `box_merah`, `trash`) dengan PyTorch.
Dua eksperimen terpisah, masing-masing di foldernya sendiri:

| Folder | Backbone | Mode | Output |
|---|---|---|---|
| [`resnet18_3mode/`](resnet18_3mode/) | ResNet18 | 3 mode: `feature`, `partial`, `scratch` | perbandingan + checkpoint tiap mode |
| [`resnet50_1mode/`](resnet50_1mode/) | ResNet50 | 1 mode: `feature` | checkpoint tunggal |

Keduanya memakai **dataset yang sama persis** (`data/`) dengan seed dan
hyperparameter yang sama, jadi hasil dua folder bisa dibandingkan langsung.
Tiap folder mandiri: `dataset.py` dan `model.py` di dalamnya adalah salinan,
tidak saling mengimpor.

## Mulai cepat

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ResNet18, bandingkan 3 mode
python resnet18_3mode/compare.py --epochs 20 --patience 5

# ResNet50, mode feature
python resnet50_1mode/train.py --epochs 20 --batch-size 32 --amp

# Grafik perbandingan kedua model
python compare_models.py
```

Path `--data` dan `--out` sudah relatif ke masing-masing file, jadi script bisa
dijalankan dari root repo atau dari dalam foldernya tanpa adjust.

## Struktur

```
data/
  raw/<kelas>/*.jpg     # master, tidak ikut training
  train/<kelas>/*.jpg   # 232 gambar (70 + 70 + 92)
  val/<kelas>/*.jpg     # 50 gambar  (15 + 15 + 20)
  test/<kelas>/*.jpg    # 50 gambar  (15 + 15 + 20)
  metadata.csv          # class, image, split, frame_index, seconds (332 baris)
resnet18_3mode/
  dataset.py model.py compare.py predict.py
  outputs/              # best_<mode>.pt, compare.json, results.md, grafik
resnet50_1mode/
  dataset.py model.py train.py predict.py
  outputs/              # best_feature.pt, last_feature.pt, history, kurva
prepare/
  split.py              # menyusun train/val/test dari data/raw
  build_metadata.py     # bangun ulang metadata.csv
compare_models.py       # grafik perbandingan ResNet18 vs ResNet50
outputs/                # compare_models.png, compare_models.md
```

## Menyusun dataset

Sumber kelas dibaca dari `data/raw/`, hasilnya ditulis ke `data/{train,val,test}/`.

```bash
python prepare/split.py --data data --keep-src
```

Split dilakukan per blok waktu berurutan, bukan acak. Frame berasal dari video
yang sama, jadi split acak membuat frame train dan val nyaris identik dan
`val_acc` jadi tidak bermakna. Flag penting: `--ratios 0.7 0.15 0.15`, `--block 5`,
`--keep-src` (folder sumber tidak dihapus), `--copy`, `--link` (symlink).

Menambah kelas baru? Pakai `--classes` supaya split kelas yang sudah ada tidak
diacak ulang:

```bash
python prepare/split.py --data data --classes trash --keep-src
python prepare/build_metadata.py --data data --force
```

`--force` di `build_metadata.py` wajib dipakai kalau `data/raw/` sudah berisi
file untuk kelas itu. Tanpa itu file yang sudah ada dilewati dan kelas barunya
tidak masuk ke `metadata.csv`.

`data/metadata.csv` mencatat setiap gambar: `class`, `image`, `split`,
`frame_index`, `seconds`. `frame_index` penting karena berasal dari video yang
sama — dipakai untuk memastikan split tidak memisahkan frame bersebelahan.

Kalau `data/raw/` atau `metadata.csv` hilang, bangun ulang dari file hasil split
(nama file masih menyimpan frame index aslinya):

```bash
python prepare/build_metadata.py --data data          # hardlink, tidak duplikat disk
python prepare/build_metadata.py --data data --force  # tulis ulang
```

## Grafik perbandingan ResNet18 vs ResNet50

`compare_models.py` di root membaca output training kedua folder lalu menggambar
perbandingannya. Script ini tidak training apa pun — jalankan training dulu,
lalu plotting:

```bash
python resnet18_3mode/compare.py --epochs 20 --patience 20   # 3 mode
python resnet50_1mode/train.py  --epochs 20 --patience 20     # 1 mode
python compare_models.py
```

Output: `outputs/compare_models.png` dan `outputs/compare_models.md`.

Isi grafik 4 panel: val accuracy per epoch, validation loss per epoch (log),
durasi per epoch, dan loss akhir + jumlah parameter. Mode `partial`/`scratch`
ResNet18 digambar sebagai garis konteks (`--no-context` untuk disembunyikan).

Yang dibandingkan adalah **feature vs feature** — mode sama di kedua sisi, jadi
selisihnya murni pengaruh arsitektur.

### Hasil (seed 42, 20 epoch, CPU)

| model | param | param dilatih | val_acc | test_acc | val_loss | test_loss | detik/epoch |
|---|---|---|---|---|---|---|---|
| ResNet18 feature | 11,2M | 1,5k | 1.0000 | 1.0000 | 0.1515 | 0.4842 | 16.1 |
| ResNet50 feature | 23,6M | 6,1k | 1.0000 | 1.0000 | 0.1836 | 0.5382 | 28.4 |

Konteks ResNet18: `partial` val_loss 0.0062, `scratch` 0.0001.

**Akurasi tidak dipakai jadi pembanding utama** karena semua konfigurasi mentok
di 1.0000. Pembeda yang nyata hanya loss dan kecepatan: ResNet18 hampir 2× lebih
cepat dengan parameterseparuh lebih sedikit, dan loss-nya justru lebih rendah.

Waktu per epoch di sini diukur di CPU. Di GPU angkanya berbeda dan ResNet50 akan
jauh lebih lambat karena jumlah layer dan channel-nya lebih banyak.

## Catatan eksperimen

- `feature`/`partial`/ResNet50 pakai bobot pretrained, jadi LR kecil wajar.
  `scratch` membuang bobot itu sehingga butuh LR jauh lebih besar (`1e-2`).
- Head `feature` cuma ~1.539–6.147 parameter. Di dataset 332 gambar itu sudah
  cukup untuk mencapai akurasi tinggi, tapi tidak generalisasi ke kelas atau
  kondisi baru — pakai `partial` kalau butuh model yang lebih kuat.
- Ketiga kelas tidak seimbang sumbernya. `box_cokelat` dan `box_merah` diambil
  dari video panjang dengan frame di-skip (100 gambar dari ribuan frame),
  sedangkan `trash` 132 gambar berurutan dari klip pendek. Artinya kelas `trash`
  punya variasi visual lebih sedikit dan lebih rawan overfitting, jadi
  `val_acc`/`test_acc` yang tinggi belum tentu berarti model generalize bagus.
- Split per blok waktu membuat `val` dan `test` berisi frame dari segmen video
  yang berbeda, jadi angkanya lebih jujur daripada split acak.
- `--amp` hanya speeding di CUDA; di CPU tidak berpengaruh.
- `history_*.json` berisi akurasi per epoch untuk melihat overfitting;
  `curves_*.png` / `compare_*.png` dibuat otomatis setiap training selesai.

## Tips

- Kalau dataset kecil (<2000 gambar), mulai dari `--epochs 10 --dropout 0.2`.
- Rasio split yang umum: train 70%, val 15%, test 15%.
- Setiap kelas wajib ada di train dan val, kalau tidak script berhenti dengan pesan error.
