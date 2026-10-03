# Perbandingan ResNet18 vs ResNet50

Kedua model dilatih dengan data, seed, dan mode yang sama persis
(mode `feature`), jadi selisihnya mencerminkan arsitektur saja.

**Akurasi tidak dipakai sebagai pembanding utama**: semua konfigurasi
mentok di `val_acc`/`test_acc` 1.0000, jadi angkanya tidak membedakan.
Yang masih membedakan adalah `loss` dan waktu training.

| model | mode | param total | param dilatih | best val_acc | test_acc | val_loss (akhir) | test_loss | epoch | epoch ke val=1.0 | detik/epoch | total menit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ResNet18** | `feature` | 11,178,051 | 1,539 | 1.0000 | 1.0000 | 0.1515 | 0.4842 | 20 | 2 | 16.1 | 5.4 |
| **ResNet50** | `feature` | 23,567,352 | 6,147 | 1.0000 | 1.0000 | 0.1836 | 0.5382 | 20 | 2 | 28.4 | 9.5 |

## Konteks: ResNet18 mode `partial`

- best val_acc 1.0000, test_acc 1.0000
- val_loss akhir 0.0062, test_loss 0.1113
- 20 epoch, 16.3 detik/epoch

## Konteks: ResNet18 mode `scratch`

- best val_acc 1.0000, test_acc 1.0000
- val_loss akhir 0.0001, test_loss 0.0001
- 20 epoch, 15.0 detik/epoch
