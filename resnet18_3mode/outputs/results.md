# Hasil perbandingan 3 mode — resnet18

Data: 232 train / 50 val / 50 test, kelas `box_cokelat, box_merah, trash`, backbone `resnet18`, seed 42, epochs maks 20, patience 20, batch 32, dropout 0.2.

| mode | learning rate | param dilatih | % | best val_acc | test_acc | test_loss | epoch ke val=1.0 | epoch |
|---|---|---|---|---|---|---|---|---|
| `feature` | 1e-03 | 1,539 | 0% | 1.0000 | 1.0000 | 0.4842 | 2 | 20 |
| `partial` | 1e-04 | 8,395,267 | 75% | 1.0000 | 1.0000 | 0.1113 | 1 | 20 |
| `scratch` | 1e-02 | 11,178,051 | 100% | 1.0000 | 1.0000 | 0.0001 | 9 | 20 |

## Arti kolom

- **param dilatih** — jumlah bobot yang di-update, bukan total bobot model.
- **epoch ke val=1.0** — epoch pertama val_acc mencapai 1.0, proxy kecepatan konvergensi.
- **test_loss** — nilai loss, bukan cuma akurasi. Acc 1.0 dengan loss besar
  berarti model yakin tapi belum presisi.
