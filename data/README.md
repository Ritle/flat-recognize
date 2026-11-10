# data/

Datasets live here. Keep private images and generated annotations out of Git;
the new source images currently remain in the ignored `exmpls/` directory.

## Local raster training plan

The seven images added on 2026-11-10 are inventoried in
[`configs/training_samples.json`](../configs/training_samples.json).
See [`docs/TRAINING_PLAN.md`](../docs/TRAINING_PLAN.md) for annotation priorities,
the planned train/validation/test split and evaluation criteria. They are not
training-ready: five partially reviewed masks are prepared locally, but base-dataset
family review is pending and two images remain deferred drafts. The trainer supports mixing
CubiCasa SVG samples with reviewed raster pairs using `data.raster_manifest`.
See [`docs/RASTER_ANNOTATIONS.md`](../docs/RASTER_ANNOTATIONS.md) for preparation,
configuration and the `--check-data` command.

## CubiCasa5K

Official dataset: [CubiCasa5K on Zenodo](https://zenodo.org/records/2613548),
linked by the [authors' repository](https://github.com/CubiCasa/CubiCasa5k).
The `cubicasa5k.zip` archive is **5,469,495,706 bytes** (about 5.47 GB).
Official MD5: `0ce0b203d1e3c125b51087b219bd23b9`.
Downloads and the extracted dataset stay under the ignored `data/cubicasa5k/`.

Downloaded and verified locally on 2026-11-10:

- Extracted root: `data/cubicasa5k/`.
- Archive: `data/cubicasa5k/_download/cubicasa5k.zip`.
- Receipt: `data/cubicasa5k/download-receipt.json`.
- 5,000 `model.svg` annotations; original PNG/SVG pairs exist for every split entry.
- Official split: 4,200 train / 400 validation / 400 test, with no overlapping paths.
- Official MD5 matches; ZIP member CRCs checked during extraction.
- Archive SHA-256: `a4a3398534d6abb7db7b8c6ca8877a6edf18897dcbd20f4582b3fdd3c46632a5`.

Temporary download parts were removed after verification; the complete ZIP is retained.
Family comparison against the new local plans is still pending. Downloading the
dataset does not enable training or approve the new annotations.

Expected layout after extraction:

```
data/cubicasa5k/
  colorful/
  high_quality/
  high_quality_architectural/
  train.txt
  val.txt
  test.txt
```

The `*.txt` files list relative paths (one per line) into the three quality buckets. Each sample directory contains `F1_original.png` (the floor plan) and `model.svg` (the annotation).
