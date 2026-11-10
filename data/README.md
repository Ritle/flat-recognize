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

Download from https://zenodo.org/records/2613548 — direct link is `cubicasa5k.zip` (~3 GB).

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
