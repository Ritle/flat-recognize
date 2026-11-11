# data/

Datasets live here. Keep private images and generated annotations out of Git;
the new source images currently remain in the ignored `exmpls/` directory.

## Local raster training plan

The seven images added on 2026-11-10 are inventoried in
[`configs/training_samples.json`](../configs/training_samples.json).
See [`docs/TRAINING_PLAN.md`](../docs/TRAINING_PLAN.md) for annotation priorities,
the planned train/validation/test split and evaluation criteria. They are not
training-ready as a whole: five partially reviewed masks passed the local/base
family screen, while two images remain deferred drafts. The trainer supports mixing
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
Family screening of the seven new plans against all 5,000 originals is recorded
in `output/training_annotations/cubicasa-family-audit-20261110/`. No exact matches
or visually matching nearest candidates were found. This is a hash/nearest-image
screen, not exhaustive verification of every transformed or cropped derivative.

Five identical-image pairs cross the official train/evaluation lists. Derived
lists in `output/training_annotations/cubicasa-deduplicated-20261110/` exclude
the five training-side copies (4,195 train / 400 val / 400 test). Original files
and evaluation lists are preserved. Use `data.split_dir` to select these lists.

Initial mixed data preflight config (before the full SVG audit):
`output/training_annotations/mixed-ready-20261110/mixed-cpu.yaml`.
It combines 4,195 base training samples with two new partial masks, and 400
validation samples with two new partial masks. The extra test plan stays in test.

## Windows Cairo runtime

SVG loading additionally needs native Cairo. The project-local runtime lives in
`.tools/gtk-cairo-2026.10.0/bin/`, from the
[gvsbuild release](https://github.com/wingtk/gvsbuild/releases/tag/2026.10.0),
a Windows installation method listed by the [GTK project](https://www.gtk.org/docs/installations/windows/).
The downloaded ZIP SHA-256 was verified against its release metadata:
`8005b860447e73b10c7f508b1f348e2b6390e01dd61fd5c617b8190a046022fe`.
Version checked locally: Cairo 1.18.6, CairoSVG 2.9.1. DLLs, license files and font
configuration are local; no global installation or PATH change was made.

From the project root, set variables for the current PowerShell process:

```powershell
$env:CAIROCFFI_DLL_DIRECTORIES = (Resolve-Path .tools/gtk-cairo-2026.10.0/bin).Path
$env:PATH = $env:CAIROCFFI_DLL_DIRECTORIES + ';' + $env:PATH
& .\.venv\Scripts\python.exe -m buildingcv.train `
  --config output/training_annotations/mixed-ready-20261110/mixed-cpu.yaml `
  --check-data --check-data-per-source --limit-train 2 --limit-val 2
```

This checks two SVG and two raster examples in each of train/val, using strict
SVG reads without fallback substitution. The log is saved locally as
`output/training_annotations/mixed-ready-20261110/check-data.log`.
It does not create a model, train weights, or evaluate test images.

## Full strict SVG audit and active training lists

All 5,000 SVGs have now been checked at 512×512 without substitution:
4,989 passed and 11 failed (eight marker-rendering errors, three empty wall
masks with hidden wall geometry). Ten failures are in train and one is in test;
all 400 validation SVGs passed. This checks loader compatibility, not annotation
accuracy. Sources were not rewritten.

Active derived lists: `output/training_annotations/cubicasa-svg-checked-20261110/`.
They retain 4,185 base train / 400 val / 400 test entries after removing five
training duplicates and ten training failures. Evaluation lists are byte-identical
to their sources; `high_quality/1692` remains an explicitly recorded test rendering
failure, which must be resolved before claiming a complete test evaluation.

Active mixed config:
`output/training_annotations/finetune-ready-20261110/finetune-cpu.yaml`.
It selects these lists with `data.strict_svg: true` and totals 4,187 train / 402 val.
The eight-example mixed data preflight and real starting-weight compatibility
check passed on CPU; real-model training has not started.
See [training preflight instructions](../docs/TRAINING_PREFLIGHT.md) for repeatable
audit, curation and initialization commands.

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
