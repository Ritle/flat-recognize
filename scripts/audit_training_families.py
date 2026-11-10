"""Suggest related raster plans for manual split review; never approve training data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signatures(image):
    """Hashes of four orientations after removing uniform outer whitespace.

    These descriptors propose pairs only. Furniture, annotations and different
    crops can defeat them; small distances do not prove a shared floor plan.
    """
    gray = ImageOps.grayscale(image)
    ink = np.asarray(gray) < 220
    yy, xx = np.where(ink)
    if len(xx):
        gray = gray.crop((int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1))
    n = np.arange(32)
    basis = np.cos(np.pi * np.arange(8)[:, None] * (2 * n[None, :] + 1) / 64)
    basis[0] *= 1 / np.sqrt(2)
    result = []
    for angle in (0, 90, 180, 270):
        rotated = gray.rotate(angle, expand=True)
        small = np.array(rotated.resize((9, 8), Image.Resampling.LANCZOS), dtype=float)
        d_bits = (small[:, 1:] > small[:, :-1]).flatten()
        pixels = np.array(rotated.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
        freq = (basis @ pixels @ basis.T).flatten()[1:]
        p_bits = freq > np.median(freq)
        result.append((int("".join(map(str, d_bits.astype(int))), 2),
                       int("".join(map(str, p_bits.astype(int))), 2)))
    return result


def distance(a, b):
    # Compare equal orientations; the second image may be rotated.
    return min((((a[0][0] ^ dh).bit_count(), (a[0][1] ^ ph).bit_count(), angle * 90)
                for angle, (dh, ph) in enumerate(b)), key=lambda row: (row[1], row[0]))


def audit(training_manifest, regression_manifest, root, output):
    training = load_json(training_manifest)["samples"]
    regression = load_json(regression_manifest)["cases"]
    rows = {}
    for sample in training:
        path = (Path(root) / sample["image"]).resolve()
        if path in rows:
            raise ValueError("Duplicate training image path")
        rows[path] = {"id": sample["id"], "image": sample["image"], "split": sample["split"],
                      "group_id": sample["group_id"], "recorded_sha256": sample["sha256"],
                      "development_cases": []}
    for case in regression:
        path = (Path(root) / case["image"]).resolve()
        if path not in rows:
            rows[path] = {"id": case["id"], "image": case["image"], "split": "development",
                          "group_id": None, "development_cases": []}
        rows[path]["development_cases"].append(case["id"])
    descriptors, thumbnails, items = [], [], []
    for path, row in rows.items():
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if row.get("recorded_sha256", row["sha256"]) != row["sha256"]:
            raise ValueError(f"Recorded training source changed: {row['image']}")
        with Image.open(path) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
        row["width"], row["height"] = rgb.size
        descriptors.append(signatures(rgb))
        thumb = rgb.copy()
        thumb.thumbnail((240, 190))
        thumbnails.append(thumb)
        items.append(row)
    pairs, conflicts = [], []
    for i, first in enumerate(items):
        if first["split"] in ("val", "test") and first["development_cases"]:
            conflicts.append({"kind": "evaluation_image_used_in_development", "index": i})
        for j in range(i + 1, len(items)):
            second = items[j]
            exact = first["sha256"] == second["sha256"]
            same_group = first["group_id"] is not None and first["group_id"] == second["group_id"]
            dh, ph, angle = distance(descriptors[i], descriptors[j])
            if exact or same_group or (dh <= 12 and ph <= 12):
                pairs.append({"indices": [i, j], "ids": [first["id"], second["id"]],
                              "splits": [first["split"], second["split"]], "exact": exact,
                              "same_group": same_group, "dhash_distance": dh,
                              "phash_distance": ph, "second_rotation_deg": angle,
                              "requires_manual_review": not exact})
            if ((exact or same_group) and first["split"] != second["split"] and
                    ({first["split"], second["split"]} <= {"train", "val", "test"} or
                     first["split"] in ("val", "test") or second["split"] in ("val", "test"))):
                conflicts.append({"kind": "related_images_cross_evaluation_splits", "indices": [i, j]})
    pairs.sort(key=lambda pair: (not pair["exact"], pair["phash_distance"], pair["dhash_distance"]))
    report = {"version": 1, "method": "sha256_and_rotation_aware_dhash_phash",
              "scope": "local_training_inventory_and_development_regression_images",
              "candidate_limits": {"dhash_distance": 12, "phash_distance": 12},
              "manifest_sha256": {
                  "training": hashlib.sha256(Path(training_manifest).read_bytes()).hexdigest(),
                  "regression": hashlib.sha256(Path(regression_manifest).read_bytes()).hexdigest()},
              "base_training_dataset_checked": False, "automatic_group_approval": False,
              "images": items, "candidate_pairs": pairs, "conflicts": conflicts}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    columns, cell_w, cell_h = 4, 260, 225
    sheet = Image.new("RGB", (columns * cell_w, ((len(items) + columns - 1) // columns) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (row, thumb) in enumerate(zip(items, thumbnails)):
        x, y = (i % columns) * cell_w, (i // columns) * cell_h
        draw.text((x + 4, y + 3), f"{i}: {row['id']}"[:42], fill="black")
        draw.text((x + 4, y + 17), row["split"], fill="blue")
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 32))
    sheet.save(output / "contact-sheet.jpg")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, default=ROOT / "configs/training_samples.json")
    parser.add_argument("--regression-manifest", type=Path, default=ROOT / "test/recognition_cases.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.training_manifest, args.regression_manifest, args.root, args.output)
    print(f"{len(report['images'])} images; {len(report['candidate_pairs'])} candidate pairs; "
          f"{len(report['conflicts'])} confirmed split conflicts. No groups auto-approved.")
    return int(bool(report["conflicts"]))


if __name__ == "__main__":
    raise SystemExit(main())
