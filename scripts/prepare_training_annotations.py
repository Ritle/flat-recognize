"""Prepare explicit, reviewed raster labels without using model predictions as truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from buildingcv.labels import IGNORE_ID
from buildingcv.raster_annotations import reviewed_ignore_mask, unresolved_mask

ROOT = Path(__file__).resolve().parent.parent
CLASSES = {"floor": 0, "wall": 1, "door": 2, "window": 3}
COLORS = np.array([[240, 240, 235], [40, 40, 45], [230, 120, 50],
                   [60, 150, 220]], dtype=np.uint8)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def verify_source(sample, root):
    path = Path(root) / sample["image"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"]:
        raise ValueError(f"Source image changed: {sample['image']}")
    with Image.open(path) as source:
        if source.size != (sample["width"], sample["height"]):
            raise ValueError(f"Source dimensions changed: {sample['image']}")
        if source.getexif().get(274, 1) != 1:
            raise ValueError("Normalize EXIF orientation before recording source coordinates")
        return source.convert("RGB")


def load_samples(manifest, root):
    payload = read_json(manifest)
    if (not isinstance(payload, dict) or payload.get("version") != 1 or
            not isinstance(payload.get("samples"), list)):
        raise ValueError("Expected a version 1 training sample manifest")
    samples = {}
    images = set()
    group_splits = {}
    hash_splits = {}
    for sample in payload["samples"]:
        if not isinstance(sample, dict):
            raise ValueError("Samples must be dictionaries")
        sid = sample.get("id")
        if (not isinstance(sid, str) or not sid or
                any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in sid)):
            raise ValueError("Sample IDs must use lowercase ASCII letters, digits, '-' or '_'")
        if sid in samples or sample.get("image") in images:
            raise ValueError(f"Duplicate sample ID or image: {sid}")
        split = sample.get("split")
        group = sample.get("group_id")
        if split not in ("train", "val", "test") or not isinstance(group, str) or not group:
            raise ValueError(f"Missing split or family group: {sid}")
        for seen, key in ((group_splits, group), (hash_splits, sample.get("sha256"))):
            if key in seen and seen[key] != split:
                raise ValueError(f"Related samples cross dataset splits: {sid}")
            seen[key] = split
        verify_source(sample, root)
        samples[sid] = sample
        images.add(sample["image"])
    if not samples:
        raise ValueError("Empty sample manifest")
    return samples


def draft_annotation(sample):
    return {
        "version": 1,
        "sample_id": sample["id"],
        "image": {key: sample[key] for key in ("image", "sha256", "width", "height")},
        "review": {"status": "draft", "coverage": "none", "reviewer": ""},
        "unresolved": [],
        "objects": [],
    }


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, c):
    return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))


def _intersects(a, b, c, d):
    ab_c, ab_d, cd_a, cd_b = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    if ab_c * ab_d < 0 and cd_a * cd_b < 0:
        return True
    return any(cross == 0 and _on_segment(p, q, r) for cross, p, q, r in (
        (ab_c, a, b, c), (ab_d, a, b, d), (cd_a, c, d, a), (cd_b, c, d, b)))


def checked_polygon(points, width, height):
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("Each object needs a polygon with at least three vertices")
    result = []
    for point in points:
        if (not isinstance(point, list) or len(point) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
            raise ValueError("Polygon coordinates must be finite numbers")
        x, y = point
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Polygon coordinates must be source-image pixel coordinates")
        result.append((x, y))
    if len(set(result)) != len(result):
        raise ValueError("Polygon vertices must be distinct; omit the repeated closing vertex")
    n = len(result)
    area2 = sum(result[i][0] * result[(i + 1) % n][1] -
                result[(i + 1) % n][0] * result[i][1] for i in range(n))
    if abs(area2) < 1:
        raise ValueError("Polygon has no measurable area")
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _intersects(result[i], result[(i + 1) % n], result[j], result[(j + 1) % n]):
                raise ValueError("Polygon self-intersects")
    return result


def _annotation_source(annotation, sample, root):
    if (not isinstance(annotation, dict) or annotation.get("version") != 1 or
            annotation.get("sample_id") != sample["id"]):
        raise ValueError("Annotation version or sample ID does not match")
    expected = {key: sample[key] for key in ("image", "sha256", "width", "height")}
    if annotation.get("image") != expected:
        raise ValueError("Annotation refers to a different image or coordinate system")
    return verify_source(sample, root)


def render_annotation(annotation, sample, root, *, allow_partial=False):
    source = _annotation_source(annotation, sample, root)
    ignored = reviewed_ignore_mask(annotation, source.size, allow_partial=allow_partial)
    mask, overlay = _render_objects(annotation, source)
    values = np.array(mask)
    values[ignored] = IGNORE_ID
    if not np.any(values == CLASSES["wall"]):
        raise ValueError("No reviewed wall pixels remain outside unresolved regions")
    if ignored.any():
        pixels = np.array(overlay)
        pixels[ignored] = (0.5 * pixels[ignored] + 0.5 * np.array([220, 90, 210])).astype(np.uint8)
        overlay = Image.fromarray(pixels)
    return Image.fromarray(values), overlay


def _render_objects(annotation, source):
    objects = annotation.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("An empty draft must not become an all-background training mask")
    checked, ids = [], set()
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError("Objects must be dictionaries")
        oid, label = obj.get("id"), obj.get("class")
        if not isinstance(oid, str) or not oid.strip() or oid in ids:
            raise ValueError("Objects need unique nonempty IDs")
        if not isinstance(label, str) or label not in CLASSES or label == "floor":
            raise ValueError("Only wall, door and window polygons are supported; floor is implicit")
        if set(obj) != {"id", "class", "polygon"}:
            raise ValueError("Object fields must be id/class/polygon; split shapes with holes into polygons")
        polygon = checked_polygon(obj["polygon"], source.width, source.height)
        checked.append((label, polygon))
        ids.add(oid)
    if not any(label == "wall" for label, _ in checked):
        raise ValueError("A floor-plan annotation must contain walls")
    mask = Image.new("L", source.size, 0)
    draw = ImageDraw.Draw(mask)
    # Class priority is fixed, regardless of the order in the annotation file.
    for label in ("wall", "door", "window"):
        for obj_label, polygon in checked:
            if obj_label == label:
                draw.polygon(polygon, fill=CLASSES[label])
    array = np.asarray(mask)
    for label, _ in checked:
        if not np.any(array == CLASSES[label]):
            raise ValueError(f"Class {label} disappeared during rasterization")
    colored = COLORS[array]
    blended = np.array(source).copy()
    selected = array != 0
    blended[selected] = (0.45 * blended[selected] + 0.55 * colored[selected]).astype(np.uint8)
    return mask, Image.fromarray(blended)


def preview_annotation(annotation, sample, root):
    """Render draft geometry and questions, without returning a training mask."""
    source = _annotation_source(annotation, sample, root)
    _, overlay = _render_objects(annotation, source)
    unresolved = annotation.get("unresolved")
    unresolved_mask(unresolved, source.size)
    draw = ImageDraw.Draw(overlay)
    for index, region in enumerate(unresolved, 1):
        bbox = region.get("bbox") if isinstance(region, dict) else None
        if (not isinstance(bbox, list) or len(bbox) != 4 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox)):
            raise ValueError("Unresolved regions need finite bbox coordinates")
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 < source.width and 0 <= y1 < y2 < source.height):
            raise ValueError("Unresolved region bbox must lie within the source image")
        draw.rectangle(bbox, outline=(220, 0, 0), width=2)
        draw.text((x1 + 2, y1 + 2), f"?{index}", fill=(220, 0, 0))
    draw.rectangle((0, 0, source.width - 1, 15), fill="white")
    draw.text((3, 1), "DRAFT PREVIEW - NOT FOR TRAINING", fill=(190, 0, 0))
    return overlay


def prepare_drafts(manifest, output, root=ROOT):
    samples = load_samples(manifest, root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for sample in samples.values():
        write_json(output / f"{sample['id']}.annotation.json", draft_annotation(sample))
    write_json(output / "index.json", {
        "version": 1, "classes": CLASSES,
        "samples": [{"sample_id": s["id"], "split": s["split"],
                     "annotation": f"{s['id']}.annotation.json"} for s in samples.values()],
    })
    return len(samples)


def build_masks(manifest, annotations, output, root=ROOT, *, allow_partial=False):
    samples = load_samples(manifest, root)
    rendered, seen = [], set()
    # Validate everything before creating the destination or publishing any mask.
    for path in annotations:
        annotation = read_json(path)
        if not isinstance(annotation, dict):
            raise ValueError("Annotation must be a dictionary")
        sid = annotation.get("sample_id")
        if not isinstance(sid, str) or sid not in samples or sid in seen:
            raise ValueError("Unknown or duplicate annotation sample ID")
        sample = samples[sid]
        mask, overlay = render_annotation(annotation, sample, root, allow_partial=allow_partial)
        rendered.append((sample, annotation, mask, overlay))
        seen.add(sid)
    if not rendered:
        raise ValueError("No reviewed annotations supplied")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for sample, annotation, mask, overlay in rendered:
        sid = sample["id"]
        mask.save(output / f"{sid}.mask.png")
        overlay.save(output / f"{sid}.overlay.png")
        write_json(output / f"{sid}.annotation.json", annotation)
        rows.append({"sample_id": sid, "split": sample["split"],
                     "mask": f"{sid}.mask.png", "group_id": sample["group_id"],
                     "group_review": sample.get("group_review", "pending"),
                     "training_ready": sample.get("group_review") == "reviewed",
                     "coverage": annotation["review"]["coverage"],
                     "ignore_index": IGNORE_ID if annotation["review"]["coverage"] == "outside_unresolved" else None,
                     "image": annotation["image"],
                     "mask_sha256": hashlib.sha256(
                         (output / f"{sid}.mask.png").read_bytes()).hexdigest(),
                     "annotation_sha256": hashlib.sha256(
                         (output / f"{sid}.annotation.json").read_bytes()).hexdigest()})
    write_json(output / "masks.json", {
        "version": 1, "classes": CLASSES, "samples": rows,
        "source_manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
    })
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/training_samples.json")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create draft JSONs, without creating any training masks")
    init.add_argument("--output", type=Path, required=True)
    build = sub.add_parser("build", help="Validate full reviewed annotations and render masks")
    build.add_argument("annotations", type=Path, nargs="+")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--allow-partial", action="store_true",
                       help="allow reviewed coverage outside explicitly excluded rectangles")
    preview = sub.add_parser("preview", help="Inspect draft geometry without producing a training mask")
    preview.add_argument("annotation", type=Path)
    preview.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            count = prepare_drafts(args.manifest, args.output)
            print(f"Prepared {count} drafts; no training masks were created: {args.output}")
        elif args.command == "build":
            count = build_masks(args.manifest, args.annotations, args.output,
                                allow_partial=args.allow_partial)
            print(f"Rendered {count} reviewed masks and overlays: {args.output}")
        else:
            samples = load_samples(args.manifest, ROOT)
            annotation = read_json(args.annotation)
            sid = annotation.get("sample_id") if isinstance(annotation, dict) else None
            if not isinstance(sid, str) or sid not in samples:
                raise ValueError("Unknown annotation sample ID")
            if args.output.exists():
                raise ValueError("Preview output already exists; choose a new filename")
            if args.output.suffix.lower() != ".png":
                raise ValueError("Preview output must be PNG")
            overlay = preview_annotation(annotation, samples[sid], ROOT)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(args.output)
            print(f"Draft preview only; no training mask created: {args.output}")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"ANNOTATION_ERROR: {exc}\n")


if __name__ == "__main__":
    main()
