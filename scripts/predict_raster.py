import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from shapely.geometry import Polygon
from shapely.ops import unary_union

from buildingcv.data import IMAGENET_MEAN, IMAGENET_STD
from buildingcv.extract_polygons import PolygonExtractor, mask_to_polygons
from buildingcv.labels import FLOOR_ID


def prepare_image(path, extractor):
    image = Image.open(path).convert("RGB")
    orig_w, orig_h = image.size

    model_h, model_w = extractor.image_size
    scale = min(model_w / orig_w, model_h / orig_h)

    inner_w = max(1, round(orig_w * scale))
    inner_h = max(1, round(orig_h * scale))

    resized = image.resize(
        (inner_w, inner_h),
        Image.Resampling.LANCZOS
    )

    arr = np.asarray(resized).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)

    if extractor.normalize:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std

    top = (model_h - inner_h) // 2
    left = (model_w - inner_w) // 2

    if extractor.normalize:
        canvas = torch.zeros(3, model_h, model_w)
    else:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        canvas = mean.expand(3, model_h, model_w).clone()

    canvas[:, top:top + inner_h, left:left + inner_w] = tensor

    return (
        canvas,
        (orig_w, orig_h),
        (left, top, inner_w, inner_h),
        scale,
    )


def convert_point(point, left, top, scale, orig_w, orig_h):
    x = (point[0] - left) / scale
    y = (point[1] - top) / scale

    x = max(0, min(orig_w, x))
    y = max(0, min(orig_h, y))

    return [round(x, 2), round(y, 2)]


def make_shape(poly):
    points = poly.get("outer", [])

    if len(points) < 3:
        return None

    try:
        holes = [
            hole for hole in poly.get("holes", [])
            if len(hole) >= 3
        ]

        shape = Polygon(points, holes)

        if not shape.is_valid:
            shape = shape.buffer(0)

        if shape.is_empty:
            return None

        return shape

    except Exception:
        return None


def rotated_dimensions(shape):
    """
    Возвращает длину и толщину минимального
    повёрнутого прямоугольника вокруг полигона.
    """

    rect = shape.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)

    lengths = []

    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]

        length = math.hypot(x2 - x1, y2 - y1)

        if length > 0:
            lengths.append(length)

    if not lengths:
        return 0.0, 0.0

    return max(lengths), min(lengths)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/predict_raster.py image.png"
        )
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    extractor = PolygonExtractor(
        run_dir=Path("weights"),
        ckpt="best.safetensors",
        device="cpu",
    )

    tensor, image_size, rect, scale = prepare_image(
        image_path,
        extractor
    )

    orig_w, orig_h = image_size
    left, top, inner_w, inner_h = rect

    with torch.no_grad():
        logits = extractor.model(
            tensor.unsqueeze(0).to(extractor.device)
        )

        mask = (
            logits
            .argmax(dim=1)
            .squeeze(0)
            .cpu()
            .to(torch.uint8)
            .numpy()
        )

    cleaned = np.full_like(mask, FLOOR_ID)

    cleaned[
        top:top + inner_h,
        left:left + inner_w
    ] = mask[
        top:top + inner_h,
        left:left + inner_w
    ]

    polygons = mask_to_polygons(cleaned)

    # --------------------------------------------------
    # WALLS
    # --------------------------------------------------

    walls = []

    for polygon in polygons.get("wall", []):
        wall = {
            "outer": [
                convert_point(
                    p,
                    left,
                    top,
                    scale,
                    orig_w,
                    orig_h,
                )
                for p in polygon["outer"]
            ],
            "holes": [
                [
                    convert_point(
                        p,
                        left,
                        top,
                        scale,
                        orig_w,
                        orig_h,
                    )
                    for p in hole
                ]
                for hole in polygon["holes"]
            ],
        }

        walls.append(wall)

    wall_shapes = []

    for wall in walls:
        shape = make_shape(wall)

        if shape is not None:
            wall_shapes.append(shape)

    wall_union = (
        unary_union(wall_shapes)
        if wall_shapes
        else None
    )

    # --------------------------------------------------
    # OPENINGS
    # --------------------------------------------------

    candidates = []

    for model_class in ("door", "window"):
        for polygon in polygons.get(model_class, []):

            opening = {
                "outer": [
                    convert_point(
                        p,
                        left,
                        top,
                        scale,
                        orig_w,
                        orig_h,
                    )
                    for p in polygon["outer"]
                ],
                "holes": [],
                "source_class": model_class,
            }

            candidates.append(opening)

    valid_openings = []
    rejected_openings = []

    min_dimension = min(orig_w, orig_h)

    # Допуск расстояния до стены.
    wall_tolerance = max(
        6.0,
        min_dimension * 0.015
    )

    # Минимальная длина реального проёма.
    min_opening_length = max(
        12.0,
        min_dimension * 0.02
    )

    # Максимальная допустимая толщина проёма.
    max_opening_thickness = max(
        20.0,
        min_dimension * 0.06
    )

    for index, opening in enumerate(candidates):

        shape = make_shape(opening)

        if shape is None:
            opening["reject_reason"] = "invalid_polygon"
            rejected_openings.append(opening)
            continue

        center = shape.centroid

        minx, miny, maxx, maxy = shape.bounds

        length, thickness = rotated_dimensions(shape)

        aspect_ratio = (
            length / thickness
            if thickness > 0
            else 999.0
        )

        wall_distance = (
            shape.distance(wall_union)
            if wall_union is not None
            else 999999
        )

        orientation = (
            "horizontal"
            if (maxx - minx) >= (maxy - miny)
            else "vertical"
        )

        opening.update({
            "id": index,
            "center": [
                round(center.x, 2),
                round(center.y, 2),
            ],
            "bbox": [
                round(minx, 2),
                round(miny, 2),
                round(maxx, 2),
                round(maxy, 2),
            ],
            "length": round(length, 2),
            "thickness": round(thickness, 2),
            "aspect_ratio": round(aspect_ratio, 2),
            "orientation": orientation,
            "wall_distance": round(wall_distance, 2),
        })

        reasons = []

        if wall_distance > wall_tolerance:
            reasons.append("too_far_from_wall")

        if length < min_opening_length:
            reasons.append("too_short")

        if thickness > max_opening_thickness:
            reasons.append("too_thick")

        if aspect_ratio < 1.8:
            reasons.append("not_elongated")

        if reasons:
            opening["reject_reason"] = reasons
            rejected_openings.append(opening)
        else:
            valid_openings.append(opening)

    # --------------------------------------------------
    # RESULT JSON
    # --------------------------------------------------

    result = {
        "meta": {
            "source": image_path.name,
            "width": orig_w,
            "height": orig_h,
            "model_size": [
                extractor.image_size[1],
                extractor.image_size[0],
            ],
        },

        "walls": walls,

        "openings": valid_openings,

        "diagnostics": {
            "raw_openings": len(candidates),
            "accepted_openings": len(valid_openings),
            "rejected_openings": len(rejected_openings),

            "wall_tolerance": round(
                wall_tolerance,
                2
            ),

            "rejected": rejected_openings,
        },
    }

    # --------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    json_path = (
        output_dir /
        f"{image_path.stem}.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------
    # OVERLAY
    # --------------------------------------------------

    original = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(original)

    # Стены — красным
    for wall in walls:

        points = [
            tuple(p)
            for p in wall["outer"]
        ]

        if len(points) >= 2:
            draw.line(
                points + [points[0]],
                fill="red",
                width=3,
            )

    # Валидные openings — зелёным
    for opening in valid_openings:

        points = [
            tuple(p)
            for p in opening["outer"]
        ]

        if len(points) >= 2:
            draw.line(
                points + [points[0]],
                fill="green",
                width=4,
            )

        cx, cy = opening["center"]

        draw.ellipse(
            (
                cx - 3,
                cy - 3,
                cx + 3,
                cy + 3,
            ),
            fill="green",
        )

    # Отброшенные openings — оранжевым
    for opening in rejected_openings:

        points = [
            tuple(p)
            for p in opening["outer"]
        ]

        if len(points) >= 2:
            draw.line(
                points + [points[0]],
                fill="orange",
                width=3,
            )

    overlay_path = (
        output_dir /
        f"{image_path.stem}_overlay.png"
    )

    original.save(overlay_path)

    # --------------------------------------------------
    # CONSOLE
    # --------------------------------------------------

    print()
    print("Done")
    print(f"JSON:      {json_path}")
    print(f"Overlay:   {overlay_path}")
    print()
    print(f"Walls:             {len(walls)}")
    print(f"Raw openings:      {len(candidates)}")
    print(f"Valid openings:    {len(valid_openings)}")
    print(f"Rejected openings: {len(rejected_openings)}")
    print()


if __name__ == "__main__":
    main()
