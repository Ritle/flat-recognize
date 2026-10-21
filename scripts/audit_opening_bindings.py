"""Diagnose opening-to-segment fit before exporting door/window elements.

This report is advisory. The final project still uses add_openings.py and the
existing quality gate for authoritative binding and export decisions.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
from add_openings import (
    choose_segment_binding,
    door_width_cm,
    opening_export_geometry,
)
from PIL import Image, ImageDraw
from recover_windows_topology_v2 import measured_window_axis
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from wall_band_measurement import flanking_wall_band


def raster_span_on_current_walls(opening, segments, transform, walls=None):
    """Counterfactual binding; never changes the opening or project."""
    recovery = opening.get("recovery")
    evidence = recovery.get("raster_axis_evidence") if isinstance(recovery, dict) else None
    if not isinstance(evidence, dict):
        return None
    if evidence.get("extent_basis") != "raster_clear_bay":
        return None
    try:
        (x0, y0), (x1, y1) = evidence["span"]
        x0, y0, x1, y1 = (float(value) for value in (x0, y0, x1, y1))
        if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
            return None
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if max(dx, dy) <= 0 or min(dx, dy) > max(dx, dy) * .27:
            return None
        orientation = "horizontal" if dx >= dy else "vertical"
        px, py = (x0 + x1) / 2, (y0 + y1) / 2
        sx, sy, ox, oy = transform
        width = max(dx, dy) * (sx if orientation == "horizontal" else sy)
        binding = choose_segment_binding(segments, ox + px * sx, oy + py * sy,
                                         width, orientation)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    result = {"span": evidence["span"], "width_cm": round(width, 2),
              "wall_uuid": None, "distance_to_wall_cm": None,
              "overhang_cm": None, "aligned": None, "reasons": []}
    if walls is not None:
        band = flanking_wall_band(walls, ((x0, y0), (x1, y1)))
        if band is not None:
            result["source_wall_band"] = band
    if binding is None:
        result["reasons"].append("no_wall_segments")
        return result
    segment = binding["segment"]
    thickness = segment.get("thickness", 20)
    thickness = float(thickness) if isinstance(thickness, (int, float)) else 20.0
    if not math.isfinite(thickness) or thickness <= 0:
        thickness = 20.0
    result.update(wall_uuid=segment.get("uuid"),
                  distance_to_wall_cm=round(binding["distance"], 2),
                  overhang_cm=round(binding["overhang"], 2),
                  aligned=bool(binding["aligned"]),
                  selection=binding.get("selection"))
    if not binding["aligned"]:
        result["reasons"].append("misaligned_wall")
    if binding["distance"] > max(20, thickness * 1.5):
        result["reasons"].append("far_from_wall")
    if binding["overhang"] > 5:
        result["reasons"].append("exceeds_segment")
    if "source_wall_band" in result:
        normal_scale = sy if orientation == "horizontal" else sx
        measured_cm = result["source_wall_band"]["median_thickness_px"] * normal_scale
        result["source_wall_band"]["median_thickness_cm"] = round(measured_cm, 2)
        result["source_wall_band"]["bound_wall_thickness_cm"] = round(thickness, 2)
        if (result["source_wall_band"]["consistent_flanks"]
                and measured_cm > thickness * 2.5):
            result["reasons"].append("wall_thickness_mismatch")
    return result


def audit_segment_wall_bands(segments, support, transform):
    """Advisory comparison of exported wall axes with source structural bands."""
    sx, sy, ox, oy = transform
    result = {"examined": len(segments), "sampled": 0, "suspect": 0,
              "segments": []}
    if support.is_empty:
        return result
    for item in segments:
        segment = item["segment"]
        ax, ay = (item["ax"] - ox) / sx, (item["ay"] - oy) / sy
        bx, by = (item["bx"] - ox) / sx, (item["by"] - oy) / sy
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 30:
            continue
        nx, ny = -dy / length, dx / length
        expected = segment.get("thickness", 20)
        if not isinstance(expected, (int, float)) or not math.isfinite(expected) or expected <= 0:
            continue
        normal_scale = math.hypot(nx * sx, ny * sy)
        expected_px = expected / normal_scale
        reach = max(100, expected_px * 5)
        samples = []
        for fraction in (.2, .5, .8):
            x, y = ax + dx * fraction, ay + dy * fraction
            cut = support.intersection(LineString([
                (x - nx * reach, y - ny * reach),
                (x + nx * reach, y + ny * reach)]))
            pieces = ([cut] if cut.geom_type == "LineString" else
                      list(getattr(cut, "geoms", [])))
            nearby = [piece for piece in pieces
                      if piece.geom_type == "LineString" and 3 <= piece.length <= reach
                      and piece.distance(Point(x, y)) <= max(12, expected_px * .75)]
            if not nearby:
                break
            band = max(nearby, key=lambda piece: piece.length)
            middle = band.interpolate(.5, normalized=True)
            signed_offset = (middle.x - x) * nx + (middle.y - y) * ny
            samples.append((band.length, signed_offset))
        if len(samples) != 3:
            continue
        widths = [width for width, _ in samples]
        offsets = [offset for _, offset in samples]
        if (max(widths) > min(widths) * 1.5
                or max(offsets) - min(offsets) > max(8, expected_px * .75)):
            continue
        result["sampled"] += 1
        measured = statistics.median(widths) * normal_scale
        center_offset = statistics.median(offsets) * normal_scale
        reasons = []
        if measured > expected * 2.5 or measured < expected * .35:
            reasons.append("wall_thickness_mismatch")
        if (abs(center_offset) > max(8 * normal_scale, expected * .75)
                and all(offset * center_offset > 0 for offset in offsets)):
            reasons.append("wall_axis_off_center")
        if reasons:
            source_offset_px = statistics.median(offsets)
            result["segments"].append({
                "wall_uuid": segment.get("uuid"),
                "pixel_start": [round(ax, 2), round(ay, 2)],
                "pixel_end": [round(bx, 2), round(by, 2)],
                "source_axis_pixel_start": [round(ax + nx * source_offset_px, 2),
                                            round(ay + ny * source_offset_px, 2)],
                "source_axis_pixel_end": [round(bx + nx * source_offset_px, 2),
                                          round(by + ny * source_offset_px, 2)],
                "measured_thickness_cm": round(measured, 2),
                "project_thickness_cm": round(expected, 2),
                "axis_offset_cm": round(center_offset, 2),
                "reasons": reasons})
    result["suspect"] = len(result["segments"])
    return result


def audit_opening_bindings(classified, project, geometry, gray=None):
    report = {"version": 1, "method": "polygon_segment_preflight",
              "status": "skipped", "reason": None, "openings": [],
              "summary": {"examined": 0, "suspect": 0}}
    if geometry.get("adapter") != "polygon":
        report["reason"] = "exact_pixel_to_world_transform_unavailable"
        return report
    transform = geometry.get("pixel_to_world") or {}
    try:
        sx, sy = float(transform["scale_x"]), float(transform["scale_y"])
        ox, oy = float(transform["origin_x"]), float(transform["origin_y"])
        if not all(math.isfinite(v) for v in (sx, sy, ox, oy)) or min(sx, sy) <= 0:
            raise ValueError("invalid transform")
        level = project["levels"][0]
        connectors = {item["uuid"]: item for item in level["connectors"]}
        segments = []
        for segment in level["segments"]:
            a, b = connectors[segment["start"]], connectors[segment["end"]]
            segments.append({"segment": segment, "ax": float(a["x"]),
                             "ay": float(a["y"]), "bx": float(b["x"]),
                             "by": float(b["y"])})
        wall_shapes = []
        for item in classified.get("walls", []) + classified.get("reclassified_wall_support", []):
            shape = Polygon(item["outer"], item.get("holes", []))
            if not shape.is_valid:
                shape = shape.buffer(0)
            if not shape.is_empty:
                wall_shapes.append(shape)
        walls = unary_union(wall_shapes) if wall_shapes else None
        opening_shapes = []
        for item in classified.get("doors", []) + classified.get("windows", []):
            if len(item.get("outer", [])) < 3:
                continue
            shape = Polygon(item["outer"])
            if not shape.is_valid:
                shape = shape.buffer(0)
            if not shape.is_empty:
                opening_shapes.append(shape)
        support = unary_union(wall_shapes + opening_shapes) if wall_shapes or opening_shapes else Polygon()
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        report["reason"] = f"invalid_project_geometry: {exc}"
        return report
    report["status"] = "audited"
    report["wall_geometry"] = audit_segment_wall_bands(
        segments, support, (sx, sy, ox, oy))
    for kind in ("doors", "windows"):
        for opening in classified.get(kind, []):
            row = {"opening_id": opening.get("id"), "kind": kind,
                   "pixel_center": None,
                   "bbox": opening.get("bbox"), "source_class": opening.get("source_class"),
                   "orientation": None, "width_cm": None,
                   "wall_uuid": None, "distance_to_wall_cm": None,
                   "overhang_cm": None, "aligned": None, "reasons": []}
            try:
                export_geometry = opening_export_geometry(opening, door=kind == "doors")
                px, py = export_geometry["center"]
                orientation = export_geometry["orientation"]
                length = export_geometry["length_px"]
                width = length * (sx if orientation == "horizontal" else
                                  sy if orientation == "vertical" else (sx + sy) / 2)
                required = door_width_cm(width) if kind == "doors" else width
                if not all(math.isfinite(v) for v in (px, py, required)) or required <= 0:
                    raise ValueError("nonfinite or empty opening")
                row.update(pixel_center=[round(px, 2), round(py, 2)],
                           orientation=orientation, width_cm=round(required, 2),
                           geometry_source=export_geometry["source"])
                binding = choose_segment_binding(segments, ox + px * sx,
                                                 oy + py * sy, required, orientation)
            except (KeyError, TypeError, ValueError, OverflowError):
                row["reasons"].append("invalid_opening_geometry")
                report["openings"].append(row)
                continue
            if binding is None:
                row["reasons"].append("no_wall_segments")
            else:
                segment = binding["segment"]
                thickness = segment.get("thickness", 20)
                thickness = float(thickness) if isinstance(thickness, (int, float)) else 20.0
                if not math.isfinite(thickness) or thickness <= 0:
                    thickness = 20.0
                threshold = max(20, thickness * 1.5)
                row.update(wall_uuid=segment.get("uuid"),
                           distance_to_wall_cm=round(binding["distance"], 2),
                           overhang_cm=round(binding["overhang"], 2),
                           aligned=bool(binding["aligned"]),
                           selection=binding.get("selection"),
                           distance_threshold_cm=round(threshold, 2))
                if not binding["aligned"]:
                    row["reasons"].append("misaligned_wall")
                if binding["distance"] > threshold:
                    row["reasons"].append("far_from_wall")
                if binding["overhang"] > 5:
                    row["reasons"].append("exceeds_segment")
            if kind == "windows":
                if gray is not None and row["reasons"] and not opening.get("recovery"):
                    evidence = measured_window_axis(gray, opening)
                    if evidence is not None:
                        row["raster_axis_evidence"] = evidence
                        (_, ay), (_, by) = evidence["span"]
                        if orientation == "horizontal":
                            axis_center = (ay + by) / 2
                            bbox_center = (opening["bbox"][1] + opening["bbox"][3]) / 2
                        else:
                            (ax, _), (bx, _) = evidence["span"]
                            axis_center = (ax + bx) / 2
                            bbox_center = (opening["bbox"][0] + opening["bbox"][2]) / 2
                        row["bbox_axis_offset_px"] = round(axis_center - bbox_center, 2)
                        opening_for_span = {**opening, "recovery": {
                            "raster_axis_evidence": evidence}}
                    else:
                        opening_for_span = opening
                else:
                    opening_for_span = opening
                alternative = raster_span_on_current_walls(
                    opening_for_span, segments, (sx, sy, ox, oy), walls)
                if alternative is not None:
                    row["raster_span_on_current_walls"] = alternative
            report["openings"].append(row)
    report["summary"] = {"examined": len(report["openings"]),
                         "suspect": sum(bool(row["reasons"]) for row in report["openings"]),
                         "by_reason": {reason: sum(reason in row["reasons"] for row in report["openings"])
                                       for reason in ("invalid_opening_geometry", "no_wall_segments",
                                                      "misaligned_wall", "far_from_wall",
                                                      "exceeds_segment")}}
    return report


def save_wall_geometry_overlay(image_path, report, overlay_path):
    """Show only corroborated suspect axes; the lines are diagnostic proposals."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for row in report.get("wall_geometry", {}).get("segments", []):
        start, end = row["pixel_start"], row["pixel_end"]
        source_start, source_end = row["source_axis_pixel_start"], row["source_axis_pixel_end"]
        draw.line([tuple(start), tuple(end)], fill="red", width=4)
        draw.line([tuple(source_start), tuple(source_end)], fill="yellow", width=3)
        draw.ellipse((start[0] - 4, start[1] - 4, start[0] + 4, start[1] + 4),
                     fill="red")
    for row in report.get("openings", []):
        if row.get("raster_axis_evidence") and row.get("reasons"):
            span = row["raster_axis_evidence"]["span"]
            draw.line([tuple(span[0]), tuple(span[1])], fill="cyan", width=3)
    overlay_path = Path(overlay_path)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(overlay_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("classified", type=Path)
    parser.add_argument("project", type=Path)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()
    if bool(args.image) != bool(args.overlay):
        parser.error("--image and --overlay must be used together")
    gray = None
    if args.image:
        with Image.open(args.image) as source:
            gray = np.asarray(source.convert("L"))
    data = audit_opening_bindings(
        json.loads(args.classified.read_text(encoding="utf-8")),
        json.loads(args.project.read_text(encoding="utf-8")),
        json.loads(args.geometry.read_text(encoding="utf-8")), gray=gray)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.overlay and (data.get("wall_geometry", {}).get("suspect", 0) or
                         any(row.get("raster_axis_evidence") for row in data["openings"])):
        save_wall_geometry_overlay(args.image, data, args.overlay)
    print(f"Opening binding preflight: {data['status']}; "
          f"{data['summary']['suspect']}/{data['summary']['examined']} suspect")


if __name__ == "__main__":
    main()
