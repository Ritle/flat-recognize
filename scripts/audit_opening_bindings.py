"""Diagnose opening-to-segment fit before exporting door/window elements.

This report is advisory. The final project still uses add_openings.py and the
existing quality gate for authoritative binding and export decisions.
"""

import argparse
import json
import math
from pathlib import Path

from add_openings import (choose_segment_binding, door_width_cm, get_center,
                          get_opening_length_px, opening_orientation)


def audit_opening_bindings(classified, project, geometry):
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
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        report["reason"] = f"invalid_project_geometry: {exc}"
        return report
    report["status"] = "audited"
    for kind in ("doors", "windows"):
        for opening in classified.get(kind, []):
            row = {"opening_id": opening.get("id"), "kind": kind,
                   "pixel_center": None,
                   "bbox": opening.get("bbox"), "source_class": opening.get("source_class"),
                   "orientation": None, "width_cm": None,
                   "wall_uuid": None, "distance_to_wall_cm": None,
                   "overhang_cm": None, "aligned": None, "reasons": []}
            try:
                px, py = get_center(opening)
                orientation = opening_orientation(opening)
                length = get_opening_length_px(opening)
                width = length * (sx if orientation == "horizontal" else
                                  sy if orientation == "vertical" else (sx + sy) / 2)
                required = door_width_cm(width) if kind == "doors" else width
                if not all(math.isfinite(v) for v in (px, py, required)) or required <= 0:
                    raise ValueError("nonfinite or empty opening")
                row.update(pixel_center=[round(px, 2), round(py, 2)],
                           orientation=orientation, width_cm=round(required, 2))
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
            report["openings"].append(row)
    report["summary"] = {"examined": len(report["openings"]),
                         "suspect": sum(bool(row["reasons"]) for row in report["openings"]),
                         "by_reason": {reason: sum(reason in row["reasons"] for row in report["openings"])
                                       for reason in ("invalid_opening_geometry", "no_wall_segments",
                                                      "misaligned_wall", "far_from_wall",
                                                      "exceeds_segment")}}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("classified", type=Path)
    parser.add_argument("project", type=Path)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = audit_opening_bindings(
        json.loads(args.classified.read_text(encoding="utf-8")),
        json.loads(args.project.read_text(encoding="utf-8")),
        json.loads(args.geometry.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Opening binding preflight: {data['status']}; "
          f"{data['summary']['suspect']}/{data['summary']['examined']} suspect")


if __name__ == "__main__":
    main()
