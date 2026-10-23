"""Restore recognition uncertainty diagnostics after user opening choices."""

import argparse
import json
from pathlib import Path

import numpy as np
from filter_interior_windows_v2 import reviewed_window_report
from PIL import Image


def recheck_reviewed_rooms(classified, previous_rooms, reviewed_rooms, rgb):
    report = reviewed_window_report(classified, reviewed_rooms, rgb)
    diagnostics = reviewed_rooms.setdefault("diagnostics", {})
    diagnostics["interior_window_filter"] = report
    previous_diagnostics = previous_rooms.get("diagnostics", {})
    if "opening_recovery" in previous_diagnostics:
        diagnostics["opening_recovery"] = previous_diagnostics["opening_recovery"]
    if "topology_window_reconciliation" in previous_diagnostics:
        # An opening choice does not confirm the complete room topology.
        diagnostics["topology_window_reconciliation"] = previous_diagnostics[
            "topology_window_reconciliation"]
    previous_filter = previous_diagnostics.get("interior_window_filter", {})
    if (isinstance(previous_filter, dict)
            and "endcap_room_geometry_changes" in previous_filter):
        report["endcap_room_geometry_changes"] = previous_filter[
            "endcap_room_geometry_changes"]
    if "reclassified_wall_support" in previous_rooms:
        reviewed_rooms["reclassified_wall_support"] = previous_rooms[
            "reclassified_wall_support"]
    return reviewed_rooms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("classified", type=Path)
    parser.add_argument("previous_rooms", type=Path)
    parser.add_argument("reviewed_rooms", type=Path)
    args = parser.parse_args()
    with Image.open(args.image) as image:
        rgb = np.asarray(image.convert("RGB"))
    classified = json.loads(args.classified.read_text(encoding="utf-8"))
    previous = json.loads(args.previous_rooms.read_text(encoding="utf-8"))
    reviewed = json.loads(args.reviewed_rooms.read_text(encoding="utf-8"))
    recheck_reviewed_rooms(classified, previous, reviewed, rgb)
    args.reviewed_rooms.write_text(json.dumps(reviewed, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print("Rechecked remaining window uncertainty after user choices")


if __name__ == "__main__":
    main()
