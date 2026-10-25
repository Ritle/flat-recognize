"""Apply image-bound user choices for possible room partitions (V2 only)."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from .review_decisions import MAX_ANSWERS, image_sha256
except ImportError:  # Direct CLI execution from scripts/.
    from review_decisions import MAX_ANSWERS, image_sha256


PARTITION_CHOICES = {"separate", "open"}


def partition_fingerprint(candidate):
    encoded = json.dumps({key: candidate.get(key) for key in ("start", "end", "width")},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def partition_questions(rooms, source_sha256):
    meta = rooms.get("meta", {})
    size = [meta.get("width"), meta.get("height")]
    selected = {item["partition_id"]: item["choice"]
                for item in rooms.get("reviewed_partitions", [])}
    questions = []
    for index, candidate in enumerate(rooms.get("diagnostics", {}).get(
            "possible_room_partitions", [])):
        start, end = candidate["start"], candidate["end"]
        questions.append({"id": f"partition:{index}", "kind": "partition",
                          "partition_id": index, "fingerprint": partition_fingerprint(candidate),
                          "image_sha256": source_sha256, "image_size": size,
                          "current": selected.get(index, "open"),
                          "bbox": [min(start[0], end[0]), min(start[1], end[1]),
                                   max(start[0], end[0]), max(start[1], end[1])],
                          "center": [round((start[0] + end[0]) / 2, 2),
                                     round((start[1] + end[1]) / 2, 2)],
                          "reasons": ["facing_wall_caps_only"],
                          "rooms_if_closed": candidate.get("rooms_if_closed"),
                          "choices": ["separate", "open"], "severity": "review"})
    return questions[:MAX_ANSWERS]


def apply_partition_decisions(classified, previous_rooms, rgb, decisions, source_sha256):
    # Geometry imports stay local so the FastAPI demo can load question metadata
    # without requiring scripts/ to be on the Python import path.
    from extract_rooms import fill_polygon
    from extract_rooms_v2 import apply_closures, extract_rooms, room_candidates
    from filter_interior_windows_v2 import same_room_topology

    if not isinstance(decisions, dict) or decisions.get("version") != 1:
        raise ValueError("Unsupported review decisions format")
    if decisions.get("image_sha256") != source_sha256:
        raise ValueError("Review decisions belong to another image")
    answers = decisions.get("answers")
    if not isinstance(answers, list) or len(answers) > MAX_ANSWERS:
        raise ValueError("Invalid number of review answers")
    candidates = previous_rooms.get("diagnostics", {}).get("possible_room_partitions", [])
    selected = {}
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError("Invalid review answer")
        if "partition_id" not in answer:
            continue
        index, choice = answer.get("partition_id"), answer.get("choice")
        if (type(index) is not int or index < 0 or index >= len(candidates)
                or index in selected or choice not in PARTITION_CHOICES
                or answer.get("fingerprint") != partition_fingerprint(candidates[index])):
            raise ValueError("Partition answer refers to a changed or missing boundary")
        selected[index] = choice
    result = copy.deepcopy(previous_rooms)
    result["reviewed_partitions"] = [
        {"partition_id": index, "choice": choice} for index, choice in selected.items()]
    if not any(choice == "separate" for choice in selected.values()):
        return result, None, None
    baseline, barrier, inferred = extract_rooms(classified, rgb)
    if not same_room_topology(previous_rooms.get("rooms", []), baseline.get("rooms", [])):
        raise ValueError("Room topology changed before partition review")
    walls = np.zeros(barrier.shape, dtype=np.uint8)
    for polygon in classified.get("walls", []):
        fill_polygon(walls, polygon)
    current_rooms = baseline["rooms"]
    current_diagnostics = baseline["diagnostics"]
    accepted = []
    for index, choice in selected.items():
        if choice != "separate":
            continue
        candidate = candidates[index]
        trial, additions = apply_closures(barrier, [candidate])
        trial_inferred = inferred | additions
        trial_rooms, trial_diagnostics = room_candidates(trial, walls, trial_inferred)
        if (len(trial_rooms) <= len(current_rooms)
                or trial_diagnostics["interior_area_px"]
                < current_diagnostics["interior_area_px"] * .98
                or trial_diagnostics["outside_fraction_in_structure"]
                > current_diagnostics["outside_fraction_in_structure"] + .02):
            raise ValueError("Selected boundary no longer separates a room safely")
        barrier, inferred = trial, trial_inferred
        current_rooms, current_diagnostics = trial_rooms, trial_diagnostics
        accepted.append({"partition_id": index, **candidate})
    result["rooms"] = current_rooms
    diagnostics = result.setdefault("diagnostics", {})
    diagnostics["reviewed_partition_closures"] = accepted
    diagnostics["rooms_found"] = len(current_rooms)
    diagnostics["interior_area_px"] = current_diagnostics["interior_area_px"]
    diagnostics["outside_fraction_in_structure"] = current_diagnostics[
        "outside_fraction_in_structure"]
    diagnostics.setdefault("warnings", []).append(
        "Границы помещений разделены по выбору пользователя; опора стены или двери не подтверждена.")
    return result, barrier, inferred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("classified", type=Path)
    parser.add_argument("rooms", type=Path)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    with Image.open(args.image) as image:
        rgb = np.asarray(image.convert("RGB"))
    try:
        result, barrier, inferred = apply_partition_decisions(
            json.loads(args.classified.read_text(encoding="utf-8")),
            json.loads(args.rooms.read_text(encoding="utf-8")), rgb,
            json.loads(args.decisions.read_text(encoding="utf-8")), image_sha256(args.image))
    except ValueError as exc:
        print(f"REVIEW_DECISION_ERROR: {exc}", file=sys.stderr)
        return 3
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    (args.output_dir / f"{stem}_rooms_partition_reviewed.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if barrier is not None:
        Image.fromarray(barrier).save(args.output_dir / f"{stem}_partition_reviewed_barrier.png")
        Image.fromarray(inferred).save(args.output_dir / f"{stem}_partition_reviewed_closures.png")
        with Image.open(args.image) as image:
            overlay = image.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for room in result["rooms"]:
            ring = [tuple(point) for point in room["polygon"]]
            draw.line(ring + ring[:1], fill="magenta", width=3)
        for candidate in result["diagnostics"]["reviewed_partition_closures"]:
            draw.line([tuple(candidate["start"]), tuple(candidate["end"])],
                      fill="cyan", width=3)
        overlay.save(args.output_dir / f"{stem}_partition_reviewed_overlay.png")
    print(f"Applied partition choices: {len(result['reviewed_partitions'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
