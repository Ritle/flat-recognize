"""Review exterior proposals and test closures separately from project geometry."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

try:
    from .review_decisions import MAX_ANSWERS, image_sha256
except ImportError:
    from review_decisions import MAX_ANSWERS, image_sha256


EXTERIOR_CHOICES = {"confirm", "ignore"}


def exterior_fingerprint(candidate):
    payload = {key: candidate.get(key) for key in ("polygon", "rail_spans", "seed_ids")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf8")).hexdigest()[:20]


def exterior_questions(report, source_sha256, image_size):
    if len(image_size) != 2 or any(type(value) is not int or value <= 0 for value in image_size):
        return []
    questions = []
    for candidate in report.get("candidates", []):
        x0, y0, x1, y1 = candidate["bbox"]
        uid = candidate["id"]
        questions.append({"id": f"exterior:{uid}", "kind": "exterior", "exterior_id": uid,
            "fingerprint": exterior_fingerprint(candidate), "image_sha256": source_sha256,
            "image_size": image_size, "bbox": candidate["bbox"],
            "center": [(x0 + x1) / 2, (y0 + y1) / 2],
            "current": candidate.get("reviewed_choice", "pending"),
            "choices": ["confirm", "ignore"], "severity": "review",
            "trial": candidate.get("trial")})
    return questions[:MAX_ANSWERS]


def preserved_rooms(before, after):
    from shapely.geometry import Polygon

    remaining = [Polygon(room["polygon"]) for room in after]
    for room in before:
        source = Polygon(room["polygon"])
        scores = [source.intersection(shape).area / source.union(shape).area
                  if shape.is_valid and shape.area > 0 else 0 for shape in remaining]
        if not scores or max(scores) < .98:
            return False
        remaining.pop(scores.index(max(scores)))
    return True


def apply_exterior_decisions(classified, rooms, rgb, decisions, source_sha256):
    import cv2
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from detect_exterior_enclosures_v2 import detect_exterior_enclosures
    from extract_rooms import fill_polygon
    from extract_rooms_v2 import extract_rooms, room_candidates

    if (not isinstance(decisions, dict) or decisions.get("version") != 1
            or decisions.get("image_sha256") != source_sha256):
        raise ValueError("Exterior decisions do not match this image")
    answers = decisions.get("answers")
    if not isinstance(answers, list) or len(answers) > MAX_ANSWERS:
        raise ValueError("Invalid number of review answers")
    # Re-detect from current pixels and rooms; stored proposal JSON is not authority.
    report = detect_exterior_enclosures(classified, rooms, rgb)
    by_id = {candidate["id"]: candidate for candidate in report["candidates"]}
    selected = {}
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError("Invalid review answer")
        if "exterior_id" not in answer:
            continue
        uid, choice = answer.get("exterior_id"), answer.get("choice")
        if (type(uid) is not int or uid in selected or uid not in by_id
                or not isinstance(choice, str) or choice not in EXTERIOR_CHOICES
                or answer.get("fingerprint") != exterior_fingerprint(by_id[uid])):
            raise ValueError("Exterior answer refers to a changed or missing contour")
        selected[uid] = choice
    for uid, choice in selected.items():
        by_id[uid]["reviewed_choice"] = choice
    if "confirm" not in selected.values():
        return report, None

    baseline, barrier, inferred = extract_rooms(classified, rgb)
    walls = np.zeros(barrier.shape, np.uint8)
    for wall in classified.get("walls", []):
        fill_polygon(walls, wall)
    baseline_ok = preserved_rooms(rooms.get("rooms", []), baseline["rooms"])
    current_rooms = baseline["rooms"]
    active_count = len(rooms.get("rooms", []))
    successful = 0
    for uid, choice in selected.items():
        if choice != "confirm":
            continue
        candidate = by_id[uid]
        trial = {"status": "rejected", "rooms_before": active_count + successful,
                 "rooms_after": active_count + successful,
                 "reason": "baseline_rooms_changed"}
        candidate["trial"] = trial
        if not baseline_ok:
            continue
        additions = np.zeros_like(barrier)
        seal_width = max(3, round(max(candidate["anchor_distance_px"]) * 2 + 3))
        for start, end in candidate["rail_spans"]:
            cv2.line(additions, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)),
                     255, seal_width)
        trial_barrier = barrier | additions
        trial_inferred = inferred | additions
        trial_rooms, _ = room_candidates(trial_barrier, walls, trial_inferred)
        if len(trial_rooms) != len(current_rooms) + 1 or not preserved_rooms(current_rooms, trial_rooms):
            trial["reason"] = "closure_did_not_add_one_room_safely"
            continue
        old = unary_union([Polygon(room["polygon"]) for room in current_rooms])
        outline = Polygon(candidate["polygon"])
        added = [room for room in trial_rooms
                 if (shape := Polygon(room["polygon"])).intersection(old).area < shape.area * .02
                 and shape.intersection(outline).area > shape.area * .9
                 and shape.area > outline.area * .5]
        if len(added) != 1:
            trial["reason"] = "new_room_does_not_match_exterior_contour"
            continue
        successful += 1
        trial.update(status="accepted", reason="one_new_room_preserving_existing_rooms",
                     rooms_after=active_count + successful, preserved_rooms=True,
                     new_room=copy.deepcopy(added[0]), seal_width_px=seal_width)
        barrier, inferred, current_rooms = trial_barrier, trial_inferred, trial_rooms
    return report, barrier


def main():
    import numpy as np
    from PIL import Image
    from detect_exterior_enclosures_v2 import save_overlay

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("classified", type=Path)
    parser.add_argument("rooms", type=Path)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--barrier-output", type=Path, required=True)
    args = parser.parse_args()
    with Image.open(args.image) as image:
        rgb = np.asarray(image.convert("RGB"))
    try:
        report, barrier = apply_exterior_decisions(
            json.loads(args.classified.read_text(encoding="utf8")),
            json.loads(args.rooms.read_text(encoding="utf8")), rgb,
            json.loads(args.decisions.read_text(encoding="utf8")), image_sha256(args.image))
    except ValueError as exc:
        print(f"REVIEW_DECISION_ERROR: {exc}", file=sys.stderr)
        return 3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.overlay.parent.mkdir(parents=True, exist_ok=True)
    args.barrier_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    save_overlay(args.image, report, args.overlay)
    if barrier is not None:
        Image.fromarray(barrier).save(args.barrier_output)
    print("Exterior contour choices checked; project recognition unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
