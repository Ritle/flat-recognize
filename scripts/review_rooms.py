"""Offer small ambiguous areas for review and apply image-bound room choices."""

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

try:
    from .review_decisions import MAX_ANSWERS, image_sha256
except ImportError:  # Direct CLI execution from scripts/.
    from review_decisions import MAX_ANSWERS, image_sha256


ROOM_CHOICES = {"room", "ignore"}
SMALL_AREA_FRACTION = .003


def room_fingerprint(room):
    payload = {"id": room.get("id"), "polygon": room.get("polygon")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def room_questions(data, source_sha256):
    meta = data.get("meta", {})
    width, height = meta.get("width"), meta.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        return []
    selected = {entry["room_id"]: entry["choice"]
                for entry in data.get("reviewed_rooms", [])}
    ignored = {room.get("id") for room in data.get("ignored_rooms", [])}
    questions = []
    for room in data.get("rooms", []) + data.get("ignored_rooms", []):
        uid = room.get("id")
        bbox = room.get("bbox")
        area = room.get("area_px")
        if (type(uid) is not int or not isinstance(bbox, list) or len(bbox) != 4
                or not isinstance(area, (int, float)) or not math.isfinite(area) or area <= 0):
            continue
        if area / (width * height) >= SMALL_AREA_FRACTION and uid not in ignored:
            continue
        questions.append({
            "id": f"room:{uid}", "kind": "room", "room_id": uid,
            "fingerprint": room_fingerprint(room), "image_sha256": source_sha256,
            "image_size": [width, height], "current": "ignore" if uid in ignored else "room",
            "bbox": bbox, "center": [round((bbox[0] + bbox[2]) / 2, 2),
                                      round((bbox[1] + bbox[3]) / 2, 2)],
            "reasons": ["small_enclosed_area"] + (["user_choice"] if uid in selected else []),
            "choices": ["room", "ignore"], "severity": "review",
        })
    return questions[:MAX_ANSWERS]


def apply_room_decisions(source, decisions, source_sha256):
    if not isinstance(decisions, dict) or decisions.get("version") != 1:
        raise ValueError("Unsupported review decisions format")
    if decisions.get("image_sha256") != source_sha256:
        raise ValueError("Review decisions belong to another image")
    answers = decisions.get("answers")
    if not isinstance(answers, list) or len(answers) > MAX_ANSWERS:
        raise ValueError("Invalid number of review answers")
    data = copy.deepcopy(source)
    rooms = data.get("rooms", [])
    by_id = {room.get("id"): room for room in rooms}
    if len(by_id) != len(rooms):
        raise ValueError("Room IDs must be unique")
    reviewable = {question["room_id"] for question in room_questions(source, source_sha256)}
    selected = {}
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError("Invalid review answer")
        if "room_id" not in answer:
            continue
        uid, choice = answer.get("room_id"), answer.get("choice")
        if type(uid) is not int or uid in selected or choice not in ROOM_CHOICES:
            raise ValueError("Invalid or repeated room choice")
        room = by_id.get(uid)
        if uid not in reviewable or room is None or room_fingerprint(room) != answer.get("fingerprint"):
            raise ValueError("Review answer refers to a changed or missing room")
        selected[uid] = choice
    if selected and all(choice == "ignore" for choice in selected.values()) and len(selected) == len(rooms):
        raise ValueError("At least one room must remain")
    if selected:
        data["ignored_rooms"] = [room for room in rooms if selected.get(room.get("id")) == "ignore"]
        data["rooms"] = [room for room in rooms if selected.get(room.get("id")) != "ignore"]
        data["reviewed_rooms"] = [{"room_id": uid, "choice": choice}
                                   for uid, choice in selected.items()]
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rooms", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()
    try:
        reviewed = apply_room_decisions(
            json.loads(args.rooms.read_text(encoding="utf-8")),
            json.loads(args.decisions.read_text(encoding="utf-8")),
            image_sha256(args.image))
    except ValueError as exc:
        print(f"REVIEW_DECISION_ERROR: {exc}", file=sys.stderr)
        return 3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reviewed, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.overlay:
        from PIL import Image, ImageDraw
        with Image.open(args.image) as source_image:
            image = source_image.convert("RGB")
        draw = ImageDraw.Draw(image)
        for kind, color in (("rooms", "blue"), ("ignored_rooms", "orange")):
            for room in reviewed.get(kind, []):
                points = [tuple(point) for point in room.get("polygon", [])]
                if len(points) >= 3:
                    draw.line(points + points[:1], fill=color, width=3)
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.overlay)
    print(f"Applied room choices: {len(reviewed.get('reviewed_rooms', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
