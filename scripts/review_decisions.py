"""Describe uncertain openings and apply explicit user choices before export."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path


CHOICES = {"door", "window", "ignore"}
MAX_ANSWERS = 50


def image_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opening_fingerprint(opening):
    geometry = {"id": opening.get("id"), "outer": opening.get("outer")}
    encoded = json.dumps(geometry, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def opening_location(opening):
    points = opening.get("outer") or []
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    if not xs or not ys:
        raise ValueError("Opening has no polygon")
    bbox = [round(min(xs), 2), round(min(ys), 2),
            round(max(xs), 2), round(max(ys), 2)]
    return bbox, [round((bbox[0] + bbox[2]) / 2, 2),
                  round((bbox[1] + bbox[3]) / 2, 2)]


def make_review_questions(classified, quality, source_sha256):
    """Only create actionable questions; a question is never a quality approval."""
    reasons = {}
    severity = {}
    doors, windows = classified.get("doors", []), classified.get("windows", [])
    opening_ids = {opening.get("id") for opening in doors + windows}
    for issue in (quality or {}).get("issues", []):
        if issue.get("code") not in ("opening_perpendicular_wall", "opening_far_from_wall",
                                     "opening_exceeds_segment"):
            continue
        uid = issue.get("opening_id")
        if not isinstance(uid, int):
            kind = issue.get("opening_kind")
            index = issue.get("opening_index")
            openings = doors if kind == "doors" else windows if kind == "windows" else []
            if not isinstance(index, int) or index < 1 or index > len(openings):
                continue
            uid = openings[index - 1].get("id")
        if uid not in opening_ids:
            continue
        reasons.setdefault(uid, set()).add(issue["code"])
        severity[uid] = max(severity.get(uid, 0), 2 if issue.get("severity") == "error" else 1)
    candidates = doors + windows + [opening for opening in classified.get("rejected_openings", [])
                                     if (opening.get("classification") or {}).get("method") == "user_ignored"]
    questions = []
    seen = set()
    for opening in candidates:
        uid = opening.get("id")
        if not isinstance(uid, int) or uid in seen:
            continue
        seen.add(uid)
        classification = opening.get("classification") or {}
        if classification.get("requires_review"):
            reasons.setdefault(uid, set()).add("classification_uncertain")
        if classification.get("method") in ("user_selected", "user_ignored"):
            reasons.setdefault(uid, set()).add("user_choice")
        if uid not in reasons:
            continue
        bbox, center = opening_location(opening)
        current = "ignore" if classification.get("method") == "user_ignored" else opening.get("type")
        questions.append({
            "id": f"opening:{uid}", "kind": "opening", "opening_id": uid,
            "fingerprint": opening_fingerprint(opening), "image_sha256": source_sha256,
            "image_size": [classified.get("meta", {}).get("width"),
                           classified.get("meta", {}).get("height")],
            "current": current, "bbox": bbox, "center": center,
            "reasons": sorted(reasons[uid]),
            "choices": ["door", "window", "ignore"],
            "severity": "error" if severity.get(uid) == 2 else "review",
        })
    questions.sort(key=lambda item: (item["severity"] != "error", item["opening_id"]))
    return questions[:MAX_ANSWERS]


def apply_review_decisions(classified, decisions, source_sha256):
    if not isinstance(decisions, dict) or decisions.get("version") != 1:
        raise ValueError("Unsupported review decisions format")
    if decisions.get("image_sha256") != source_sha256:
        raise ValueError("Review decisions belong to another image")
    answers = decisions.get("answers")
    if not isinstance(answers, list) or len(answers) > MAX_ANSWERS:
        raise ValueError("Invalid number of review answers")
    data = copy.deepcopy(classified)
    openings = data.get("doors", []) + data.get("windows", [])
    by_id = {opening.get("id"): opening for opening in openings}
    if len(by_id) != len(openings):
        raise ValueError("Opening IDs must be unique")
    selected = {}
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError("Invalid review answer")
        uid, choice = answer.get("opening_id"), answer.get("choice")
        if not isinstance(uid, int) or uid in selected or choice not in CHOICES:
            raise ValueError("Invalid or repeated opening choice")
        opening = by_id.get(uid)
        if opening is None or opening_fingerprint(opening) != answer.get("fingerprint"):
            raise ValueError("Review answer refers to a changed or missing opening")
        selected[uid] = choice
    rejected = data.setdefault("rejected_openings", [])
    for uid, choice in selected.items():
        opening = by_id[uid]
        previous = copy.deepcopy(opening.get("classification"))
        opening["type"] = choice
        opening["classification"] = {
            "method": "user_ignored" if choice == "ignore" else "user_selected",
            "confirmed": False, "requires_review": False,
            "user_confirmed": True, "previous": previous,
            "source_class": opening.get("source_class"),
        }
        if choice == "ignore":
            rejected.append(opening)
    accepted = [opening for opening in openings if opening["type"] != "ignore"]
    data["doors"] = [opening for opening in accepted if opening["type"] == "door"]
    data["windows"] = [opening for opening in accepted if opening["type"] == "window"]
    data["openings"] = accepted
    data["review_decisions"] = [{"opening_id": uid, "choice": choice}
                                for uid, choice in selected.items()]
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classified", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    classified = json.loads(args.classified.read_text(encoding="utf-8"))
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    try:
        reviewed = apply_review_decisions(classified, decisions, image_sha256(args.image))
    except ValueError as exc:
        print(f"REVIEW_DECISION_ERROR: {exc}", file=sys.stderr)
        return 3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reviewed, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Applied review choices: {len(reviewed['review_decisions'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
