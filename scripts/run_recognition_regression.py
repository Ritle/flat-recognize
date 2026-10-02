"""Measure the existing CLI pipeline without changing recognition algorithms.

Run sequentially: V1 writes intermediates to the shared output/ directory.
Only artifacts changed by this invocation are copied into the run archive.
Unknown ground truth remains unknown; CLI success is not recognition accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from process_floorplan import count_results

ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = (
    ".json", "_overlay.png", "_classified.json", "_classified_overlay.png",
    "_rooms.json", "_rooms_overlay.png", "_barrier.png", "_project.json",
    "_project_doors.json", "_window_bindings.json",
    "_preprocess.json", "_preprocessed.png", "_grayscale.png", "_preprocess_preview.png",
    "_closures.png",
    "_opening_bindings.json",
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stamp(path):
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def audit_project(path):
    """Basic integration invariants, not a production quality/confidence score."""
    errors = []
    data = load_json(path)
    levels = data.get("levels", [])
    if not levels:
        return ["No levels"]
    for index, level in enumerate(levels):
        prefix = f"level {index}: "
        connectors = {c["uuid"]: c for c in level.get("connectors", [])}
        segments = {s["uuid"]: s for s in level.get("segments", [])}
        edges = set()
        for segment in segments.values():
            start, end = segment.get("start"), segment.get("end")
            if start not in connectors or end not in connectors:
                errors.append(prefix + "Segment references missing connector")
                continue
            a, b = connectors[start], connectors[end]
            if (a["x"], a["y"]) == (b["x"], b["y"]):
                errors.append(prefix + "Zero-length segment")
            edges.add(frozenset((start, end)))
        if not level.get("areas"):
            errors.append(prefix + "No areas")
        for area in level.get("areas", []):
            cycle = area.get("connectors", [])
            if len(set(cycle)) < 3 or any(c not in connectors for c in cycle):
                errors.append(prefix + "Invalid area connectors")
                continue
            if any(frozenset((a, b)) not in edges
                   for a, b in zip(cycle, cycle[1:] + cycle[:1])):
                errors.append(prefix + "Area boundary has missing segment")
        for element in level.get("elements", []):
            if element.get("type") not in ("doorItem", "windowItem"):
                continue
            wall = element.get("wall", {})
            if wall.get("uuid") not in segments:
                errors.append(prefix + "Opening references missing wall")
            position = wall.get("position")
            if not isinstance(position, (int, float)) or not 0 <= position <= 100:
                errors.append(prefix + "Opening position outside segment")
            if any(m.get("itemUuid") != element.get("uuid")
                   for m in element.get("materials", [])):
                errors.append(prefix + f"{element['type']} material UUID mismatch")
            if element.get("type") == "doorItem" and not element.get("model"):
                errors.append(prefix + "Door has no model")
    return errors


def run_case(case, run_dir, env, pipeline_v2=False):
    image = ROOT / case["image"]
    case_dir = run_dir / case["id"]
    case_dir.mkdir()
    row = {"id": case["id"], "image": case["image"], "tags": case.get("tags", []),
           "expected": case.get("expected"), "artifacts": []}
    if not image.is_file():
        return {**row, "status": "missing_input", "seconds": 0}
    row["input_sha256"] = sha256(image)
    source_paths = [ROOT / "output" / (image.stem + suffix) for suffix in SUFFIXES]
    previous = {path: stamp(path) for path in source_paths}
    command = [sys.executable, str(ROOT / "scripts/process_floorplan.py"),
               str(image), "--output", str(case_dir / "project.json")]
    if pipeline_v2:
        command.append("--pipeline-v2")
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    row["seconds"] = round(time.perf_counter() - started, 2)
    row["exit_code"] = result.returncode
    (case_dir / "pipeline.log").write_text(result.stdout, encoding="utf-8")
    changed = {}
    for path in source_paths:
        if stamp(path) is not None and stamp(path) != previous[path]:
            destination = case_dir / path.name
            shutil.copy2(path, destination)
            row["artifacts"].append(path.name)
            changed[path.name] = destination
    prediction = changed.get(image.stem + ".json")
    classified = changed.get(image.stem + "_classified.json")
    rooms = changed.get(image.stem + "_rooms.json")
    if prediction:
        data = load_json(prediction)
        row["segmentation"] = {"wall_polygons": len(data.get("walls", [])),
                               **data.get("diagnostics", {})}
    if classified:
        data = load_json(classified)
        row["classified"] = {key: len(data.get(key, [])) for key in ("doors", "windows")}
    if rooms:
        data = load_json(rooms)
        row["rooms_found"] = len(data.get("rooms", []))
        row["room_diagnostics"] = data.get("diagnostics", {})
        row["recognition_warnings"] = row["room_diagnostics"].get("warnings", [])
    preprocessing = changed.get(image.stem + "_preprocess.json")
    if preprocessing:
        row["preprocessing"] = load_json(preprocessing)
    final = case_dir / "project.json"
    quality_path = final.with_suffix(".quality.json")
    if quality_path.exists():
        row["quality"] = load_json(quality_path)
        row["artifacts"].append(quality_path.name)
    if final.exists():
        row["artifacts"].append("project.json")
        row["final"] = count_results(final)
    row["status"] = "pipeline_failed"
    if result.returncode != 0:
        stage = re.search(r"ERROR: stage failed: (.+)", result.stdout)
        row["failed_stage"] = stage.group(1) if stage else "unknown"
        row["error_tail"] = result.stdout[-3000:]
        if row.get("quality", {}).get("status") == "invalid":
            row["status"] = "quality_invalid"
    elif final.exists():
        row["integration_errors"] = audit_project(final)
        row["expectation_match"] = (
            all(row["final"].get(k) == v for k, v in row["expected"].items())
            if row["expected"] else None
        )
        row["status"] = ("count_mismatch" if row["expectation_match"] is False else
                         "integration_failed" if row["integration_errors"] else "pipeline_ok")
    else:
        row["failed_stage"] = "missing final project"
    (case_dir / "result.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def write_report(report, run_dir):
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# {report['pipeline'].upper()} recognition regression", "",
             "Counts are observations. Only clean-baseline has confirmed expectations.", "",
             "| Case | Rooms | Classified D/W | Final D/W | Status | Quality | Seconds |",
             "|---|---:|---:|---:|---|---|---:|"]
    for row in report["cases"]:
        classified, final = row.get("classified", {}), row.get("final", {})
        stage = row.get("failed_stage", row["status"])
        lines.append(f"| [{row['id']}]({row['id']}/pipeline.log) | "
                     f"{row.get('rooms_found', '—')} | "
                     f"{classified.get('doors', '—')}/{classified.get('windows', '—')} | "
                     f"{final.get('doors', '—')}/{final.get('windows', '—')} | "
                     f"{stage} | {row.get('quality', {}).get('status', '—')} | {row['seconds']} |")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "test/recognition_cases.json")
    parser.add_argument("--case", action="append", help="Case ID; repeat to select several")
    parser.add_argument("--output", type=Path, help="New run directory; existing paths rejected")
    parser.add_argument("--threads", type=int, default=4, help="OMP/MKL CPU threads per process")
    parser.add_argument("--pipeline-v2", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    cases = load_json(args.manifest)["cases"]
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-z0-9-]+", i) for i in ids):
        parser.error("Case IDs must be unique safe directory names")
    if args.case:
        unknown = set(args.case) - set(ids)
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in args.case]
    required = [ROOT / "weights/best.safetensors", ROOT / "weights/config.yaml"]
    if any(not path.is_file() for path in required):
        parser.error("Download weights/best.safetensors and weights/config.yaml first")
    run_dir = args.output or ROOT / "output/regression" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": str(args.threads),
           "MKL_NUM_THREADS": str(args.threads), "BUILDINGCV_DEVICE": "cpu"}
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    report = {"pipeline": "v2" if args.pipeline_v2 else "v1",
              "started_utc": datetime.now(timezone.utc).isoformat(), "revision": revision,
              "python": platform.python_version(), "platform": platform.platform(),
              "cpu": platform.processor(), "cpu_threads": args.threads, "packages": packages,
              "manifest_sha256": sha256(args.manifest),
              "weights": {p.name: sha256(p) for p in required}, "cases": []}
    source_paths = sorted((ROOT / "scripts").glob("*.py")) + sorted(
        (ROOT / "src/buildingcv").glob("*.py"))
    report["source_sha256"] = {
        p.relative_to(ROOT).as_posix(): sha256(p) for p in source_paths
    }
    for case in cases:
        print(f"Running {case['id']} ...", flush=True)
        row = run_case(case, run_dir, env, pipeline_v2=args.pipeline_v2)
        report["cases"].append(row)
        write_report(report, run_dir)
        print(f"  {row['status']}: rooms={row.get('rooms_found', '—')}, "
              f"classified={row.get('classified')}, final={row.get('final')}, "
              f"{row['seconds']}s", flush=True)
    print(f"Report: {run_dir / 'report.md'}", flush=True)
    return int(any(row["status"] != "pipeline_ok" for row in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
