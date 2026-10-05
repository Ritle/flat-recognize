#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
OUTPUT = ROOT / "output"


def run_command(title, command):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(" ".join(str(x) for x in command))
    print()

    started = time.time()

    result = subprocess.run(
        [str(x) for x in command],
        cwd=ROOT
    )

    elapsed = time.time() - started

    if result.returncode != 0:
        print()
        print(
            f"ERROR: stage failed: {title}"
        )
        print(
            f"Exit code: {result.returncode}"
        )
        sys.exit(result.returncode)

    print()
    print(
        f"OK: {title} "
        f"({elapsed:.1f} sec)"
    )


def require_file(path, description):
    path = Path(path)

    if not path.exists():
        print()
        print(
            f"ERROR: {description} not found:"
        )
        print(path)
        sys.exit(1)

    return path


def count_results(final_json):
    with open(
        final_json,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    levels = data.get("levels", [])

    if not levels:
        return {
            "rooms": 0,
            "segments": 0,
            "connectors": 0,
            "doors": 0,
            "windows": 0,
        }

    level = levels[0]

    elements = level.get(
        "elements",
        []
    )

    return {
        "rooms": len(
            level.get("areas", [])
        ),
        "segments": len(
            level.get("segments", [])
        ),
        "connectors": len(
            level.get("connectors", [])
        ),
        "doors": sum(
            e.get("type") == "doorItem"
            for e in elements
        ),
        "windows": sum(
            e.get("type") == "windowItem"
            for e in elements
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Full raster floorplan -> project JSON pipeline"
        )
    )

    parser.add_argument(
        "image",
        help="Input PNG/JPG/WEBP floorplan"
    )

    parser.add_argument(
        "--pipeline-v2", action="store_true",
        help="Experimental structural ROI, normalization and room topology repair"
    )

    parser.add_argument(
        "--template",
        default="test/template.json",
        help=(
            "Base application JSON template "
            "containing doorItem"
        )
    )

    parser.add_argument(
        "--window-template",
        default="test/template_window.json",
        help=(
            "Application JSON containing windowItem"
        )
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Project scale: centimeters per pixel"
    )

    parser.add_argument(
        "--wall-thickness",
        type=float,
        default=20.0,
        help="Wall thickness in cm"
    )

    parser.add_argument(
        "--height",
        type=float,
        default=270.0,
        help="Wall height in cm"
    )

    parser.add_argument(
        "--door-height",
        type=float,
        default=202.5,
        help="Default door height in cm"
    )

    parser.add_argument(
        "--window-height",
        type=float,
        default=None,
        help=(
            "Window height in cm. "
            "Default = value from template"
        )
    )

    parser.add_argument(
        "--window-z",
        type=float,
        default=None,
        help=(
            "Window position.z. "
            "Default = value from template"
        )
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Final JSON output path"
    )

    args = parser.parse_args()

    python = sys.executable

    image = Path(args.image)

    if not image.is_absolute():
        image = ROOT / image

    require_file(
        image,
        "Input image"
    )

    template = Path(args.template)

    if not template.is_absolute():
        template = ROOT / template

    require_file(
        template,
        "Door/base template"
    )

    window_template = Path(
        args.window_template
    )

    if not window_template.is_absolute():
        window_template = (
            ROOT /
            window_template
        )

    require_file(
        window_template,
        "Window template"
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    stem = image.stem

    # -------------------------------------------------
    # Expected intermediate filenames
    # -------------------------------------------------

    prediction_json = (
        OUTPUT /
        f"{stem}.json"
    )

    prediction_overlay = (
        OUTPUT /
        f"{stem}_overlay.png"
    )

    classified_json = (
        OUTPUT /
        f"{stem}_classified.json"
    )

    classified_overlay = (
        OUTPUT /
        f"{stem}_classified_overlay.png"
    )

    rooms_json = (
        OUTPUT /
        f"{stem}_rooms.json"
    )

    rooms_overlay = (
        OUTPUT /
        f"{stem}_rooms_overlay.png"
    )

    barrier_image = (
        OUTPUT /
        f"{stem}_barrier.png"
    )

    project_json = (
        OUTPUT /
        f"{stem}_project.json"
    )

    project_doors_json = (
        OUTPUT /
        f"{stem}_project_doors.json"
    )

    window_bindings_json = (
        OUTPUT /
        f"{stem}_window_bindings.json"
    )
    opening_bindings_json = OUTPUT / f"{stem}_opening_bindings.json"
    geometry_report = OUTPUT / f"{stem}_geometry.json"
    export_overlay = OUTPUT / f"{stem}_export_overlay.png"

    if args.output:
        final_json = Path(
            args.output
        )

        if not final_json.is_absolute():
            final_json = (
                ROOT /
                final_json
            )
    else:
        final_json = (
            OUTPUT /
            f"{stem}_final.json"
        )

    final_json.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    quality_json = final_json.with_suffix(".quality.json")

    print()
    print("Floorplan processing pipeline")
    print()
    print(f"Input:            {image}")
    print(f"Template:         {template}")
    print(
        f"Window template:  "
        f"{window_template}"
    )
    print(f"Final output:     {final_json}")

    total_started = time.time()

    prediction_command = [python, SCRIPTS / "predict_raster.py", image]
    if args.pipeline_v2:
        run_command(
            "V2 - Input preprocessing",
            [python, SCRIPTS / "preprocess_v2.py", image]
        )
        preprocessing_meta = require_file(
            OUTPUT / f"{stem}_preprocess.json", "Preprocessing transform metadata"
        )
        prediction_command.extend(["--preprocess-meta", preprocessing_meta, "--dual-pass"])

    # =================================================
    # 1. Neural network inference
    # =================================================

    run_command(
        "1/6 - Floorplan segmentation",
        prediction_command
    )

    require_file(
        prediction_json,
        "Segmentation JSON"
    )

    # =================================================
    # 2. Door / window classification
    # =================================================

    run_command(
        "2/6 - Opening classification",
        [
            python,
            SCRIPTS /
            "classify_openings.py",
            image,
            prediction_json,
        ]
    )

    require_file(
        classified_json,
        "Classified openings JSON"
    )

    # =================================================
    # 3. Room extraction
    # =================================================

    run_command(
        "3/6 - Room extraction",
        [
            python,
            SCRIPTS /
            ("extract_rooms_v2.py" if args.pipeline_v2 else "extract_rooms.py"),
            image,
            classified_json,
        ]
    )

    require_file(
        rooms_json,
        "Rooms JSON"
    )

    if args.pipeline_v2:
        run_command("V2 - Recover missed doors", [python, SCRIPTS / "recover_openings_v2.py",
                                                image, classified_json, rooms_json])
        classified_json = require_file(OUTPUT / f"{stem}_recovered_classified.json", "Recovered openings")
        rooms_json = require_file(OUTPUT / f"{stem}_rooms_recovered.json", "Rooms with recovered openings")
        room_data = json.loads(rooms_json.read_text(encoding="utf-8"))
        for warning in room_data.get("diagnostics", {}).get("warnings", []):
            print(f"WARNING: {warning}")

    validation_command = [python, SCRIPTS / "validate_recognition.py",
                          "--rooms", rooms_json, "--scale", str(args.scale),
                          "--output", quality_json]
    run_command("Quality gate - Room geometry", validation_command)

    # =================================================
    # 4. Project geometry
    # =================================================

    run_command(
        "4/6 - Project geometry",
        [
            python,
            SCRIPTS /
            "build_project_json.py",
            rooms_json,
            template,
            "--output",
            project_json,
            "--scale",
            str(args.scale),
            "--wall-thickness",
            str(args.wall_thickness),
            "--height",
            str(args.height),
            "--geometry-mode", "auto" if args.pipeline_v2 else "bbox",
            *(["--internal-walls"] if args.pipeline_v2 else []),
            "--geometry-report", geometry_report,
            "--source-image", image,
            "--overlay", export_overlay,
        ]
    )

    require_file(
        project_json,
        "Project geometry JSON"
    )
    validation_command.extend(["--geometry", geometry_report])
    run_command("Quality gate - Export geometry", validation_command + ["--project", project_json])

    # =================================================
    # 5. Doors + window bindings
    # =================================================

    run_command(
        "5/6 - Add doors and bind windows",
        [
            python,
            SCRIPTS /
            "add_openings.py",
            project_json,
            classified_json,
            rooms_json,
            template,
            "--output",
            project_doors_json,
            "--windows-output",
            window_bindings_json,
            "--door-height",
            str(args.door_height),
            "--bindings-output",
            opening_bindings_json,
            "--geometry-report",
            geometry_report,
        ]
    )

    require_file(
        project_doors_json,
        "Project with doors"
    )

    require_file(
        window_bindings_json,
        "Window bindings"
    )

    # =================================================
    # 6. Windows
    # =================================================

    window_cmd = [
        python,
        SCRIPTS /
        "add_windows.py",
        project_doors_json,
        window_bindings_json,
        window_template,
        "--output",
        final_json,
    ]

    if args.window_height is not None:
        window_cmd.extend([
            "--window-height",
            str(args.window_height),
        ])

    if args.window_z is not None:
        window_cmd.extend([
            "--window-z",
            str(args.window_z),
        ])

    run_command(
        "6/6 - Add windows",
        window_cmd
    )

    require_file(
        final_json,
        "Final project JSON"
    )
    run_command("Quality gate - Final project", validation_command +
                ["--project", final_json, "--bindings", opening_bindings_json])

    # =================================================
    # Final validation
    # =================================================

    try:
        results = count_results(
            final_json
        )
    except Exception as exc:
        print()
        print(
            "WARNING: final JSON exists "
            "but diagnostics failed:"
        )
        print(exc)
        results = None

    total_time = (
        time.time() -
        total_started
    )

    print()
    print("=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)
    print()

    if results:
        print(
            f"Rooms:      "
            f"{results['rooms']}"
        )

        print(
            f"Walls:      "
            f"{results['segments']}"
        )

        print(
            f"Connectors: "
            f"{results['connectors']}"
        )

        print(
            f"Doors:      "
            f"{results['doors']}"
        )

        print(
            f"Windows:    "
            f"{results['windows']}"
        )

        print()

    print(
        f"Total time: {total_time:.1f} sec"
    )

    print()
    print("FINAL JSON:")
    print(final_json)
    print(f"Quality report: {quality_json}")

    print()
    print("Diagnostics:")

    for diagnostic in (
        prediction_overlay,
        classified_overlay,
        rooms_overlay,
        barrier_image,
        export_overlay,
        *([OUTPUT / f"{stem}_preprocess_preview.png", OUTPUT / f"{stem}_closures.png",
           OUTPUT / f"{stem}_opening_recovery_overlay.png",
           OUTPUT / f"{stem}_segmentation_original.png",
           OUTPUT / f"{stem}_segmentation_normalized.png",
           OUTPUT / f"{stem}_segmentation_fused.png"]
          if args.pipeline_v2 else []),
    ):
        if diagnostic.exists():
            print(
                f"  {diagnostic}"
            )


if __name__ == "__main__":
    main()
