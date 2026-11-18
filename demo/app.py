import asyncio
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from html import escape
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from scripts.review_decisions import CHOICES, MAX_ANSWERS, make_review_questions
from scripts.review_rooms import ROOM_CHOICES, room_questions
from scripts.review_partitions import PARTITION_CHOICES, partition_questions
from scripts.review_exterior_enclosures import EXTERIOR_CHOICES, exterior_questions
from scripts.graph_scope import validate_graph_scope
from scripts.preprocess_v2 import preprocess
from demo.service_info import format_updated_at, load_service_info


ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo"
JOBS_DIR = ROOT / "demo_jobs"
OUTPUT_DIR = ROOT / "output"
FOCUS_REVIEW = ROOT / "focus_plan"


def saved_focus_profile(digest, data=b""):
    path = FOCUS_REVIEW / "recognition-profile.json"
    if not path.is_file():
        return None
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile["image_sha256"] == digest:
        return profile
    # Only lossless, single-frame PNGs with the same opaque pixels and axes
    # can reuse measurements recorded in the original image coordinates.
    if (not profile.get("opaque_rgb_sha256") or not data.startswith(b"\x89PNG\r\n\x1a\n")
            or len(data) < 25 or data[24] > 8):
        return None
    try:
        with Image.open(io.BytesIO(data)) as raster:
            if list(raster.size) != profile["image_size"] or raster.is_animated:
                return None
            rgba = raster.convert("RGBA")
            if rgba.getchannel("A").getextrema() != (255, 255) or raster.getexif().get(274, 1) != 1:
                return None
            pixels = hashlib.sha256(rgba.convert("RGB").tobytes()).hexdigest()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None
    return profile if pixels == profile["opaque_rgb_sha256"] else None


def saved_review_payload(name, profile_digest, upload_digest):
    review = json.loads((FOCUS_REVIEW / name).read_text(encoding="utf-8"))
    if review.get("image_sha256") != profile_digest:
        raise HTTPException(status_code=409, detail="Saved review does not match its recognition profile")
    return {**review, "image_sha256": upload_digest}


def valid_review_answer(answer, pipeline_v2):
    if not isinstance(answer, dict) or not isinstance(answer.get("fingerprint"), str):
        return False
    choices = {"opening_id": CHOICES, "room_id": ROOM_CHOICES}
    if pipeline_v2:
        choices.update(partition_id=PARTITION_CHOICES, exterior_id=EXTERIOR_CHOICES)
    keys = [key for key in ("opening_id", "room_id", "partition_id", "exterior_id") if key in answer]
    return (len(keys) == 1 and keys[0] in choices and type(answer[keys[0]]) is int
            and isinstance(answer.get("choice"), str)
            and answer.get("choice") in choices[keys[0]])

PIPELINE = ROOT / "scripts" / "process_floorplan.py"
GRAPH_PIPELINE = ROOT / "scripts" / "graph_service.py"

TEMPLATE = ROOT / "test" / "template.json"
WINDOW_TEMPLATE = ROOT / "test" / "template_window.json"

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

MAX_FILE_SIZE = 20 * 1024 * 1024
SERVICE_INFO = load_service_info(ROOT)


app = FastAPI(
    title="Floorplan Recognition Demo",
    version=SERVICE_INFO["version"],
)


def validate_job_id(job_id: str):
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid job id"
        )


def load_stats(project_path: Path):
    with project_path.open(
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    levels = data.get("levels", [])

    if not levels:
        return {
            "rooms": 0,
            "walls": 0,
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
        "walls": len(
            level.get("segments", [])
        ),
        "connectors": len(
            level.get("connectors", [])
        ),
        "doors": sum(
            element.get("type") == "doorItem"
            for element in elements
        ),
        "passages": sum(element.get("type") == "wallOpening" for element in elements),
        "windows": sum(
            element.get("type") == "windowItem"
            for element in elements
        ),
    }


def run_pipeline(
    image_path: Path,
    project_path: Path,
    pipeline_v2: bool = False,
    review_decisions: Path | None = None,
    light_fill_cleanup: bool = False,
    *, graph_inputs_only: bool = False, scale_cm_per_px: float | None = None,
):
    command = [
        sys.executable,
        str(PIPELINE),
        str(image_path),
        "--template",
        str(TEMPLATE),
        "--window-template",
        str(WINDOW_TEMPLATE),
        "--output",
        str(project_path),
    ]

    if pipeline_v2:
        command.append("--pipeline-v2")
    if light_fill_cleanup:
        command.append("--light-fill-cleanup")
    if graph_inputs_only:
        command.append("--graph-inputs-only")
    if scale_cm_per_px is not None:
        command.extend(["--scale", str(scale_cm_per_px)])
    if review_decisions is not None:
        command.extend(["--review-decisions", str(review_decisions)])

    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = DEMO_DIR / "index.html"

    if not index_path.exists():
        raise HTTPException(
            status_code=500,
            detail="demo/index.html not found"
        )

    html = index_path.read_text(encoding="utf-8")
    html = html.replace('<span id="serviceVersion">недоступна</span>',
                        f'<span id="serviceVersion">{escape(SERVICE_INFO["version"])}</span>')
    html = html.replace('<span id="serviceUpdatedAt">не указана</span>',
                        f'<span id="serviceUpdatedAt">{escape(format_updated_at(SERVICE_INFO["updated_at"]))}</span>')
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/api/version")
async def service_version():
    return JSONResponse(SERVICE_INFO, headers={"Cache-Control": "no-store"})


def create_upload(data, extension):
    job_id = str(uuid.uuid4())
    directory = JOBS_DIR / job_id
    directory.mkdir(parents=True)
    image = directory / f"{job_id}{extension}"
    image.write_bytes(data)
    return image


def simple_plan_candidate(image):
    with Image.open(image) as preview:
        preview.thumbnail((896, 896))
        _, _, meta = preprocess(preview, light_fill_cleanup=True)
    hints = meta["diagnostics"]
    return (hints.get("structural_candidates") == 1 and hints.get("fallback") == "already_tightly_framed"
            and hints.get("normalization") == "original" and "wall_level" in hints)


def acceptable_simple_project(image, project, result, quality):
    if (result.returncode or not quality or quality.get("status") != "good"
            or not quality.get("export_allowed") or not project.is_file()):
        return False
    # V1 uses bounding boxes. Compare them to the actual room contours before
    # accepting that representation on a previously unseen plan.
    rooms_file = OUTPUT_DIR / f"{image.stem}_rooms.json"
    if not rooms_file.is_file():
        return False
    rooms = json.loads(rooms_file.read_text(encoding="utf-8")).get("rooms", [])
    for room in rooms:
        ring = room.get("polygon", [])
        if len(ring) < 3:
            return False
        polygon = Polygon(ring)
        if not polygon.is_valid or polygon.area <= 0 or polygon.area < .95 * polygon.envelope.area:
            return False
    return bool(rooms)


@app.post("/api/recognize")
async def recognize(
    file: UploadFile = File(...),
    pipeline_v2: Annotated[bool | None, Form()] = None,
    decisions: str | None = Form(None),
    light_fill_cleanup: Annotated[bool | None, Form()] = None,
    use_saved_review: Annotated[bool, Form()] = True,
):
    if light_fill_cleanup and pipeline_v2 is False:
        raise HTTPException(status_code=400, detail="Light fill cleanup requires V2")
    if decisions is not None and not isinstance(decisions, str):
        decisions = None
    original_name = (
        file.filename or "floorplan.jpg"
    )

    extension = Path(
        original_name
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Поддерживаются JPG, PNG и WEBP"
            )
        )

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Файл пустой"
        )

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Максимальный размер файла — 20 МБ"
        )

    source_hash = hashlib.sha256(data).hexdigest()
    review_payload = None
    applied_decisions = []
    saved_answers = DEMO_DIR / "reviews" / f"{source_hash}.json"
    has_saved_answers = pipeline_v2 is not False and saved_answers.is_file()
    pipeline_options = {}
    saved_ink = None
    if use_saved_review and has_saved_answers:
        saved_payload = json.loads(saved_answers.read_text(encoding="utf-8"))
        if saved_payload["image_sha256"] != source_hash:
            raise HTTPException(status_code=409, detail="Saved review belongs to another image")
        saved_ink = saved_payload.get("reviewed_ink_layout")
        if decisions is None:
            decisions = json.dumps(saved_payload)
        if dimension := saved_payload.get("dimension"):
            scale = dimension["length_cm"] / math.dist(*dimension["span_px"])
            pipeline_options["scale_cm_per_px"] = scale
        elif reference := saved_payload.get("area_reference"):
            pipeline_options["scale_cm_per_px"] = math.sqrt(
                10_000 * reference["area_m2"] / math.prod(reference["interior_size_px"]))
    if decisions is not None:
        if len(decisions) > 16_384:
            raise HTTPException(status_code=400, detail="Too many review decisions")
        try:
            review_payload = json.loads(decisions)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid review decisions JSON") from exc
        answers = review_payload.get("answers") if isinstance(review_payload, dict) else None
        if (not isinstance(review_payload, dict) or review_payload.get("version") != 1
                or review_payload.get("image_sha256") != source_hash
                or not isinstance(answers, list) or len(answers) > MAX_ANSWERS
                or any(not valid_review_answer(answer, pipeline_v2 is not False) for answer in answers)):
            raise HTTPException(status_code=400, detail="Review decisions do not match this image")
        applied_decisions = answers
        review_payload.pop("reviewed_ink_layout", None)
        if saved_ink:
            review_payload["reviewed_ink_layout"] = saved_ink
    profile = await asyncio.to_thread(saved_focus_profile, source_hash, data) if not applied_decisions else None
    has_opening_answers = any("opening_id" in answer for answer in applied_decisions)
    has_partition_answers = any("partition_id" in answer for answer in applied_decisions)

    image_path = create_upload(data, extension)
    job_id, job_dir = image_path.stem, image_path.parent

    if profile and use_saved_review:
        return await recognize_saved_plan(image_path, original_name, profile)

    decisions_path = None
    if applied_decisions or saved_ink:
        decisions_path = job_dir / "review_decisions.json"
        decisions_path.write_text(json.dumps(review_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    project_path = (
        job_dir /
        "project.json"
    )

    started = time.time()

    auto_simple = False
    selection = "explicit"
    if pipeline_v2 is None:
        if not applied_decisions and not saved_ink and light_fill_cleanup is None:
            try:
                auto_simple = await asyncio.to_thread(simple_plan_candidate, image_path)
            except (OSError, Image.DecompressionBombError) as exc:
                raise HTTPException(status_code=400, detail="Не удалось прочитать планировку") from exc
        pipeline_v2 = not auto_simple
        selection = "simple_preview" if auto_simple else "v2"
    previous_job = None
    while True:
        cleanup = pipeline_v2 if light_fill_cleanup is None else light_fill_cleanup
        pipeline_args = [image_path, project_path, pipeline_v2, decisions_path]
        if cleanup:
            pipeline_args.append(True)
        result = await asyncio.to_thread(run_pipeline, *pipeline_args, **pipeline_options)
        quality_path = project_path.with_suffix(".quality.json")
        quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else None
        if not auto_simple or acceptable_simple_project(image_path, project_path, result, quality):
            break
        previous_job = job_id
        image_path = create_upload(data, extension)
        job_id, job_dir = image_path.stem, image_path.parent
        project_path = job_dir / "project.json"
        auto_simple, pipeline_v2, selection = False, True, "v2_after_simple_check"

    elapsed = (
        time.time() -
        started
    )

    rejected = quality is not None and quality.get("status") == "invalid"
    if result.returncode != 0:
        error_text = (
            result.stderr
            or result.stdout
            or "Unknown pipeline error"
        )

        error_path = (
            job_dir /
            "error.log"
        )

        error_path.write_text(
            error_text,
            encoding="utf-8"
        )

        if (decisions_path is not None and result.returncode == 3
                and "REVIEW_DECISION_ERROR:" in (result.stderr or "")):
            raise HTTPException(status_code=409, detail="Выбор устарел: загрузите план заново и уточните проём ещё раз")

        if not rejected:
            raise HTTPException(
                status_code=500,
                detail={"message": "Ошибка обработки планировки", "log": error_text[-4000:]},
            )

    if not rejected and (not project_path.exists() or quality is None):
        raise HTTPException(
            status_code=500,
            detail="Pipeline did not create project.json and quality report"
        )

    room_path = OUTPUT_DIR / f"{job_id}_rooms.json"
    recovered_room_path = OUTPUT_DIR / f"{job_id}_rooms_recovered.json"
    if pipeline_v2 and recovered_room_path.exists():
        shutil.copy2(room_path, job_dir / "rooms_before_recovery.json")
        room_path = recovered_room_path
    filtered_room_path = OUTPUT_DIR / f"{job_id}_rooms_filtered.json"
    if pipeline_v2 and filtered_room_path.exists():
        room_path = filtered_room_path
    reconciled_room_path = OUTPUT_DIR / f"{job_id}_rooms_topology_reconciled.json"
    if pipeline_v2 and reconciled_room_path.exists():
        room_path = reconciled_room_path
    reviewed_dir = OUTPUT_DIR / f"{job_id}_reviewed"
    if pipeline_v2 and has_opening_answers:
        room_path = reviewed_dir / f"{job_id}_rooms.json"
    if decisions_path is not None:
        room_path = (reviewed_dir / f"{job_id}_rooms_selected.json" if pipeline_v2 else
                     OUTPUT_DIR / f"{job_id}_rooms_selected.json")
    room_data = json.loads(room_path.read_text(encoding="utf-8"))
    metrics = quality.get("metrics", {})
    stats = load_stats(project_path) if not rejected else {
        "rooms": metrics.get("exported_rooms", len(room_data.get("rooms", []))),
        "walls": metrics.get("exported_segments", 0),
        "connectors": metrics.get("exported_connectors", 0),
        "doors": len(room_data.get("doors", [])),
        "windows": len(room_data.get("windows", [])),
    }
    shutil.copy2(quality_path, job_dir / "quality.json")
    shutil.copy2(room_path, job_dir / "rooms.json")
    classified_path = OUTPUT_DIR / f"{job_id}_classified.json"
    if pipeline_v2:
        reconciled_classified_path = OUTPUT_DIR / f"{job_id}_topology_reconciled_classified.json"
        classified_path = (reconciled_classified_path if reconciled_classified_path.exists() else
                           OUTPUT_DIR / f"{job_id}_interior_filtered_classified.json")
    if has_opening_answers:
        classified_path = OUTPUT_DIR / f"{job_id}_reviewed_classified.json"
    if pipeline_v2 and has_partition_answers:
        classified_path = reviewed_dir / f"{job_id}_partition_reviewed_classified.json"
    classified_data = json.loads(classified_path.read_text(encoding="utf-8"))
    shutil.copy2(classified_path, job_dir / "classified.json")
    review_questions = (make_review_questions(classified_data, quality, source_hash)
                        + room_questions(room_data, source_hash)
                        + (partition_questions(room_data, source_hash) if pipeline_v2 else []))
    binding_path = OUTPUT_DIR / f"{job_id}_opening_bindings.json"
    if binding_path.exists():
        shutil.copy2(binding_path, job_dir / "opening_bindings.json")
    preflight_path = OUTPUT_DIR / f"{job_id}_opening_preflight.json"
    exterior_candidates = []
    exterior_path = OUTPUT_DIR / f"{job_id}_exterior_enclosures.json"
    if pipeline_v2 and exterior_path.exists():
        shutil.copy2(exterior_path, job_dir / "exterior_enclosures.json")
        exterior_report = json.loads(exterior_path.read_text(encoding="utf-8"))
        exterior_candidates = exterior_report.get("candidates", [])
        review_questions += exterior_questions(exterior_report, source_hash,
            [room_data.get("meta", {}).get("width"), room_data.get("meta", {}).get("height")])
    review_questions = review_questions[:MAX_ANSWERS]
    opening_preflight = None
    if preflight_path.exists():
        shutil.copy2(preflight_path, job_dir / "opening_preflight.json")
        preflight_data = json.loads(preflight_path.read_text(encoding="utf-8"))
        opening_preflight = {"status": preflight_data.get("status"),
                             "summary": preflight_data.get("summary", {}),
                             "wall_geometry": {
                                 key: preflight_data.get("wall_geometry", {}).get(key, 0)
                                 for key in ("sampled", "suspect")}}
    geometry_path = OUTPUT_DIR / f"{job_id}_geometry.json"
    geometry_review_candidates = []
    if geometry_path.exists():
        shutil.copy2(geometry_path, job_dir / "geometry.json")
        geometry_data = json.loads(geometry_path.read_text(encoding="utf-8"))
        geometry_review_candidates = [
            {**candidate, "image_size": [room_data.get("meta", {}).get("width"),
                                         room_data.get("meta", {}).get("height")]}
            for candidate in geometry_data.get("unsupported_closure_groups", [])
        ]
    interior_report = OUTPUT_DIR / f"{job_id}_interior_windows.json"
    if interior_report.exists():
        shutil.copy2(interior_report, job_dir / "interior_windows.json")
    candidate_path = OUTPUT_DIR / f"{job_id}_project.json"
    if candidate_path.exists():
        shutil.copy2(candidate_path, job_dir / "geometry_candidate.json")

    # ------------------------------------------------
    # Забираем diagnostic images,
    # которые создал pipeline.
    # ------------------------------------------------

    generated_images = {
        "segmentation":
            OUTPUT_DIR /
            f"{job_id}_overlay.png",

        "classified":
            OUTPUT_DIR /
            f"{job_id}_classified_overlay.png",

        "rooms":
            OUTPUT_DIR /
            f"{job_id}_rooms_overlay.png",

        "barrier":
            OUTPUT_DIR /
            f"{job_id}_barrier.png",
        "export": OUTPUT_DIR / f"{job_id}_export_overlay.png",
    }
    if pipeline_v2 and has_opening_answers:
        generated_images.update({
            "barrier": reviewed_dir / f"{job_id}_barrier.png",
        })
    if decisions_path is not None:
        generated_images["rooms"] = room_path.with_name(f"{job_id}_rooms_selected_overlay.png")

    warnings = quality["warnings"]
    inference = None
    light_fill_applied = False
    if pipeline_v2:
        if exterior_candidates:
            generated_images["exterior_enclosures"] = OUTPUT_DIR / f"{job_id}_exterior_enclosures_overlay.png"
            if any(candidate.get("trial") for candidate in exterior_candidates):
                generated_images["exterior_trial"] = OUTPUT_DIR / f"{job_id}_exterior_trial_barrier.png"
        generated_images.update({
            "preprocessing": OUTPUT_DIR / f"{job_id}_preprocess_preview.png",
            "normalized": OUTPUT_DIR / f"{job_id}_preprocessed.png",
            "segmentation_original": OUTPUT_DIR / f"{job_id}_segmentation_original.png",
            "segmentation_normalized": OUTPUT_DIR / f"{job_id}_segmentation_normalized.png",
            "segmentation_fused": OUTPUT_DIR / f"{job_id}_segmentation_fused.png",
            "closures": OUTPUT_DIR / f"{job_id}_closures.png",
            "recovery": OUTPUT_DIR / f"{job_id}_opening_recovery_overlay.png",
            "topology_windows": OUTPUT_DIR / f"{job_id}_topology_windows_overlay.png",
            "wall_geometry": OUTPUT_DIR / f"{job_id}_wall_geometry_overlay.png",
        })
        if has_opening_answers:
            generated_images["closures"] = reviewed_dir / f"{job_id}_closures.png"
        topology_report_path = OUTPUT_DIR / f"{job_id}_topology_reconciliation.json"
        if topology_report_path.exists():
            shutil.copy2(topology_report_path, job_dir / "topology_reconciliation.json")
            topology_report = json.loads(topology_report_path.read_text(encoding="utf-8"))
            if topology_report.get("removed_from_room_mask") and not has_opening_answers:
                if decisions_path is None:
                    generated_images["rooms"] = OUTPUT_DIR / f"{job_id}_topology_reconciled_overlay.png"
                generated_images["barrier"] = OUTPUT_DIR / f"{job_id}_topology_reconciled_barrier.png"
                generated_images["closures"] = OUTPUT_DIR / f"{job_id}_topology_reconciled_closures.png"
        if any(item.get("choice") in ("separate", "door")
               for item in room_data.get("reviewed_partitions", [])):
            generated_images["barrier"] = reviewed_dir / f"{job_id}_partition_reviewed_barrier.png"
            generated_images["closures"] = reviewed_dir / f"{job_id}_partition_reviewed_closures.png"
        interior_counts = (json.loads(interior_report.read_text(encoding="utf-8"))
                           if interior_report.exists() else {})
        if interior_counts.get("rejected_windows", 0) or interior_counts.get("ambiguous_windows", 0):
            generated_images["interior_windows"] = (
                OUTPUT_DIR / f"{job_id}_interior_windows_overlay.png")
        for suffix in ("_preprocess.json", "_inference.json"):
            shutil.copy2(OUTPUT_DIR / f"{job_id}{suffix}", job_dir / suffix[1:])
        inference = json.loads((job_dir / "inference.json").read_text(encoding="utf-8"))
        preprocessing = json.loads((job_dir / "preprocess.json").read_text(encoding="utf-8"))
        light_fill_applied = preprocessing.get("diagnostics", {}).get("light_fill_applied", False)
        for suffix in ("_opening_recovery.json", "_recovered_classified.json", "_classified.json",
                       "_topology_windows.json", "_topology_classified.json",
                       "_interior_filtered_classified.json", "_rooms_filtered.json",
                       "_topology_reconciled_classified.json", "_rooms_topology_reconciled.json"):
            source = OUTPUT_DIR / f"{job_id}{suffix}"
            if source.exists():
                name = "initial_classified.json" if suffix == "_classified.json" else suffix[1:]
                shutil.copy2(source, job_dir / name)

    partition_candidates = room_data.get("diagnostics", {}).get("possible_room_partitions", [])
    reviewed_partitions = {item["partition_id"]: item["choice"]
                           for item in room_data.get("reviewed_partitions", [])}
    if pipeline_v2 and partition_candidates:
        with Image.open(image_path) as original:
            overlay = original.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for room in room_data.get("rooms", []):
            ring = [tuple(point) for point in room.get("polygon", [])]
            if len(ring) >= 3:
                draw.line(ring + ring[:1], fill="magenta", width=3)
                draw.text(tuple(room["center"]), f"Room {room['id'] + 1}", fill="magenta")
        for index, candidate in enumerate(partition_candidates):
            choice = reviewed_partitions.get(index)
            draw.line([tuple(candidate["start"]), tuple(candidate["end"])],
                      fill="lime" if choice == "door" else "cyan" if choice == "separate"
                      else "orange",
                      width=3)
            hint = candidate.get("unconfirmed_leaf_hint")
            if hint:
                draw.line([tuple(hint["hinge"]), tuple(hint["tip"])], fill="blue", width=3)
        partition_overlay_path = job_dir / "possible_room_partitions.png"
        overlay.save(partition_overlay_path)
        generated_images["rooms"] = partition_overlay_path

    images = {}

    for kind, source in generated_images.items():
        if not source.exists():
            continue

        destination = (
            job_dir /
            f"{kind}.png"
        )

        shutil.copy2(
            source,
            destination
        )

        images[kind] = (
            f"/api/jobs/{job_id}/image/{kind}"
        )

    metadata = {
        "job_id": job_id,
        "filename": original_name,
        "seconds": round(
            elapsed,
            2
        ),
        "stats": stats,
        "images": images,
        "pipeline": "v2" if pipeline_v2 else "v1",
        "light_fill_cleanup": cleanup,
        "light_fill_applied": light_fill_applied,
        "pipeline_selection": selection,
        "first_attempt_job_id": previous_job,
        "warnings": warnings,
        "quality": quality,
        "inference": inference,
        "review_questions": review_questions,
        "geometry_review_candidates": geometry_review_candidates,
        "possible_room_partitions": room_data.get("diagnostics", {}).get(
            "possible_room_partitions", []),
        "opening_preflight": opening_preflight,
        "exterior_enclosures": exterior_candidates,
        "applied_decisions": applied_decisions,
        "image_sha256": source_hash,
        "use_saved_review": use_saved_review,
        "saved_review_available": bool(profile) or has_saved_answers,
    }

    (
        job_dir /
        "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return {
        "ok": quality["export_allowed"],
        **metadata,
        "original_url": (
            f"/api/jobs/{job_id}/image/original"
        ),
        "quality_url": f"/api/jobs/{job_id}/quality",
        "opening_preflight_url": (
            f"/api/jobs/{job_id}/opening-preflight"
        ) if opening_preflight is not None else None,
        "download_url": (
            f"/api/jobs/{job_id}/download"
        ) if quality["export_allowed"] else None,
        "json_url": (
            f"/api/jobs/{job_id}/json"
        ) if quality["export_allowed"] else None,
    }


async def recognize_saved_plan(image, filename, profile):
    started = time.monotonic()
    parent = image.parent
    result = await asyncio.to_thread(run_pipeline, image, parent / "project.json", True, None, True,
                                     graph_inputs_only=True)
    if result.returncode:
        (parent / "error.log").write_text(result.stderr or result.stdout, encoding="utf-8")
        raise HTTPException(status_code=500, detail="Ошибка сегментации или классификации проёмов.")
    for suffix, name in ((".json", "prediction.json"), ("_classified.json", "classified.json")):
        source = OUTPUT_DIR / (image.stem + suffix)
        if not source.is_file():
            raise HTTPException(status_code=500, detail="Отсутствуют результаты распознавания для графа.")
        shutil.copy2(source, parent / name)
    graph = await build_graph_job(
        image.stem, None, profile["scale_cm_per_px"], json.dumps(profile["scope"]),
        saved_profile_sha256=profile["image_sha256"],
        wall_thickness_cm=profile.get("wall_thickness_cm"))
    graph.update(ok=graph["quality"]["export_allowed"], filename=filename,
                 seconds=round(time.monotonic() - started, 2), saved_review=True)
    metadata = dict(pipeline="graph-inputs", filename=filename, image_sha256=graph["image_sha256"],
                    saved_profile_sha256=profile["image_sha256"],
                    use_saved_review=True, graph_job_id=graph["job_id"])
    (parent / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    (JOBS_DIR / graph["job_id"] / "metadata.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return graph


def run_graph_pipeline(image, source, prediction, destination, graph=None, answers=None, scale=1.0, scope=None,
                       saved_review=False, wall_thickness_cm=None):
    command = [sys.executable, str(GRAPH_PIPELINE), str(image), str(source), str(prediction), str(destination),
               "--scale", str(scale)]
    if graph is not None:
        command.extend(["--graph", str(graph)])
    if answers is not None:
        command.extend(["--answers", str(answers)])
    if scope is not None:
        command.extend(["--scope", str(scope)])
    if wall_thickness_cm is not None:
        command.extend(["--wall-thickness-cm", str(wall_thickness_cm)])
    if saved_review:
        command.extend(["--furniture-review", str(destination / "furniture-review.json"),
                        "--allow-single-door-for-paired-leaves"])
        axes = destination / "wall-axis-review.json"
        if axes.is_file():
            command.extend(["--wall-axis-review", str(axes)])
    return subprocess.run(command, cwd=ROOT, capture_output=True, encoding="utf-8",
                          env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})


@app.post("/api/jobs/{job_id}/graph-review")
async def graph_review(job_id: str, decisions: str | None = Form(None), scale: float = Form(1.0),
                       scope: str | None = Form(None)):
    """Build or reuse the graph, then apply any answers to its current questions."""
    return await build_graph_job(job_id, decisions, scale, scope)


async def build_graph_job(job_id, decisions, scale, scope, *, saved_profile_sha256=None, wall_thickness_cm=None):
    validate_job_id(job_id)
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise HTTPException(status_code=400, detail="Scale must be a positive finite number")
    parent = JOBS_DIR / job_id
    if not parent.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")
    originals = [p for p in parent.iterdir() if p.stem == job_id and p.suffix.lower() in ALLOWED_EXTENSIONS]
    source = parent / "classified.json"
    prediction = parent / "prediction.json"
    if not prediction.exists():
        prediction = OUTPUT_DIR / f"{job_id}.json"
    if len(originals) != 1 or not source.is_file() or not prediction.is_file():
        raise HTTPException(status_code=409, detail="Recognition artifacts are unavailable for graph review")
    parent_graph = parent / "graph.json"
    if not isinstance(scope, str):
        scope = None
    scope_payload = None
    if scope is not None:
        if parent_graph.is_file() or len(scope) > 16384:
            raise HTTPException(status_code=400, detail="Excluded regions require a new graph")
        try:
            scope_payload = json.loads(scope)
            with Image.open(originals[0]) as raster:
                validate_graph_scope(scope_payload, raster.size)
        except (ValueError, TypeError, OSError) as exc:
            raise HTTPException(status_code=400, detail="Invalid excluded regions") from exc
    if not isinstance(decisions, str):
        decisions = None
    payload = None
    if saved_profile_sha256 is not None or decisions is not None:
        digest = hashlib.sha256(originals[0].read_bytes()).hexdigest()
    if saved_profile_sha256 is not None:
        payload = saved_review_payload("opening-answers.json", saved_profile_sha256, digest)
    elif decisions is not None:
        if len(decisions) > 16384:
            raise HTTPException(status_code=400, detail="Graph answers payload is too large")
        try:
            payload = json.loads(decisions)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid graph answers JSON") from exc
    if decisions is not None or saved_profile_sha256 is not None:
        rows = payload.get("answers") if isinstance(payload, dict) else None
        if (not isinstance(payload, dict) or payload.get("version") != 1
                or payload.get("image_sha256") != digest or not isinstance(rows, list)
                or len(rows) > MAX_ANSWERS or any(not isinstance(row, dict) for row in rows)):
            raise HTTPException(status_code=400, detail="Graph answers do not match this image")
    new_id = str(uuid.uuid4())
    job = JOBS_DIR / new_id
    job.mkdir(parents=True)
    image = job / f"{new_id}{originals[0].suffix.lower()}"
    shutil.copy2(originals[0], image)
    shutil.copy2(source, job / "classified.json")
    shutil.copy2(prediction, job / "prediction.json")
    answers_path = None
    if payload is not None:
        answers_path = job / "graph_answers.json"
        answers_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    scope_path = None
    if scope_payload is not None:
        scope_path = job / "graph_scope.json"
        scope_path.write_text(json.dumps(scope_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.time()
    options = {}
    if wall_thickness_cm is not None:
        options["wall_thickness_cm"] = wall_thickness_cm
    if saved_profile_sha256 is not None:
        for name in ("furniture-review.json", "wall-axis-review.json"):
            if name == "wall-axis-review.json" and not (FOCUS_REVIEW / name).is_file():
                continue
            review = saved_review_payload(name, saved_profile_sha256, digest)
            (job / name).write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
        options["saved_review"] = True
    result = await asyncio.to_thread(run_graph_pipeline, image, job / "classified.json",
                                     job / "prediction.json", job,
                                     parent_graph if parent_graph.is_file() else None, answers_path, scale, scope_path,
                                     **options)
    if result.returncode:
        (job / "error.log").write_text(result.stderr or result.stdout or "Unknown error", encoding="utf-8")
        if result.returncode == 3 and "GRAPH_REVIEW_ERROR:" in result.stdout:
            raise HTTPException(status_code=409, detail="Graph choice is stale or unsupported")
        if result.returncode == 4 and "GRAPH_BUILD_ERROR:" in result.stdout:
            raise HTTPException(status_code=422, detail="Не удалось построить граф. Проверьте масштаб и исключаемые области.")
        raise HTTPException(status_code=500, detail="Graph review processing failed")
    quality = json.loads((job / "quality.json").read_text(encoding="utf-8"))
    questions = json.loads((job / "graph_questions.json").read_text(encoding="utf-8"))
    graph = json.loads((job / "graph.json").read_text(encoding="utf-8"))
    allowed = quality.get("export_allowed") is True and quality.get("status") in ("good", "review")
    stats = load_stats(job / "project.json") if allowed else {
        "rooms": 0, "walls": 0, "connectors": 0, "doors": 0, "windows": 0}
    images = {"graph_questions": f"/api/jobs/{new_id}/image/graph_questions"}
    if allowed and (job / "graph_export.png").is_file():
        images["graph_export"] = f"/api/jobs/{new_id}/image/graph_export"
    return {"job_id": new_id, "parent_job_id": job_id, "pipeline": "graph-review",
            "quality": quality, "review_questions": questions["questions"],
            "image_sha256": questions["image_sha256"], "scale_cm_per_px": graph["scale_cm_per_px"],
            "questions_image_url": f"/api/jobs/{new_id}/image/graph_questions",
            "images": images,
            "original_url": f"/api/jobs/{new_id}/image/original", "stats": stats,
            "seconds": round(time.time() - started, 2), "warnings": quality.get("warnings", []),
            "quality_url": f"/api/jobs/{new_id}/quality",
            "download_url": f"/api/jobs/{new_id}/download" if allowed else None,
            "json_url": f"/api/jobs/{new_id}/json" if allowed else None}


def require_export_allowed(job_id: str):
    if not (JOBS_DIR / job_id).is_dir():
        raise HTTPException(status_code=404, detail="Job not found")
    path = JOBS_DIR / job_id / "quality.json"
    if not path.exists():
        raise HTTPException(status_code=409, detail="Нет отчёта проверки качества; обработайте изображение повторно.")
    quality = json.loads(path.read_text(encoding="utf-8"))
    if quality.get("export_allowed") is not True or quality.get("status") not in ("good", "review"):
        raise HTTPException(status_code=409, detail={
            "message": "Проект структурно некорректен; экспорт заблокирован.", "quality": quality,
        })


@app.get("/api/jobs/{job_id}/quality")
async def job_quality(job_id: str):
    validate_job_id(job_id)
    path = JOBS_DIR / job_id / "quality.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Quality report not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/opening-preflight")
async def job_opening_preflight(job_id: str):
    validate_job_id(job_id)
    path = JOBS_DIR / job_id / "opening_preflight.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Opening preflight report not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get(
    "/api/jobs/{job_id}/download"
)
async def download_project(
    job_id: str
):
    validate_job_id(
        job_id
    )
    require_export_allowed(job_id)

    path = (
        JOBS_DIR /
        job_id /
        "project.json"
    )

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Project not found"
        )

    return FileResponse(
        path,
        media_type="application/json",
        filename="project.json",
    )


@app.get(
    "/api/jobs/{job_id}/json"
)
async def project_json(
    job_id: str
):
    validate_job_id(
        job_id
    )
    require_export_allowed(job_id)

    path = (
        JOBS_DIR /
        job_id /
        "project.json"
    )

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Project not found"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


@app.get(
    "/api/jobs/{job_id}/image/{kind}"
)
async def job_image(
    job_id: str,
    kind: str,
):
    validate_job_id(
        job_id
    )

    job_dir = (
        JOBS_DIR /
        job_id
    )

    if kind == "original":
        matches = [
            p for p in job_dir.iterdir()
            if p.suffix.lower()
            in ALLOWED_EXTENSIONS
            and p.stem == job_id
        ]

        if not matches:
            raise HTTPException(
                status_code=404,
                detail="Original image not found"
            )

        return FileResponse(
            matches[0]
        )

    allowed = {
        "segmentation",
        "classified",
        "rooms",
        "barrier",
        "preprocessing",
        "normalized",
        "segmentation_original",
        "segmentation_normalized",
        "segmentation_fused",
        "closures",
        "export",
        "recovery",
        "topology_windows",
        "interior_windows",
        "wall_geometry",
        "exterior_enclosures",
        "exterior_trial",
        "graph_questions",
        "graph_export",
    }

    if kind not in allowed:
        raise HTTPException(
            status_code=404,
            detail="Image not found"
        )

    path = (
        job_dir /
        f"{kind}.png"
    )

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Image not found"
        )

    return FileResponse(
        path,
        media_type="image/png"
    )
