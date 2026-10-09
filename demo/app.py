import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from scripts.review_decisions import CHOICES, MAX_ANSWERS, make_review_questions


ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo"
JOBS_DIR = ROOT / "demo_jobs"
OUTPUT_DIR = ROOT / "output"

PIPELINE = ROOT / "scripts" / "process_floorplan.py"

TEMPLATE = ROOT / "test" / "template.json"
WINDOW_TEMPLATE = ROOT / "test" / "template_window.json"

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

MAX_FILE_SIZE = 20 * 1024 * 1024


app = FastAPI(
    title="Floorplan Recognition Demo",
    version="0.1.0",
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

    return HTMLResponse(
        index_path.read_text(
            encoding="utf-8"
        )
    )


@app.post("/api/recognize")
async def recognize(
    file: UploadFile = File(...),
    pipeline_v2: bool = Form(False),
    decisions: str | None = Form(None),
):
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
                or any(not isinstance(answer, dict)
                       or not isinstance(answer.get("opening_id"), int)
                       or answer.get("choice") not in CHOICES
                       or not isinstance(answer.get("fingerprint"), str)
                       for answer in answers)):
            raise HTTPException(status_code=400, detail="Review decisions do not match this image")
        applied_decisions = answers

    job_id = str(
        uuid.uuid4()
    )

    job_dir = (
        JOBS_DIR /
        job_id
    )

    job_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Имя изображения специально делаем уникальным:
    # pipeline использует stem для промежуточных файлов.
    image_name = (
        f"{job_id}{extension}"
    )

    image_path = (
        job_dir /
        image_name
    )

    image_path.write_bytes(
        data
    )

    decisions_path = None
    if review_payload is not None:
        decisions_path = job_dir / "review_decisions.json"
        decisions_path.write_text(json.dumps(review_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    project_path = (
        job_dir /
        "project.json"
    )

    started = time.time()

    result = await asyncio.to_thread(
        run_pipeline,
        image_path,
        project_path,
        pipeline_v2,
        decisions_path,
    )

    elapsed = (
        time.time() -
        started
    )

    quality_path = project_path.with_suffix(".quality.json")
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else None
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
    reviewed_dir = OUTPUT_DIR / f"{job_id}_reviewed"
    if pipeline_v2 and decisions_path is not None:
        room_path = reviewed_dir / f"{job_id}_rooms.json"
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
        classified_path = OUTPUT_DIR / f"{job_id}_interior_filtered_classified.json"
    if decisions_path is not None:
        classified_path = OUTPUT_DIR / f"{job_id}_reviewed_classified.json"
    classified_data = json.loads(classified_path.read_text(encoding="utf-8"))
    shutil.copy2(classified_path, job_dir / "classified.json")
    review_questions = make_review_questions(classified_data, quality, source_hash)
    binding_path = OUTPUT_DIR / f"{job_id}_opening_bindings.json"
    if binding_path.exists():
        shutil.copy2(binding_path, job_dir / "opening_bindings.json")
    geometry_path = OUTPUT_DIR / f"{job_id}_geometry.json"
    if geometry_path.exists():
        shutil.copy2(geometry_path, job_dir / "geometry.json")
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
    if pipeline_v2 and decisions_path is not None:
        generated_images.update({
            "rooms": reviewed_dir / f"{job_id}_rooms_overlay.png",
            "barrier": reviewed_dir / f"{job_id}_barrier.png",
        })

    warnings = quality["warnings"]
    inference = None
    if pipeline_v2:
        generated_images.update({
            "preprocessing": OUTPUT_DIR / f"{job_id}_preprocess_preview.png",
            "normalized": OUTPUT_DIR / f"{job_id}_preprocessed.png",
            "segmentation_original": OUTPUT_DIR / f"{job_id}_segmentation_original.png",
            "segmentation_normalized": OUTPUT_DIR / f"{job_id}_segmentation_normalized.png",
            "segmentation_fused": OUTPUT_DIR / f"{job_id}_segmentation_fused.png",
            "closures": OUTPUT_DIR / f"{job_id}_closures.png",
            "recovery": OUTPUT_DIR / f"{job_id}_opening_recovery_overlay.png",
            "topology_windows": OUTPUT_DIR / f"{job_id}_topology_windows_overlay.png",
        })
        if decisions_path is not None:
            generated_images["closures"] = reviewed_dir / f"{job_id}_closures.png"
        interior_counts = (json.loads(interior_report.read_text(encoding="utf-8"))
                           if interior_report.exists() else {})
        if interior_counts.get("rejected_windows", 0) or interior_counts.get("ambiguous_windows", 0):
            generated_images["interior_windows"] = (
                OUTPUT_DIR / f"{job_id}_interior_windows_overlay.png")
        for suffix in ("_preprocess.json", "_inference.json"):
            shutil.copy2(OUTPUT_DIR / f"{job_id}{suffix}", job_dir / suffix[1:])
        inference = json.loads((job_dir / "inference.json").read_text(encoding="utf-8"))
        for suffix in ("_opening_recovery.json", "_recovered_classified.json", "_classified.json",
                       "_topology_windows.json", "_topology_classified.json",
                       "_interior_filtered_classified.json", "_rooms_filtered.json"):
            source = OUTPUT_DIR / f"{job_id}{suffix}"
            if source.exists():
                shutil.copy2(source, job_dir / suffix[1:])

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
        "warnings": warnings,
        "quality": quality,
        "inference": inference,
        "review_questions": review_questions,
        "applied_decisions": applied_decisions,
        "image_sha256": source_hash,
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
        "download_url": (
            f"/api/jobs/{job_id}/download"
        ) if quality["export_allowed"] else None,
        "json_url": (
            f"/api/jobs/{job_id}/json"
        ) if quality["export_allowed"] else None,
    }


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
