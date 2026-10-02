import asyncio
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
):
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
    room_data = json.loads(room_path.read_text(encoding="utf-8"))
    metrics = quality.get("metrics", {})
    stats = load_stats(project_path) if not rejected else {
        "rooms": metrics.get("exported_rooms", len(room_data.get("rooms", []))),
        "walls": metrics.get("exported_segments", 0),
        "connectors": metrics.get("exported_connectors", 0),
        "doors": metrics.get("exported_doors", len(room_data.get("doors", []))),
        "windows": metrics.get("exported_windows", len(room_data.get("windows", []))),
    }
    shutil.copy2(quality_path, job_dir / "quality.json")
    shutil.copy2(room_path, job_dir / "rooms.json")
    binding_path = OUTPUT_DIR / f"{job_id}_opening_bindings.json"
    if binding_path.exists():
        shutil.copy2(binding_path, job_dir / "opening_bindings.json")

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
    }

    warnings = quality["warnings"]
    if pipeline_v2:
        generated_images.update({
            "preprocessing": OUTPUT_DIR / f"{job_id}_preprocess_preview.png",
            "normalized": OUTPUT_DIR / f"{job_id}_preprocessed.png",
            "closures": OUTPUT_DIR / f"{job_id}_closures.png",
        })
        for suffix in ("_preprocess.json",):
            shutil.copy2(OUTPUT_DIR / f"{job_id}{suffix}", job_dir / suffix[1:])

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
        "closures",
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
