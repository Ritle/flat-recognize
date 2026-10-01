import asyncio
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
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

    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
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
    file: UploadFile = File(...)
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
    )

    elapsed = (
        time.time() -
        started
    )

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

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Ошибка обработки планировки"
                ),
                "log": error_text[-4000:],
            }
        )

    if not project_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Pipeline did not create project.json"
        )

    stats = load_stats(
        project_path
    )

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
        "ok": True,
        **metadata,
        "original_url": (
            f"/api/jobs/{job_id}/image/original"
        ),
        "download_url": (
            f"/api/jobs/{job_id}/download"
        ),
        "json_url": (
            f"/api/jobs/{job_id}/json"
        ),
    }


@app.get(
    "/api/jobs/{job_id}/download"
)
async def download_project(
    job_id: str
):
    validate_job_id(
        job_id
    )

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
