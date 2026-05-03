import json
import os
import re
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from .analysis_engine import analyze_pdf_with_profile
from .profile_engine import list_profiles, load_profile, preview_profile, save_profile
from .sample_data import SAMPLE_ANALYSIS

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT / "frontend" / "dist"

RUNTIME_DIR = Path(os.environ.get("APP_RUNTIME_DIR", str(ROOT / "v4_runtime")))
PROFILES_DIR = RUNTIME_DIR / "profiles"
ANALYSES_DIR = RUNTIME_DIR / "analyses"
UPLOADS_DIR = RUNTIME_DIR / "uploads"
TEMP_DIR = RUNTIME_DIR / "tmp"

for folder in [PROFILES_DIR, ANALYSES_DIR, UPLOADS_DIR, TEMP_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Personalized Reading Assistant v4.2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_topics_json(text: str) -> list[str]:
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data if str(x).strip()]
    except Exception:
        pass
    return [item.strip() for item in text.split(",") if item.strip()]


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "uploaded.pdf")


async def save_target_pdf(upload: UploadFile) -> Path:
    if upload is None or not upload.filename:
        raise ValueError("Please upload a target PDF.")
    filename = safe_filename(upload.filename)
    path = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{filename}"
    content = await upload.read()
    path.write_bytes(content)
    return path


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "app": "personalized-reading-assistant-v4.2.1",
        "runtime_dir": str(RUNTIME_DIR),
    }


@app.get("/api/mock-analysis")
def mock_analysis():
    return SAMPLE_ANALYSIS


@app.get("/api/profiles")
def api_list_profiles():
    return {"profiles": list_profiles(PROFILES_DIR)}


@app.post("/api/profiles/preview")
async def api_preview_profile(
    profile_name: str = Form("Untitled profile"),
    selected_topics: str = Form("[]"),
    keywords: str = Form(""),
    research_statement: str = Form(""),
    seed_papers: Optional[list[UploadFile]] = File(default=None),
):
    topics = parse_topics_json(selected_topics)
    preview = await preview_profile(
        profile_name=profile_name,
        selected_topics=topics,
        keywords=keywords,
        research_statement=research_statement,
        seed_papers=seed_papers or [],
        temp_dir=TEMP_DIR / ("preview_" + uuid.uuid4().hex[:8]),
    )
    return preview


@app.post("/api/profiles/save")
async def api_save_profile(
    profile_name: str = Form("Untitled profile"),
    selected_topics: str = Form("[]"),
    keywords: str = Form(""),
    research_statement: str = Form(""),
    seed_papers: Optional[list[UploadFile]] = File(default=None),
):
    topics = parse_topics_json(selected_topics)
    try:
        result = await save_profile(
            profile_name=profile_name,
            selected_topics=topics,
            keywords=keywords,
            research_statement=research_statement,
            seed_papers=seed_papers or [],
            profiles_dir=PROFILES_DIR,
        )
        return result
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "message": f"Profile save failed: {exc}"},
        )


@app.post("/api/analyze")
async def api_analyze(
    profile_name: str = Form("AI Reading Assistant Profile"),
    density: str = Form("Standard"),
    profile_id: str = Form(""),
    paper: UploadFile | None = File(default=None),
):
    if not profile_id:
        mock = dict(SAMPLE_ANALYSIS)
        mock["profile_name"] = profile_name
        mock["summary"] = dict(mock["summary"])
        mock["summary"]["density"] = density
        return mock

    try:
        profile = load_profile(profile_id, PROFILES_DIR)
        pdf_path = await save_target_pdf(paper)
        result = analyze_pdf_with_profile(
            pdf_path=pdf_path,
            profile=profile,
            profiles_dir=PROFILES_DIR,
            analysis_dir=ANALYSES_DIR,
            density=density,
        )
        return result
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": f"Analyze failed: {exc}"},
        )


@app.get("/files/analyses/{analysis_id}/{filename}")
def get_analysis_file(analysis_id: str, filename: str):
    path = ANALYSES_DIR / analysis_id / safe_filename(filename)
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(path)


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_react(full_path: str):
        target = FRONTEND_DIST / full_path
        if target.exists() and target.is_file():
            return FileResponse(target)
        return FileResponse(FRONTEND_DIST / "index.html")
