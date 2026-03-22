"""
FastAPI application: serves API + static frontend.
Default: rembg-only (no heavy models). Set SKIP_HAIR_PIPELINE=0 for full try-on.
"""

import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path (for pipeline, models, utils).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.backend import routes
from app.backend.model_loader import PROJECT_ROOT as ML_ROOT, load_models
from utils.helpers import setup_logging

LOGGER = logging.getLogger(__name__)

_SKIP_HAIR = os.environ.get("SKIP_HAIR_PIPELINE", "1").strip().lower() in ("1", "true", "yes", "on")

PROJECT_ROOT = ML_ROOT
FRONTEND_DIR = PROJECT_ROOT / "app" / "frontend"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

app = FastAPI(title="AI Hairstyle Try-On System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    setup_logging(LOG_DIR)
    if _SKIP_HAIR:
        routes.set_model_bundle(None)
        LOGGER.info("SKIP_HAIR_PIPELINE=1 — rembg-only; POST /remove-background. Set SKIP_HAIR_PIPELINE=0 for /try-on.")
    else:
        bundle = load_models()
        routes.set_model_bundle(bundle)
        LOGGER.info("Models loaded; API ready.")


app.include_router(routes.router, tags=["try-on"])


@app.get("/")
async def serve_index():
    """Serve the upload UI (HTML)."""
    index = FRONTEND_DIR / "index.html"
    if not index.is_file():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Frontend not found. Place index.html in app/frontend/.")
    return FileResponse(index)


# CSS/JS live under app/frontend/
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
