"""FastAPI routes: POST /remove-background (fast rembg) and optional POST /try-on."""

import io
import logging
import os
import re
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.backend.model_loader import PROJECT_ROOT, ModelBundle
from pipeline.full_pipeline import run_remove_hair_only, run_try_on
from utils.preprocessing import ensure_bgr_uint8
from utils.remove_background import downscale_max_side, remove_background_png_bytes

LOGGER = logging.getLogger(__name__)

router = APIRouter()
HAIRSTYLES_DIR = PROJECT_ROOT / "data" / "hairstyles"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

_bundle: ModelBundle | None = None


def set_model_bundle(bundle: ModelBundle | None) -> None:
    """Called from FastAPI startup after load_models()."""
    global _bundle
    _bundle = bundle


def _remove_bg_only_mode() -> bool:
    return os.environ.get("SKIP_HAIR_PIPELINE", "1").strip().lower() in ("1", "true", "yes", "on")


@router.get("/health")
def health():
    """
    JSON status for the SPA: backend up, models loaded, device.
    remove-bg-only mode: ok=True, models_loaded=False, mode=remove_bg_only
    """
    if _bundle is None:
        if _remove_bg_only_mode():
            return {
                "ok": True,
                "status": "ready",
                "models_loaded": False,
                "hair_removal_available": False,
                "mode": "remove_bg_only",
                "device": None,
                "message": "rembg only — POST /remove-background (fast). Set SKIP_HAIR_PIPELINE=0 for hair removal & try-on.",
            }
        return {
            "ok": False,
            "status": "unavailable",
            "models_loaded": False,
            "hair_removal_available": False,
            "device": None,
            "message": "Model bundle not initialized (startup may still be loading or failed).",
        }
    return {
        "ok": True,
        "status": "ready",
        "models_loaded": True,
        "hair_removal_available": True,
        "mode": "full",
        "device": str(_bundle.device),
        "message": "All models loaded; try-on API is ready.",
    }


@router.post("/remove-background")
async def remove_background_endpoint(
    image: UploadFile = File(..., description="JPEG/PNG — returns PNG with transparent background"),
    max_side: int = Form(1280, description="Downscale longest edge before rembg (speed); 0 = no resize"),
):
    """Fast path: rembg only. No face / hair / LaMa models required."""
    try:
        raw = await image.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image — use JPEG or PNG.")
        bgr = ensure_bgr_uint8(bgr)
        try:
            ms = int(max_side)
        except (TypeError, ValueError):
            ms = 1280
        if ms and ms > 0:
            bgr = downscale_max_side(bgr, max_side=ms)
        png_bytes = remove_background_png_bytes(bgr)
        return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception("remove-background failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/remove-hair")
async def remove_hair_endpoint(
    image: UploadFile = File(..., description="Portrait JPEG/PNG — returns PNG with hair inpainted away"),
    max_side: int = Form(1280, description="Downscale longest edge before inference (0 = original size)"),
    remove_bg: str = Form("true", description="rembg then composite on white before hair removal (matches try-on prep)"),
):
    """BiSeNet + LaMa — requires ModelBundle (set SKIP_HAIR_PIPELINE=0 and restart)."""
    if _bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Hair removal needs loaded models. Set SKIP_HAIR_PIPELINE=0 and restart the server.",
        )
    bundle = _bundle
    try:
        raw = await image.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image — use JPEG or PNG.")
        bgr = ensure_bgr_uint8(bgr)
        try:
            ms = int(max_side)
        except (TypeError, ValueError):
            ms = 1280
        if ms and ms > 0:
            bgr = downscale_max_side(bgr, max_side=ms)
        do_bg = _parse_bool_form(remove_bg, True)
        out_bgr = run_remove_hair_only(bgr, bundle, remove_background=do_bg)
        ok, buf = cv2.imencode(".png", out_bgr)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode result PNG.")
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")
    except HTTPException:
        raise
    except ValueError as e:
        LOGGER.warning("Remove-hair validation: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        LOGGER.exception("remove-hair failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _safe_hairstyle_filename(name: str) -> str:
    """Basename only; must be *.png under data/hairstyles (no path traversal)."""
    base = Path(name).name
    if not base or base != name.strip():
        raise ValueError("Invalid hairstyle name.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.png", base):
        raise ValueError("Hairstyle must be a simple .png filename (letters, numbers, _ . -).")
    return base


def _hairstyle_label(filename: str) -> str:
    stem = Path(filename).stem
    return stem.replace("_", " ").strip().title()


@router.get("/hairstyles")
def list_hairstyles():
    """PNG filenames in data/hairstyles for the UI dropdown."""
    if not HAIRSTYLES_DIR.is_dir():
        return {"hairstyles": []}
    items = []
    for p in sorted(HAIRSTYLES_DIR.glob("*.png")):
        items.append({"filename": p.name, "label": _hairstyle_label(p.name)})
    return {"hairstyles": items}


@router.get("/hairstyles/preview/{filename}")
def hairstyle_preview(filename: str):
    """Serve a hairstyle PNG for the UI grid (same files as data/hairstyles/)."""
    try:
        safe = _safe_hairstyle_filename(filename)
    except ValueError:
        raise HTTPException(status_code=404, detail="Invalid filename") from None
    path = HAIRSTYLES_DIR / safe
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Hairstyle not found")
    return FileResponse(path, media_type="image/png")


def _clamp_hair_scale(v: float) -> float:
    """UI slider range; landmarks already set base fit."""
    return float(max(0.75, min(1.35, v)))


def _parse_bool_form(v: str | bool | None, default: bool = True) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("false", "0", "no", "off"):
        return False
    if s in ("true", "1", "yes", "on"):
        return True
    return default


@router.post("/try-on")
async def try_on(
    image: UploadFile = File(..., description="User portrait (JPEG/PNG)"),
    hairstyle: str = Form("beatrice_curly.png", description="Hairstyle PNG in data/hairstyles/ (default: Beatrice from curly pack)"),
    hair_scale: float = Form(1.0, description="Extra scale vs face fit (0.75–1.35, default 1.0)"),
    remove_bg: str = Form("true", description="Remove background with rembg before try-on (true/false)"),
):
    """
    Accept multipart upload: `image` file + form field `hairstyle` (filename).
    Returns PNG image bytes of the composited result.
    """
    if _bundle is None:
        detail = (
            "Full try-on is disabled (SKIP_HAIR_PIPELINE=1). Use POST /remove-background or set SKIP_HAIR_PIPELINE=0 and restart."
            if _remove_bg_only_mode()
            else "Models not loaded yet."
        )
        raise HTTPException(status_code=503, detail=detail)
    bundle = _bundle
    try:
        raw = await image.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image — use JPEG or PNG.")
        bgr = ensure_bgr_uint8(bgr)
        hair_file = _safe_hairstyle_filename(hairstyle)
        if not (HAIRSTYLES_DIR / hair_file).is_file():
            raise HTTPException(status_code=404, detail=f"Hairstyle not found: {hair_file}")
        try:
            hs = _clamp_hair_scale(float(hair_scale))
        except (TypeError, ValueError):
            hs = 1.0
        do_bg = _parse_bool_form(remove_bg, True)
        final_bgr, saved_path = run_try_on(
            bgr,
            hair_file,
            bundle,
            HAIRSTYLES_DIR,
            save_result=True,
            results_dir=RESULTS_DIR,
            hair_scale=hs,
            remove_background=do_bg,
        )
        ok, buf = cv2.imencode(".png", final_bgr)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode result PNG.")
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")
    except ValueError as e:
        LOGGER.warning("Try-on validation: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        LOGGER.exception("Try-on failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
