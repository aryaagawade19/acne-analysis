from __future__ import annotations

import logging

from fastapi import APIRouter, File, Header, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.services.analysis_engine import AnalysisError, analyze_skin_image
from app.utils.image_utils import ImageValidationError, decode_image, maybe_save_debug_image, validate_image_upload


router = APIRouter(tags=["analysis"])
logger = logging.getLogger("skin_ai")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


def _verify_api_key(x_ai_api_key: str | None) -> JSONResponse | None:
    if settings.ai_api_key and x_ai_api_key != settings.ai_api_key:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return None


@router.post("/analyze")
@router.post("/v1/skin-analyze")
async def analyze_skin(
    file: UploadFile = File(...),
    x_ai_api_key: str | None = Header(default=None),
):
    auth_error = _verify_api_key(x_ai_api_key)
    if auth_error:
        return auth_error

    try:
        payload = await file.read()
        validate_image_upload(file.content_type, payload, settings.max_upload_mb)
        image = decode_image(payload)

        result, skin = analyze_skin_image(image)

        if settings.debug_save_skin:
            result.update(maybe_save_debug_image(skin, settings.debug_output_path))

        return result

    except ImageValidationError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except AnalysisError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        logger.exception("Unexpected analysis failure")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
