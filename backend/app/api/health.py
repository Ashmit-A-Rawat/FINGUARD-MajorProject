from typing import Annotated

from fastapi import APIRouter, Depends, Request

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])

APP_VERSION = "0.2.0"


@router.get("/health", response_model=HealthResponse)
def health(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> HealthResponse:
    engine = getattr(request.app.state, "engine", None)
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=APP_VERSION,
        environment=settings.app_env,
        engine_ready=bool(engine and engine.ready),
        engine_error=engine.error if engine else None,
    )
