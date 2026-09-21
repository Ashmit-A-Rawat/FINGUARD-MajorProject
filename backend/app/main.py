"""FIN-GUARD backend entrypoint."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import auth, cases, health
from backend.app.core.config import Settings, get_settings
from backend.app.core.security import LoginLimiter, resolve_secret
from backend.app.database.base import Base, make_engine, make_session_factory
from backend.app.services.engine import EngineService

logger = logging.getLogger(__name__)


def _build_engine_context(settings: Settings):  # type: ignore[no-untyped-def]
    """Heavy imports live here so that the API process starts (and tests import) quickly."""
    from agents.context import build_context
    from knowledge_base.embeddings.embedder import SentenceTransformerEmbedder
    from llm.inference.factory import create_provider

    return build_context(
        Path(settings.dataset_dir),
        SentenceTransformerEmbedder(),
        create_provider(settings),
        semantic_tripwire=settings.semantic_tripwire,
    )


def create_app(settings: Settings | None = None, engine: EngineService | None = None) -> FastAPI:
    settings = settings or get_settings()
    secret = resolve_secret(settings.api_secret_key, settings.app_env)
    db_engine = make_engine(settings.database_url)
    session_factory = make_session_factory(db_engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        Base.metadata.create_all(db_engine)
        service = app.state.engine
        if service is None and settings.engine_enabled:
            service = app.state.engine = EngineService()
            service.start_loading(lambda: _build_engine_context(settings))
        yield
        if service is not None:
            service.shutdown()
        db_engine.dispose()

    app = FastAPI(
        title="FIN-GUARD",
        description="Research prototype. Operates on SYNTHETIC data only. Advisory output only.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.secret = secret
    app.state.session_factory = session_factory
    app.state.engine = engine
    app.state.limiter = LoginLimiter()
    Base.metadata.create_all(db_engine)  # also for clients that never run the lifespan (tests)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"  # case data must not be cached
        return response

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(cases.router)
    return app


app = create_app()
