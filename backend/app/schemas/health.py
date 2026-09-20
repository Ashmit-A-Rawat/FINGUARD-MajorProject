from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    engine_ready: bool = False
    engine_error: str | None = None
