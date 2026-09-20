"""Application settings, loaded from environment variables / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "fin-guard"
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173"
    data_dir: str = "./data"
    synthetic_seed: int = 42
    llm_provider: str = "mock"  # mock | qwen | mistral
    llm_model: str = ""  # empty = the provider's default model
    llm_device: str = "auto"  # auto | cpu | mps | cuda
    llm_allow_download: bool = False  # never download model weights unless explicitly enabled


@lru_cache
def get_settings() -> Settings:
    return Settings()
