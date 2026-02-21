"""Configuration for dccox-worker."""

from pydantic import Field
from pydantic_settings import BaseSettings


class WorkerSettings(BaseSettings):
    """Settings for dccox-worker."""

    model_config = {"env_file": ".env", "env_prefix": "DCCOX_", "extra": "ignore"}

    worker_port: int = Field(default=8001, description="Worker service port")
    master_url: str = Field(
        default="http://localhost:8000", description="Master service URL"
    )
    worker_name: str = Field(default="my-worker", description="Default worker name")


settings = WorkerSettings()
