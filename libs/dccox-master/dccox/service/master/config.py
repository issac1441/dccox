"""Configuration for dccox-master."""

from pydantic import Field
from pydantic_settings import BaseSettings


class MasterSettings(BaseSettings):
    """Settings for dccox-master."""

    model_config = {"env_file": ".env", "env_prefix": "DCCOX_", "extra": "ignore"}

    master_port: int = Field(default=8000, description="Master service port")


settings = MasterSettings()
