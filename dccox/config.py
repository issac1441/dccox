"""Configuration for dccox."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings for dccox."""

    model_config = {"env_file": ".env", "extra": "ignore"}

    enable_validation: bool = Field(default=False, description="Enable validation")


settings = Settings()
