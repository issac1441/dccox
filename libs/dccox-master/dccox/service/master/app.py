"""DC-Cox master — FastAPI application factory.

Single Responsibility: Wire together the repository, service, and routes.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from dccox.service.master.repository import ProjectRepository
from dccox.service.master.routes import create_router
from dccox.service.master.service import ProjectService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    logger.info("DC-Cox master starting up")
    yield
    logger.info("DC-Cox master shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="DC-Cox Master",
        description="Federated Cox PH Regression — Master Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    repository = ProjectRepository()
    service = ProjectService(repository)
    router = create_router(service)
    application.include_router(router)

    return application


# Module-level app for `uvicorn dccox.service.master.app:app`
app = create_app()

if __name__ == "__main__":
    import uvicorn

    from .config import settings

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "dccox.service.master.app:app",
        host="0.0.0.0",
        port=settings.master_port,
        reload=False,
    )
