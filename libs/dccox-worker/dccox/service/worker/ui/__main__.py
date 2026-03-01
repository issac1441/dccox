"""Allow running the worker UI as ``python -m dccox.service.worker.ui``."""

import logging

import uvicorn

from dccox.service.worker.config import settings

logging.basicConfig(level=logging.INFO)
uvicorn.run(
    "dccox.service.worker.ui:app",
    host="0.0.0.0",
    port=settings.worker_port,
    reload=False,
)
