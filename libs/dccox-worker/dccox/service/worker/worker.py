"""DC-Cox federated worker — facade coordinating client, pipeline, and results.

Single Responsibility: High-level API that delegates to specialised components.
"""

from __future__ import annotations

import logging

import pandas as pd

from dccox.cox import SurvivalFunction
from dccox.service.worker.client import MasterClient
from dccox.service.worker.pipeline import LocalPipeline
from dccox.service.worker.result_formatter import build_result_tables
from dccox.usecase import Horizontal

logger = logging.getLogger(__name__)


class DCCoxWorker:
    """Worker facade for the DC-Cox federated analysis protocol.

    Wraps HTTP calls to the master and local ``Horizontal`` operations.

    Parameters
    ----------
    master_url : str
        Base URL of the master service (e.g. ``http://localhost:8000``).
    """

    def __init__(self, master_url: str) -> None:
        self._client = MasterClient(master_url)
        self._pipeline = LocalPipeline(self._client, Horizontal())
        self._results: dict[str, SurvivalFunction] = {}
        self._feature_names: dict[str, list[str]] = {}
        self._worker_id: str | None = None

    @property
    def master_url(self) -> str:
        """Return the master service base URL."""
        return self._client.base_url

    @property
    def worker_id(self) -> str | None:
        """Return the worker ID assigned after joining a project."""
        return self._worker_id

    # ── Survival function cache ─────────────────────────────────────────

    def get_survival_function(self, project_id: str) -> SurvivalFunction | None:
        """Return the cached survival function for a project."""
        return self._results.get(project_id)

    def get_feature_names(self, project_id: str) -> list[str] | None:
        """Return the ordered feature names used for predictions."""
        names = self._feature_names.get(project_id)
        if names is None:
            surv = self._results.get(project_id)
            if surv is not None:
                names = [str(idx) for idx in surv.summary.index]
                self._feature_names[project_id] = names
        return names

    # ── Project Management (delegated to client) ────────────────────────

    def create_project(self, config: dict) -> str:
        """Create a project on the master."""
        return self._client.create_project(config)

    def list_projects(self) -> list[dict]:
        """List all projects on the master."""
        return self._client.list_projects()

    def get_project(self, project_id: str) -> dict:
        """Get project details from the master."""
        return self._client.get_project(project_id)

    def join_project(self, project_id: str, worker_name: str, n_features: int) -> str:
        """Join a project and store the assigned worker ID."""
        self._worker_id = self._client.join_project(project_id, worker_name, n_features)
        return self._worker_id

    def lock_project(self, project_id: str) -> None:
        """Lock the project on the master."""
        self._client.lock_project(project_id)

    def start_project(self, project_id: str) -> None:
        """Start the analysis on the master."""
        self._client.start_project(project_id)

    def get_events(self, project_id: str) -> list[dict[str, str]]:
        """Fetch the event log from the master."""
        return self._client.get_events(project_id)

    # ── Pipeline ────────────────────────────────────────────────────────

    def run_local_pipeline(
        self,
        project_id: str,
        data_path: str,
        *,
        poll_interval: float = 2.0,
        poll_timeout: float | None = 600.0,
    ) -> SurvivalFunction:
        """Run the full worker-side pipeline and cache the result."""
        if self._worker_id is None:
            msg = "Must join a project first"
            raise RuntimeError(msg)

        config = self.get_project(project_id)["config"]

        surv_func, feature_names = self._pipeline.run(
            project_id,
            self._worker_id,
            data_path,
            config,
            poll_interval=poll_interval,
            poll_timeout=poll_timeout,
        )

        self._results[project_id] = surv_func
        self._feature_names[project_id] = feature_names
        logger.info("Results stored for project %s", project_id)
        return surv_func

    # ── Results ─────────────────────────────────────────────────────────

    def get_worker_results(self, project_id: str) -> dict[str, str] | pd.DataFrame:
        """Return only the coefficients summary for compatibility."""
        tables = self.get_result_tables(project_id)
        if isinstance(tables, dict) and "error" in tables:
            return tables
        return tables["summary"]

    def get_result_tables(
        self, project_id: str
    ) -> dict[str, pd.DataFrame] | dict[str, str]:
        """Return formatted summary and baseline tables for a project."""
        surv = self._results.get(project_id)
        if surv is None:
            return {"error": f"No results found for project {project_id}"}
        return build_result_tables(surv)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._client.close()

    def __del__(self) -> None:
        """Close the HTTP session on garbage collection."""
        import contextlib

        with contextlib.suppress(Exception):
            self.close()
