"""Event handler functions for the DC-Cox worker UI.

Single Responsibility: Business logic for Gradio callbacks. Each handler
validates inputs, delegates to the worker/service layer, and returns
Gradio-compatible output tuples.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from dccox.service.worker.result_formatter import build_result_tables
from dccox.service.worker.ui.helpers import (
    build_coef_plot,
    create_temp_csv,
    create_temp_file,
    default_prediction_df,
    format_prediction_output,
    history_empty_outputs,
    prepare_feature_df,
    workspace_empty_outputs,
)
from dccox.service.worker.ui.prediction import PREDICTION_METHOD_MAP
from dccox.service.worker.worker import DCCoxWorker

logger = logging.getLogger(__name__)


# ── Connection ──────────────────────────────────────────────────────────


def handle_connect(url: str) -> tuple[Any, str]:
    """Connect to the master service."""
    try:
        worker = DCCoxWorker(url)
        worker.list_projects()  # smoke test
        return worker, f"**Connected** to `{url}` successfully!"
    except Exception as e:
        return None, f"**Connection failed**: {e}"


# ── Project Creation ────────────────────────────────────────────────────


def handle_create_project(
    worker: DCCoxWorker | None,
    name: str,
    uc: str,
    k: float,
    r: float,
    bs_p: float,
    bs_t: float,
    bs_r: bool,
    alpha: float,
    step: float,
    var_t: float,
    cent: str,
    keep_f: str,
    meta_c: str,
) -> str:
    """Create a new analysis project on the master."""
    if worker is None:
        return "Connect to master first (Dashboard tab)."
    if not name.strip():
        return "Enter a project name."

    feat_list = [x.strip() for x in keep_f.split(",") if x.strip()]
    meta_list = [x.strip() for x in meta_c.split(",") if x.strip()]
    centering_val = "mean" if cent == "mean" else None

    config = {
        "name": name.strip(),
        "use_case": uc,
        "k": int(k),
        "r": int(r),
        "bs_prop": float(bs_p),
        "bs_times": int(bs_t),
        "bs_replace": bs_r,
        "alpha": float(alpha),
        "step_size": float(step),
        "var_thres": float(var_t),
        "centering": centering_val,
        "keep_feature_cols": feat_list if feat_list else None,
        "meta_cols": meta_list if meta_list else None,
    }
    try:
        pid = worker.create_project(config)
        return f"**Project created successfully!** ID: `{pid}`"
    except Exception as e:
        return f"**Failed to create**: {e}"


# ── Project List ────────────────────────────────────────────────────────


def handle_refresh(worker: DCCoxWorker | None) -> list[list[str]]:
    """Refresh the projects table."""
    if worker is None:
        return []
    try:
        projects = worker.list_projects()
        return [
            [
                p["id"],
                p["config"]["name"],
                p["status"].upper(),
                str(len(p["workers"])),
                p["created_at"],
            ]
            for p in projects
        ]
    except Exception:
        return []


# ── Workspace: Join / Lock / Start ──────────────────────────────────────


def handle_join(
    worker: DCCoxWorker | None, pid: str, cname: str, dpath: str
) -> tuple[str, str]:
    """Join a project as a worker."""
    if worker is None:
        return "", "Connect first."
    if not pid.strip():
        return "", "Enter project ID."
    if not dpath.strip():
        return "", "Enter data path to determine n_features."

    try:
        project = worker.get_project(pid.strip())
        config = project["config"]
        keep_feature_cols = config.get("keep_feature_cols")
        meta_cols = config.get("meta_cols") or []

        if keep_feature_cols:
            n_features = len(keep_feature_cols)
        else:
            tmp_df = pd.read_csv(dpath)
            drop_cols = ["time", "event", *meta_cols]
            n_features = len([c for c in tmp_df.columns if c not in drop_cols])

        wid = worker.join_project(pid.strip(), cname.strip(), n_features)
        return (
            pid.strip(),
            f"Joined as `{wid}` (n_features={n_features}). Waiting for lock.",
        )
    except Exception as e:
        return "", f"Join failed: {e}"


def handle_lock(worker: DCCoxWorker | None, pid: str | None) -> str:
    """Lock the project to prevent more workers from joining."""
    if worker is None or not pid:
        return "Join a project first."
    try:
        worker.lock_project(pid)
        return "Project locked! New workers can no longer join."
    except Exception as e:
        return f"Lock failed: {e}"


def handle_start(worker: DCCoxWorker | None, pid: str | None) -> str:
    """Start the analysis (generates Xanc)."""
    if worker is None or not pid:
        return "Join a project first."
    try:
        worker.start_project(pid)
        return "Analysis started! Xanc generated by server."
    except Exception as e:
        return f"Start failed: {e}"


# ── Workspace: Run Pipeline ─────────────────────────────────────────────


def handle_run(
    worker: DCCoxWorker | None,
    pid: str | None,
    dpath: str,
    current_result_pid: str | None,
    feature_headers: list[str],
) -> tuple[Any, ...]:
    """Run the full local compute + global aggregation pipeline."""

    def _empty(msg: str) -> tuple[Any, ...]:
        return workspace_empty_outputs(msg, current_result_pid, feature_headers)

    if worker is None or not pid:
        return _empty("Join a project first.")
    if not dpath.strip():
        return _empty("Enter data path.")

    try:
        pid_clean = pid.strip()
        surv = worker.run_local_pipeline(pid_clean, dpath.strip(), poll_interval=2.0)
        tables = build_result_tables(surv)
        feature_names = worker.get_feature_names(pid_clean) or []
        prediction_df = default_prediction_df(feature_names)
        workspace_fig = build_coef_plot(tables["summary"])
        history_fig = build_coef_plot(tables["summary"])
        return (
            "Local compute & global aggregation completed!",
            tables["summary"],
            tables["baseline_cumhazards"],
            tables["baseline_survival"],
            tables["baseline_hazard"],
            pid_clean,
            feature_names,
            prediction_df,
            "",
            None,
            tables["baseline_cumhazards"],
            tables["baseline_survival"],
            tables["baseline_hazard"],
            prediction_df,
            "",
            None,
            workspace_fig,
            history_fig,
        )
    except Exception as e:
        return _empty(f"Analysis failed: {e}")


# ── Prediction ──────────────────────────────────────────────────────────


def handle_prediction(
    worker: DCCoxWorker | None,
    result_pid: str | None,
    method: str,
    t_value: float | None,
    feature_headers: list[str],
    feature_values: pd.DataFrame | None,
) -> tuple[str, Any]:
    """Execute a prediction method on the cached survival function."""
    try:
        formatted = compute_prediction(
            worker, result_pid, method, t_value, feature_headers, feature_values
        )
        return f"{method} computed.", formatted
    except ValueError as err:
        return str(err), None
    except Exception as err:
        return f"Prediction failed: {err}", None


def compute_prediction(
    worker: DCCoxWorker | None,
    result_pid: str | None,
    method: str,
    t_value: float | None,
    feature_headers: list[str],
    feature_values: pd.DataFrame | None,
) -> pd.DataFrame:
    """Core prediction logic shared by workspace and history tabs."""
    if worker is None:
        raise ValueError("Connect to master first.")
    if result_pid is None:
        raise ValueError("Run analysis or load results first.")

    surv = worker.get_survival_function(result_pid)
    if surv is None:
        raise ValueError(f"No cached results for project {result_pid}")

    spec = PREDICTION_METHOD_MAP.get(method)
    if spec is None:
        raise ValueError(f"Unknown method '{method}'")

    args: list[Any] = []
    if spec.requires_time:
        if t_value is None:
            raise ValueError("Provide a time value for this method.")
        args.append(float(t_value))
    if spec.requires_features:
        names = feature_headers or worker.get_feature_names(result_pid) or []
        X = prepare_feature_df(feature_values, names)
        args.append(X)

    prediction = getattr(surv, method)(*args)
    return format_prediction_output(prediction, method)


# ── Downloads ───────────────────────────────────────────────────────────


def download_table_csv(
    worker: DCCoxWorker | None, pid: str | None, table_key: str
) -> str:
    """Export a result table as a CSV temp file."""
    if worker is None:
        raise gr.Error("Connect to master first.")
    if not pid:
        raise gr.Error("Run analysis or load results first.")
    pid_clean = pid.strip()
    if not pid_clean:
        raise gr.Error("Run analysis or load results first.")
    tables = worker.get_result_tables(pid_clean)
    if isinstance(tables, dict) and "error" in tables:
        raise gr.Error(tables["error"])
    table = tables.get(table_key)
    if not isinstance(table, pd.DataFrame):
        raise gr.Error("Table unavailable. Run analysis first.")
    return create_temp_csv(table.to_csv(index=False), table_key)


def download_plot_path(worker: DCCoxWorker | None, pid: str | None, label: str) -> str:
    """Export a coefficient plot as a PNG temp file."""
    data = _download_plot_bytes(worker, pid)
    return create_temp_file(data, prefix=label, suffix=".png", binary=True)


def _download_plot_bytes(worker: DCCoxWorker | None, pid: str | None) -> bytes:
    """Generate the coefficient plot as PNG bytes."""
    if worker is None:
        raise gr.Error("Connect to master first.")
    if not pid:
        raise gr.Error("Run analysis or load results first.")
    pid_clean = pid.strip()
    if not pid_clean:
        raise gr.Error("Run analysis or load results first.")
    tables = worker.get_result_tables(pid_clean)
    if isinstance(tables, dict) and "error" in tables:
        raise gr.Error(tables["error"])
    summary = tables.get("summary")
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        raise gr.Error("Summary unavailable. Run analysis first.")
    fig = build_coef_plot(summary)
    if fig is None:
        raise gr.Error("Unable to generate coefficient plot.")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ── Events ──────────────────────────────────────────────────────────────


def handle_poll_events(worker: DCCoxWorker | None, pid: str | None) -> str:
    """Poll and format the event log."""
    if worker is None or not pid:
        return "Awaiting project connection..."
    try:
        events = worker.get_events(pid)
        lines = [f"[{e['time'][:19]}] {e['message']}" for e in events]
        return "\n".join(lines)
    except Exception as exc:
        logger.exception("Failed to poll events for %s: %s", pid, exc)
        return "Polling events..."


# ── History ─────────────────────────────────────────────────────────────


def handle_fetch_history(
    worker: DCCoxWorker | None,
    pid: str,
    current_result_pid: str | None,
    feature_headers: list[str],
) -> tuple[Any, ...]:
    """Fetch cached results for a previously-run project."""

    def _error(msg: str) -> tuple[Any, ...]:
        return history_empty_outputs(msg, current_result_pid, feature_headers)

    if worker is None:
        return _error("Connect to master first")
    if worker.worker_id is None:
        return _error("Join a project first to fetch your results")
    if not pid.strip():
        return _error("Project ID is required")

    try:
        pid_clean = pid.strip()
        surv = worker.get_survival_function(pid_clean)
        if surv is None:
            return _error(f"No cached results for `{pid_clean}`")
        tables = build_result_tables(surv)
        feature_names = worker.get_feature_names(pid_clean) or []
        prediction_df = default_prediction_df(feature_names)
        status_msg = f"Loaded stored results for `{pid_clean}`."
        workspace_fig = build_coef_plot(tables["summary"])
        history_fig = build_coef_plot(tables["summary"])
        return (
            status_msg,
            tables["summary"],
            status_msg,
            tables["summary"],
            tables["baseline_cumhazards"],
            tables["baseline_survival"],
            tables["baseline_hazard"],
            pid_clean,
            feature_names,
            prediction_df,
            "",
            None,
            tables["baseline_cumhazards"],
            tables["baseline_survival"],
            tables["baseline_hazard"],
            prediction_df,
            "",
            None,
            workspace_fig,
            history_fig,
        )
    except Exception as e:
        return _error(f"Fetch failed: {e}")
