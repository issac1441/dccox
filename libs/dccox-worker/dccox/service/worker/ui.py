"""Gradio frontend for DC-Cox worker operations."""

from __future__ import annotations

import io
import logging
import tempfile
from typing import Any

from fastapi import FastAPI
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dccox.service.master.schemas import ProjectSchema
from dccox.service.worker.config import settings
from dccox.service.worker.worker import DCCoxWorker, build_result_tables

logger = logging.getLogger(__name__)


PREDICTION_METHOD_ORDER = [
    "hazard_at",
    "cumhazard_at",
    "survival_at",
    "predict_log_partial_hazard",
    "predict_partial_hazard",
    "predict_hazard_at",
    "predict_cumhazard_at",
    "predict_survival_at",
    "predict_hazard",
    "predict_cumhazard",
    "predict_survival",
    "predict_expectation",
]

PREDICTION_METHOD_CONFIG = {
    "hazard_at": {
        "label": "hazard_at (baseline hazard at t)",
        "requires_time": True,
        "requires_features": False,
    },
    "cumhazard_at": {
        "label": "cumhazard_at (baseline cumulative hazard at t)",
        "requires_time": True,
        "requires_features": False,
    },
    "survival_at": {
        "label": "survival_at (baseline survival at t)",
        "requires_time": True,
        "requires_features": False,
    },
    "predict_log_partial_hazard": {
        "label": "predict_log_partial_hazard (log hazard, features only)",
        "requires_time": False,
        "requires_features": True,
    },
    "predict_partial_hazard": {
        "label": "predict_partial_hazard (hazard ratio, features only)",
        "requires_time": False,
        "requires_features": True,
    },
    "predict_hazard_at": {
        "label": "predict_hazard_at (hazard at t)",
        "requires_time": True,
        "requires_features": True,
    },
    "predict_cumhazard_at": {
        "label": "predict_cumhazard_at (cumulative hazard at t)",
        "requires_time": True,
        "requires_features": True,
    },
    "predict_survival_at": {
        "label": "predict_survival_at (survival probability at t)",
        "requires_time": True,
        "requires_features": True,
    },
    "predict_hazard": {
        "label": "predict_hazard (hazard over all times)",
        "requires_time": False,
        "requires_features": True,
    },
    "predict_cumhazard": {
        "label": "predict_cumhazard (cumulative hazard over all times)",
        "requires_time": False,
        "requires_features": True,
    },
    "predict_survival": {
        "label": "predict_survival (survival curve)",
        "requires_time": False,
        "requires_features": True,
    },
    "predict_expectation": {
        "label": "predict_expectation (expected survival time)",
        "requires_time": False,
        "requires_features": True,
    },
}

PREDICTION_OPTIONS = [
    (PREDICTION_METHOD_CONFIG[name]["label"], name) for name in PREDICTION_METHOD_ORDER
]
PREDICTION_DEFAULT_METHOD = "predict_survival_at"


def create_ui() -> gr.Blocks:
    """Create the Gradio Blocks application for worker interaction."""
    custom_css = """
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #0d9488 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 0 !important;
    }
    .subtitle {
        text-align: center;
        opacity: 0.7;
        font-size: 1.1rem !important;
        margin-top: 0 !important;
    }
    .event-log textarea {
        font-family: monospace;
        background-color: #1e1e1e !important;
        color: #00ff00 !important;
    }
    """

    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="DC-Cox Worker") as app:
        # ── Global State ───────────────────────────────────────────────
        worker_state = gr.State(value=None)  # DCCoxWorker instance
        active_project_id = gr.State(value=None)
        results_project_id = gr.State(value=None)
        feature_schema_state = gr.State(value=[])

        # ── Header ─────────────────────────────────────────────────────
        gr.Markdown('<p class="main-title">DC-Cox Worker</p>')
        gr.Markdown(
            '<p class="subtitle">Federated Cox Proportional Hazards Regression</p>',
        )

        with gr.Tabs():
            # ── Tab 1: Dashboard ───────────────────────────────────────
            with gr.TabItem("Dashboard", id="dashboard"):
                with gr.Row():
                    master_url = gr.Textbox(
                        label="Master URL",
                        value=settings.master_url,
                        scale=3,
                    )
                    connect_btn = gr.Button("Connect to Master", variant="primary")
                connect_status = gr.Markdown("")

                gr.Markdown("---")
                gr.Markdown("### 🏢 Available Projects")
                refresh_btn = gr.Button("Refresh Projects")
                projects_table = gr.Dataframe(
                    headers=[
                        "Project ID",
                        "Project Name",
                        "Status",
                        "Workers Count",
                        "Created At",
                    ],
                    interactive=False,
                )

            # ── Tab 2: Create Project ──────────────────────────────────
            with gr.TabItem("Create Project", id="create"):
                gr.Markdown("### Create a New Analysis Project")

                # Dynamically build UI from Pydantic Schema
                fields = ProjectSchema.model_fields

                with gr.Row():
                    p_name = gr.Textbox(
                        label="Project Name", info=fields["name"].description, scale=2
                    )
                    p_use_case = gr.Dropdown(
                        choices=["horizontal"],
                        value="horizontal",
                        label="Use Case",
                        info=fields["use_case"].description,
                    )
                with gr.Accordion("Projection Hyperparameters", open=True):
                    with gr.Row():
                        p_k = gr.Slider(
                            1,
                            50,
                            value=fields["k"].default,
                            step=1,
                            label="k",
                            info=fields["k"].description,
                        )
                        p_r = gr.Slider(
                            10,
                            500,
                            value=fields["r"].default,
                            step=10,
                            label="r",
                            info=fields["r"].description,
                        )
                    with gr.Row():
                        p_bs_prop = gr.Slider(
                            0.1,
                            1.0,
                            value=fields["bs_prop"].default,
                            step=0.1,
                            label="bs_prop",
                            info=fields["bs_prop"].description,
                        )
                        p_bs_times = gr.Slider(
                            1,
                            100,
                            value=fields["bs_times"].default,
                            step=1,
                            label="bs_times",
                            info=fields["bs_times"].description,
                        )
                        p_bs_replace = gr.Checkbox(
                            value=fields["bs_replace"].default,
                            label="bs_replace",
                            info=fields["bs_replace"].description,
                        )

                with gr.Accordion("Regression Hyperparameters", open=True), gr.Row():
                    p_alpha = gr.Slider(
                        0.01,
                        0.20,
                        value=fields["alpha"].default,
                        step=0.01,
                        label="alpha",
                        info=fields["alpha"].description,
                    )
                    p_step_size = gr.Slider(
                        0.1,
                        1.0,
                        value=fields["step_size"].default,
                        step=0.1,
                        label="step_size",
                        info=fields["step_size"].description,
                    )
                    p_var_thres = gr.Number(
                        value=fields["var_thres"].default,
                        label="var_thres",
                        info=fields["var_thres"].description,
                    )

                with gr.Accordion("Data Options", open=True), gr.Row():
                    p_centering = gr.Dropdown(
                        choices=["mean", "none"],
                        value="none",
                        label="centering",
                        info=fields["centering"].description,
                    )
                    p_keep_features = gr.Textbox(
                        value=",".join(fields["keep_feature_cols"].default or []),
                        label="keep_feature_cols",
                        info=fields["keep_feature_cols"].description
                        + " (comma separated)",
                    )
                    p_meta_cols = gr.Textbox(
                        value=",".join(fields["meta_cols"].default or []),
                        label="meta_cols",
                        info=fields["meta_cols"].description + " (comma separated)",
                    )

                create_project_btn = gr.Button("Create Project", variant="primary")
                create_status = gr.Markdown("")

            # ── Tab 3: Active Workspace ────────────────────────────────
            with gr.TabItem("Active Workspace", id="workspace"):
                with gr.Row():
                    join_project_id = gr.Textbox(label="Project ID to Join")
                    worker_name_input = gr.Textbox(
                        label="Your Worker Name", value=settings.worker_name
                    )
                data_path = gr.Textbox(
                    label="Local Clinical CSV File Path",
                    placeholder="/path/to/clinical.csv",
                )

                with gr.Row():
                    join_btn = gr.Button("1. Join Project")
                    lock_btn = gr.Button(
                        "2. Lock Project (Creator)", variant="secondary"
                    )
                    start_btn = gr.Button(
                        "3. Start Analysis (Creator)", variant="secondary"
                    )
                    run_btn = gr.Button("4. Run Local Compute", variant="primary")

                workspace_status = gr.Markdown("")

                gr.Markdown("### Real-time Event Log")
                event_log_area = gr.TextArea(
                    interactive=False,
                    label="Server Events",
                    lines=10,
                    elem_classes="event-log",
                    autoscroll=True,
                )

                # Summary results will appear here after run
                summary_table = gr.Dataframe(
                    label="My Results: Coefficients Summary",
                    interactive=False,
                    type="pandas",
                )
                summary_download = gr.DownloadButton("Download Summary CSV")
                coef_plot = gr.Plot(label="Coefficients Plot")
                coef_plot_download = gr.DownloadButton("Download Coefficients Plot PNG")

                with gr.Accordion("Baseline Details", open=False):
                    baseline_cumhazards_table = gr.Dataframe(
                        label="Baseline Cumulative Hazards",
                        interactive=False,
                        type="pandas",
                    )
                    baseline_cumhazards_download = gr.DownloadButton(
                        "Download Baseline Cumulative Hazards CSV"
                    )
                    baseline_survival_table = gr.Dataframe(
                        label="Baseline Survival",
                        interactive=False,
                        type="pandas",
                    )
                    baseline_survival_download = gr.DownloadButton(
                        "Download Baseline Survival CSV"
                    )
                    baseline_hazard_table = gr.Dataframe(
                        label="Baseline Hazard",
                        interactive=False,
                        type="pandas",
                    )
                    baseline_hazard_download = gr.DownloadButton(
                        "Download Baseline Hazard CSV"
                    )

                with gr.Accordion("Prediction Sandbox", open=False):
                    prediction_method = gr.Dropdown(
                        label="Prediction Method",
                        choices=PREDICTION_OPTIONS,
                        value=PREDICTION_DEFAULT_METHOD,
                    )
                    prediction_time = gr.Number(
                        label="Time t",
                        value=0.0,
                        info="Required for *_at methods; ignored otherwise.",
                    )
                    prediction_input = gr.Dataframe(
                        label="Feature Inputs",
                        datatype="number",
                        type="pandas",
                        row_count=1,
                        column_count=(0, "dynamic"),
                    )
                    run_prediction_btn = gr.Button(
                        "Run Prediction", variant="secondary"
                    )
                    prediction_status = gr.Markdown("")
                    prediction_output = gr.Dataframe(
                        label="Prediction Output", interactive=False, type="pandas"
                    )

            # ── Tab 4: History ─────────────────────────────────────────
            with gr.TabItem("History", id="history"):
                history_refresh_btn = gr.Button("Refresh History")
                history_table = gr.Dataframe(
                    headers=[
                        "Project ID",
                        "Project Name",
                        "Status",
                        "Workers Count",
                        "Created At",
                    ],
                    interactive=False,
                )
                with gr.Row():
                    history_pid = gr.Textbox(
                        label="Project ID",
                        info="Select a project above to populate automatically",
                    )
                    fetch_history_btn = gr.Button("Fetch My Results")

                history_status = gr.Markdown("")
                history_results = gr.Dataframe(
                    label="History: Coefficients Summary",
                    interactive=False,
                    type="pandas",
                )
                history_summary_download = gr.DownloadButton(
                    "Download History Summary CSV"
                )
                history_coef_plot = gr.Plot(label="History Coefficients Plot")
                history_coef_plot_download = gr.DownloadButton(
                    "Download History Coefficients Plot PNG"
                )

                with gr.Accordion("History Baseline Details", open=False):
                    history_baseline_cumhazards_table = gr.Dataframe(
                        label="Baseline Cumulative Hazards",
                        interactive=False,
                        type="pandas",
                    )
                    history_baseline_cumhazards_download = gr.DownloadButton(
                        "Download History Baseline Cumulative Hazards CSV"
                    )
                    history_baseline_survival_table = gr.Dataframe(
                        label="Baseline Survival",
                        interactive=False,
                        type="pandas",
                    )
                    history_baseline_survival_download = gr.DownloadButton(
                        "Download History Baseline Survival CSV"
                    )
                    history_baseline_hazard_table = gr.Dataframe(
                        label="Baseline Hazard",
                        interactive=False,
                        type="pandas",
                    )
                    history_baseline_hazard_download = gr.DownloadButton(
                        "Download History Baseline Hazard CSV"
                    )

                with gr.Accordion("History Prediction Sandbox", open=False):
                    history_prediction_method = gr.Dropdown(
                        label="Prediction Method",
                        choices=PREDICTION_OPTIONS,
                        value=PREDICTION_DEFAULT_METHOD,
                    )
                    history_prediction_time = gr.Number(
                        label="Time t",
                        value=0.0,
                        info="Required for *_at methods; ignored otherwise.",
                    )
                    history_prediction_input = gr.Dataframe(
                        label="Feature Inputs",
                        datatype="number",
                        type="pandas",
                        row_count=1,
                        column_count=(0, "dynamic"),
                    )
                    history_run_prediction_btn = gr.Button(
                        "Run Prediction", variant="secondary"
                    )
                    history_prediction_status = gr.Markdown("")
                    history_prediction_output = gr.Dataframe(
                        label="Prediction Output", interactive=False, type="pandas"
                    )

        # ── Event Handlers ─────────────────────────────────────────────

        def _default_prediction_df(headers: list[str]) -> pd.DataFrame | None:
            if not headers:
                return None
            return pd.DataFrame([dict.fromkeys(headers, 0.0)])

        def _prepare_feature_df(
            value: pd.DataFrame | None, headers: list[str]
        ) -> pd.DataFrame:
            if not headers:
                msg = "Feature schema unavailable. Run analysis first."
                raise ValueError(msg)
            if value is None or value.empty:
                msg = "Enter at least one row of feature values."
                raise ValueError(msg)
            missing = [h for h in headers if h not in value.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            ordered = value.loc[:, headers]
            if ordered.isnull().any().any():
                raise ValueError("Fill in all feature values before predicting.")
            return ordered.astype(float)

        def _format_prediction_output(
            result: pd.DataFrame | pd.Series | np.ndarray | float, method: str
        ) -> pd.DataFrame:
            if isinstance(result, pd.DataFrame):
                return result
            if isinstance(result, pd.Series):
                return result.to_frame(name=method)
            if np.isscalar(result):
                return pd.DataFrame({method: [float(result)]})
            arr = np.asarray(result)
            if arr.ndim == 0:
                return pd.DataFrame({method: [float(arr)]})
            if arr.ndim == 1:
                return pd.DataFrame({method: arr})
            return pd.DataFrame(arr)

        def _workspace_empty_outputs(
            message: str, pid: str | None, headers: list[str]
        ) -> tuple[Any, ...]:
            return (
                message,
                None,
                None,
                None,
                None,
                pid,
                headers,
                None,
                "",
                None,
                None,
                None,
                None,
                None,
                "",
                None,
                None,
                None,
            )

        def _history_empty_outputs(
            message: str, pid: str | None, headers: list[str]
        ) -> tuple[Any, ...]:
            return (
                message,
                None,
                message,
                None,
                None,
                None,
                None,
                pid,
                headers,
                None,
                "",
                None,
                None,
                None,
                None,
                None,
                "",
                None,
                None,
                None,
            )

        def _create_temp_file(
            data: str | bytes, *, prefix: str, suffix: str, binary: bool = False
        ) -> str:
            safe_prefix = (prefix or "export").replace(" ", "_")
            mode = "wb" if binary else "w"
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
                prefix=f"{safe_prefix}_",
                mode=mode,
            ) as tmp:
                if binary:
                    if isinstance(data, str):
                        tmp.write(data.encode("utf-8"))
                    else:
                        tmp.write(data)
                else:
                    if isinstance(data, bytes):
                        tmp.write(data.decode("utf-8"))
                    else:
                        tmp.write(data)
                return tmp.name

        def _create_temp_csv(content: str, prefix: str) -> str:
            return _create_temp_file(content, prefix=prefix, suffix=".csv")

        def _build_coef_plot(summary: pd.DataFrame) -> plt.Figure | None:
            if summary is None or summary.empty:
                return None
            if "coef" not in summary.columns:
                return None
            df = summary.copy()
            features = (
                df["feature"].astype(str)
                if "feature" in df.columns
                else df.index.astype(str)
            )
            df = df.assign(_feature=features)
            df = df.sort_values("coef")
            lower_col = next(
                (c for c in df.columns if c.startswith("coef lower")), None
            )
            upper_col = next(
                (c for c in df.columns if c.startswith("coef upper")), None
            )
            if lower_col is None or upper_col is None:
                return None
            fig, ax = plt.subplots(
                figsize=(8, max(4.0, min(10.0, 1.5 + 0.35 * len(df))))
            )
            ax.scatter(
                df["coef"],
                df["_feature"],
                color="steelblue",
                label="Mean",
                zorder=5,
            )
            lower_err = df["coef"] - df[lower_col]
            upper_err = df[upper_col] - df["coef"]
            ax.errorbar(
                df["coef"],
                df["_feature"],
                xerr=[lower_err, upper_err],
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=2,
                label=lower_col.split("coef lower")[-1].strip() or "CI",
            )
            ax.set_xlabel("log(HR) (CI)")
            ax.set_ylabel("Features")
            ax.set_title("Cox Proportional Hazard Regression Coefficients")
            ax.grid(axis="x", linestyle="--", alpha=0.7)
            ax.legend()
            fig.tight_layout()
            return fig

        def _download_table_csv(
            worker: DCCoxWorker | None, pid: str | None, table_key: str
        ) -> str:
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
            return _create_temp_csv(table.to_csv(index=False), table_key)

        def _download_plot_file(worker: DCCoxWorker | None, pid: str | None) -> bytes:
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
            fig = _build_coef_plot(summary)
            if fig is None:
                raise gr.Error("Unable to generate coefficient plot.")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            return buf.getvalue()

        def _download_plot_path(
            worker: DCCoxWorker | None, pid: str | None, label: str
        ) -> str:
            data = _download_plot_file(worker, pid)
            return _create_temp_file(data, prefix=label, suffix=".png", binary=True)

        def _compute_prediction_output(
            worker: DCCoxWorker | None,
            result_pid: str | None,
            method: str,
            t_value: float | None,
            feature_headers: list[str],
            feature_values: pd.DataFrame | None,
        ) -> pd.DataFrame:
            if worker is None:
                raise ValueError("Connect to master first.")
            if result_pid is None:
                raise ValueError("Run analysis or load results first.")
            surv = worker.get_survival_function(result_pid)
            if surv is None:
                raise ValueError(f"No cached results for project {result_pid}")
            cfg = PREDICTION_METHOD_CONFIG.get(method)
            if cfg is None:
                raise ValueError(f"Unknown method '{method}'")

            args: list[Any] = []
            if cfg["requires_time"]:
                if t_value is None:
                    raise ValueError("Provide a time value for this method.")
                args.append(float(t_value))
            if cfg["requires_features"]:
                names = feature_headers or worker.get_feature_names(result_pid) or []
                X = _prepare_feature_df(feature_values, names)
                args.append(X)

            prediction = getattr(surv, method)(*args)
            return _format_prediction_output(prediction, method)

        def handle_connect(url: str) -> tuple[Any, str]:
            try:
                c = DCCoxWorker(url)
                c.list_projects()  # smoke test
                return c, f"**Connected** to `{url}` successfully!"
            except Exception as e:
                return None, f"**Connection failed**: {e}"

        connect_btn.click(
            fn=handle_connect,
            inputs=[master_url],
            outputs=[worker_state, connect_status],
        )

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
            if worker is None:
                return "❌ Connect to master first (Dashboard tab)."
            if not name.strip():
                return "❌ Enter a project name."

            # Parse lists
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
                return f"✅ **Project created successfully!** ID: `{pid}`"
            except Exception as e:
                return f"❌ **Failed to create**: {e}"

        create_project_btn.click(
            fn=handle_create_project,
            inputs=[
                worker_state,
                p_name,
                p_use_case,
                p_k,
                p_r,
                p_bs_prop,
                p_bs_times,
                p_bs_replace,
                p_alpha,
                p_step_size,
                p_var_thres,
                p_centering,
                p_keep_features,
                p_meta_cols,
            ],
            outputs=[create_status],
        )

        def handle_refresh(worker: DCCoxWorker | None) -> list[list[str]]:
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

        refresh_btn.click(
            fn=handle_refresh, inputs=[worker_state], outputs=[projects_table]
        )
        history_refresh_btn.click(
            fn=handle_refresh, inputs=[worker_state], outputs=[history_table]
        )

        def handle_join(
            worker: DCCoxWorker | None, pid: str, cname: str, dpath: str
        ) -> tuple[str, str]:
            if worker is None:
                return "", "❌ Connect first."
            if not pid.strip():
                return "", "❌ Enter project ID."
            if not dpath.strip():
                return "", "❌ Enter data path to determine n_features."

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
                    f"✅ Joined as `{wid}` (n_features={n_features}). Waiting for lock.",
                )
            except Exception as e:
                return "", f"❌ Join failed: {e}"

        join_btn.click(
            fn=handle_join,
            inputs=[worker_state, join_project_id, worker_name_input, data_path],
            outputs=[active_project_id, workspace_status],
        )

        def handle_lock(worker: DCCoxWorker | None, pid: str | None) -> str:
            if worker is None or not pid:
                return "❌ Join a project first."
            try:
                worker.lock_project(pid)
                return "✅ Project locked! New workers can no longer join."
            except Exception as e:
                return f"❌ Lock failed: {e}"

        lock_btn.click(
            fn=handle_lock,
            inputs=[worker_state, active_project_id],
            outputs=[workspace_status],
        )

        def handle_start(worker: DCCoxWorker | None, pid: str | None) -> str:
            if worker is None or not pid:
                return "❌ Join a project first."
            try:
                worker.start_project(pid)
                return "✅ Analysis started! Xanc generated by server."
            except Exception as e:
                return f"❌ Start failed: {e}"

        start_btn.click(
            fn=handle_start,
            inputs=[worker_state, active_project_id],
            outputs=[workspace_status],
        )

        def handle_run(
            worker: DCCoxWorker | None,
            pid: str | None,
            dpath: str,
            current_result_pid: str | None,
            feature_headers: list[str],
        ) -> tuple[Any, ...]:
            def _empty(msg: str) -> tuple[Any, ...]:
                return _workspace_empty_outputs(
                    msg, current_result_pid, feature_headers
                )

            if worker is None or not pid:
                return _empty("❌ Join a project first.")
            if not dpath.strip():
                return _empty("❌ Enter data path.")

            try:
                pid_clean = pid.strip()
                surv = worker.run_local_pipeline(
                    pid_clean, dpath.strip(), poll_interval=2.0
                )
                tables = build_result_tables(surv)
                feature_names = worker.get_feature_names(pid_clean) or []
                prediction_df = _default_prediction_df(feature_names)
                workspace_fig = _build_coef_plot(tables["summary"])
                history_fig = _build_coef_plot(tables["summary"])
                status_msg = "✅ Local compute & global aggregation completed!"
                return (
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
                return _empty(f"❌ Analysis failed: {e}")

        run_btn.click(
            fn=handle_run,
            inputs=[
                worker_state,
                active_project_id,
                data_path,
                results_project_id,
                feature_schema_state,
            ],
            outputs=[
                workspace_status,
                summary_table,
                baseline_cumhazards_table,
                baseline_survival_table,
                baseline_hazard_table,
                results_project_id,
                feature_schema_state,
                prediction_input,
                prediction_status,
                prediction_output,
                history_baseline_cumhazards_table,
                history_baseline_survival_table,
                history_baseline_hazard_table,
                history_prediction_input,
                history_prediction_status,
                history_prediction_output,
                coef_plot,
                history_coef_plot,
            ],
        )

        def handle_prediction(
            worker: DCCoxWorker | None,
            result_pid: str | None,
            method: str,
            t_value: float | None,
            feature_headers: list[str],
            feature_values: pd.DataFrame | None,
        ) -> tuple[str, Any]:
            try:
                formatted = _compute_prediction_output(
                    worker, result_pid, method, t_value, feature_headers, feature_values
                )
                return f"✅ {method} computed.", formatted
            except ValueError as err:
                return f"❌ {err}", None
            except Exception as err:
                return f"❌ Prediction failed: {err}", None

        run_prediction_btn.click(
            fn=handle_prediction,
            inputs=[
                worker_state,
                results_project_id,
                prediction_method,
                prediction_time,
                feature_schema_state,
                prediction_input,
            ],
            outputs=[prediction_status, prediction_output],
        )

        history_run_prediction_btn.click(
            fn=handle_prediction,
            inputs=[
                worker_state,
                results_project_id,
                history_prediction_method,
                history_prediction_time,
                feature_schema_state,
                history_prediction_input,
            ],
            outputs=[history_prediction_status, history_prediction_output],
        )

        summary_download.click(
            fn=lambda worker, pid: _download_table_csv(worker, pid, "summary"),
            inputs=[worker_state, results_project_id],
            outputs=[summary_download],
        )
        baseline_cumhazards_download.click(
            fn=lambda worker, pid: _download_table_csv(
                worker, pid, "baseline_cumhazards"
            ),
            inputs=[worker_state, results_project_id],
            outputs=[baseline_cumhazards_download],
        )
        baseline_survival_download.click(
            fn=lambda worker, pid: _download_table_csv(
                worker, pid, "baseline_survival"
            ),
            inputs=[worker_state, results_project_id],
            outputs=[baseline_survival_download],
        )
        baseline_hazard_download.click(
            fn=lambda worker, pid: _download_table_csv(worker, pid, "baseline_hazard"),
            inputs=[worker_state, results_project_id],
            outputs=[baseline_hazard_download],
        )
        history_summary_download.click(
            fn=lambda worker, pid: _download_table_csv(worker, pid, "summary"),
            inputs=[worker_state, results_project_id],
            outputs=[history_summary_download],
        )
        history_baseline_cumhazards_download.click(
            fn=lambda worker, pid: _download_table_csv(
                worker, pid, "baseline_cumhazards"
            ),
            inputs=[worker_state, results_project_id],
            outputs=[history_baseline_cumhazards_download],
        )
        history_baseline_survival_download.click(
            fn=lambda worker, pid: _download_table_csv(
                worker, pid, "baseline_survival"
            ),
            inputs=[worker_state, results_project_id],
            outputs=[history_baseline_survival_download],
        )
        history_baseline_hazard_download.click(
            fn=lambda worker, pid: _download_table_csv(worker, pid, "baseline_hazard"),
            inputs=[worker_state, results_project_id],
            outputs=[history_baseline_hazard_download],
        )
        coef_plot_download.click(
            fn=lambda worker, pid: _download_plot_path(
                worker, pid, "coefficients_plot"
            ),
            inputs=[worker_state, results_project_id],
            outputs=[coef_plot_download],
        )
        history_coef_plot_download.click(
            fn=lambda worker, pid: _download_plot_path(
                worker, pid, "history_coefficients_plot"
            ),
            inputs=[worker_state, results_project_id],
            outputs=[history_coef_plot_download],
        )

        def handle_poll_events(worker: DCCoxWorker | None, pid: str | None) -> str:
            if worker is None or not pid:
                return "Awaiting project connection..."
            try:
                events = worker.get_events(pid)
                lines = [f"[{e['time'][:19]}] {e['message']}" for e in events]
                return "\n".join(lines)
            except Exception as exc:
                logger.exception("Failed to poll events for %s: %s", pid, exc)
                return "Polling events..."

        def handle_fetch_history(
            worker: DCCoxWorker | None,
            pid: str,
            current_result_pid: str | None,
            feature_headers: list[str],
        ) -> tuple[Any, ...]:
            def _error(msg: str) -> tuple[Any, ...]:
                return _history_empty_outputs(msg, current_result_pid, feature_headers)

            if worker is None:
                return _error("❌ Connect to master first")
            if worker.worker_id is None:
                return _error("❌ Join a project first to fetch your results")
            if not pid.strip():
                return _error("❌ Project ID is required")

            try:
                pid_clean = pid.strip()
                surv = worker.get_survival_function(pid_clean)
                if surv is None:
                    return _error(f"❌ No cached results for `{pid_clean}`")
                tables = build_result_tables(surv)
                feature_names = worker.get_feature_names(pid_clean) or []
                prediction_df = _default_prediction_df(feature_names)
                status_msg = f"✅ Loaded stored results for `{pid_clean}`."
                workspace_fig = _build_coef_plot(tables["summary"])
                history_fig = _build_coef_plot(tables["summary"])
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
                return _error(f"❌ Fetch failed: {e}")

        fetch_history_btn.click(
            fn=handle_fetch_history,
            inputs=[
                worker_state,
                history_pid,
                results_project_id,
                feature_schema_state,
            ],
            outputs=[
                history_status,
                history_results,
                workspace_status,
                summary_table,
                baseline_cumhazards_table,
                baseline_survival_table,
                baseline_hazard_table,
                results_project_id,
                feature_schema_state,
                prediction_input,
                prediction_status,
                prediction_output,
                history_baseline_cumhazards_table,
                history_baseline_survival_table,
                history_baseline_hazard_table,
                history_prediction_input,
                history_prediction_status,
                history_prediction_output,
                coef_plot,
                history_coef_plot,
            ],
        )

        def on_select_history(evt: gr.SelectData) -> str:
            # Table headers: ["Project ID", "Project Name", "Status", "Workers Count", "Created At"]
            # ID is at index 0 (evt.value)
            return str(evt.value or "")

        history_table.select(
            fn=on_select_history,
            inputs=[],
            outputs=[history_pid],
        )

        # Poll the events every 2 seconds using gr.Timer
        timer = gr.Timer(2)
        timer.tick(
            fn=handle_poll_events,
            inputs=[worker_state, active_project_id],
            outputs=[event_log_area],
        )

    app.theme = theme
    app.css = custom_css
    return app


def create_app() -> FastAPI:
    """Create a FastAPI app to host the Gradio UI for uvicorn."""
    fastapi_app = FastAPI(title="DC-Cox Worker UI")
    gradio_app = create_ui()
    return gr.mount_gradio_app(fastapi_app, gradio_app, path="/")


app = create_app()

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "dccox.service.worker.ui:app",
        host="0.0.0.0",
        port=settings.worker_port,
        reload=False,
    )
