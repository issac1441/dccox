"""Gradio frontend for DC-Cox worker operations."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
import gradio as gr
import pandas as pd

from .worker import DCCoxWorker

logger = logging.getLogger(__name__)


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
    """

    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="DC-Cox Worker") as app:
        # ── State ──────────────────────────────────────────────────────
        worker_state = gr.State(value=None)  # DCCoxWorker instance
        project_id_state = gr.State(value=None)

        # ── Header ─────────────────────────────────────────────────────
        gr.Markdown('<p class="main-title">DC-Cox Worker</p>')
        gr.Markdown(
            '<p class="subtitle">Federated Cox Proportional Hazards Regression</p>',
        )

        with gr.Tabs():
            # ── Tab 1: Connect & Projects ──────────────────────────────
            with gr.TabItem("Setup", id="setup"):
                with gr.Row():
                    master_url = gr.Textbox(
                        label="Master URL",
                        value="http://localhost:8000",
                        scale=3,
                    )
                    connect_btn = gr.Button("Connect", variant="primary")
                connect_status = gr.Markdown("")

                gr.Markdown("---")
                gr.Markdown("### Create New Project")
                with gr.Row():
                    project_name = gr.Textbox(label="Project Name", scale=2)
                    use_case = gr.Dropdown(
                        choices=["horizontal"],
                        value="horizontal",
                        label="Use Case",
                    )
                with gr.Accordion("Hyperparameters", open=False):
                    with gr.Row():
                        k_input = gr.Slider(1, 50, value=20, step=1, label="k")
                        r_input = gr.Slider(10, 500, value=100, step=10, label="r")
                    with gr.Row():
                        alpha_input = gr.Slider(
                            0.01, 0.20, value=0.05, step=0.01, label="alpha"
                        )
                        step_size_input = gr.Slider(
                            0.1, 1.0, value=0.5, step=0.1, label="Step size"
                        )
                create_project_btn = gr.Button("Create Project")
                create_status = gr.Markdown("")

                gr.Markdown("---")
                gr.Markdown("### Existing Projects")
                refresh_btn = gr.Button("Refresh")
                projects_table = gr.Dataframe(
                    headers=["ID", "Name", "Status", "Workers"],
                    interactive=False,
                )

            # ── Tab 2: Join & Run ──────────────────────────────────────
            with gr.TabItem("Analysis", id="analysis"):
                with gr.Row():
                    join_project_id = gr.Textbox(label="Project ID")
                    worker_name_input = gr.Textbox(
                        label="Worker Name", value="my-worker"
                    )
                data_path = gr.Textbox(
                    label="Local CSV Data Path",
                    placeholder="/path/to/clinical.csv",
                )
                with gr.Row():
                    join_btn = gr.Button("Join Project")
                    start_btn = gr.Button("Start Project", variant="secondary")
                    run_btn = gr.Button("Run Analysis", variant="primary", size="lg")
                analysis_status = gr.Markdown("")

            # ── Tab 3: Results ─────────────────────────────────────────
            with gr.TabItem("Results", id="results"):
                summary_table = gr.Dataframe(
                    label="Coefficients Summary", interactive=False
                )
                results_json = gr.JSON(label="Full Results")

        # ── Event Handlers ─────────────────────────────────────────────

        def handle_connect(
            url: str,
        ) -> tuple[Any, str]:
            try:
                c = DCCoxWorker(url)
                c.list_projects()  # smoke test
                return c, "Connected to master"
            except Exception as e:
                return None, f"Connection failed: {e}"

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
            alpha: float,
            step_size: float,
        ) -> str:
            if worker is None:
                return "Connect to master first"
            if not name.strip():
                return "Enter a project name"
            pid = worker.create_project(
                {
                    "name": name.strip(),
                    "use_case": uc,
                    "k": int(k),
                    "r": int(r),
                    "alpha": float(alpha),
                    "step_size": float(step_size),
                }
            )
            return f"Project created: `{pid}`"

        create_project_btn.click(
            fn=handle_create_project,
            inputs=[
                worker_state,
                project_name,
                use_case,
                k_input,
                r_input,
                alpha_input,
                step_size_input,
            ],
            outputs=[create_status],
        )

        def handle_refresh(
            worker: DCCoxWorker | None,
        ) -> list[list[str]]:
            if worker is None:
                return []
            projects = worker.list_projects()
            return [
                [
                    p["id"],
                    p["config"]["name"],
                    p["status"],
                    str(len(p["workers"])),
                ]
                for p in projects
            ]

        refresh_btn.click(
            fn=handle_refresh,
            inputs=[worker_state],
            outputs=[projects_table],
        )

        def handle_join(
            worker: DCCoxWorker | None,
            pid: str,
            cname: str,
            dpath: str,
        ) -> tuple[str, str]:
            if worker is None:
                return "", "Connect first"
            if not pid.strip():
                return "", "Enter project ID"
            if not dpath.strip():
                return "", "Enter data path to determine n_features"

            # Determine n_features based on project config and local data
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
            except Exception as e:
                return "", f"Error computing n_features: {e}"

            try:
                wid = worker.join_project(pid.strip(), cname.strip(), n_features)
                return pid.strip(), f"Joined as `{wid}` (n_features={n_features})"
            except Exception as e:
                return "", f"Join failed: {e}"

        join_btn.click(
            fn=handle_join,
            inputs=[worker_state, join_project_id, worker_name_input, data_path],
            outputs=[project_id_state, analysis_status],
        )

        def handle_start(worker: DCCoxWorker | None, pid: str | None) -> str:
            if worker is None:
                return "Connect first"
            if not pid:
                return "Join a project first"
            try:
                worker.start_project(pid)
                return "Project started — Xanc generated"
            except Exception as e:
                return f"Start failed: {e}"

        start_btn.click(
            fn=handle_start,
            inputs=[worker_state, project_id_state],
            outputs=[analysis_status],
        )

        def handle_run(
            worker: DCCoxWorker | None,
            pid: str | None,
            dpath: str,
        ) -> tuple[str, Any, Any]:
            if worker is None:
                return "Connect first", None, None
            if not pid:
                return "Join a project first", None, None
            if not dpath.strip():
                return "Enter data path", None, None

            try:
                surv = worker.run_local_pipeline(pid, dpath.strip())
                summary_df = surv.summary
                return (
                    "Analysis completed",
                    summary_df,
                    surv.coef.to_dict(),
                )
            except Exception as e:
                return f"Analysis failed: {e}", None, None

        run_btn.click(
            fn=handle_run,
            inputs=[worker_state, project_id_state, data_path],
            outputs=[analysis_status, summary_table, results_json],
        )

    # In Gradio 6.0, theme and css shouldn't be in Blocks constructor for uvicorn mounting
    app.theme = theme
    app.css = custom_css
    return app


def create_app() -> FastAPI:
    """Create a FastAPI app to host the Gradio UI for uvicorn."""
    fastapi_app = FastAPI(title="DC-Cox Worker UI")
    gradio_app = create_ui()
    # Using mount_gradio_app is standard for running Gradio through ASGI (Uvicorn)
    return gr.mount_gradio_app(fastapi_app, gradio_app, path="/")


app = create_app()

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "dccox.service.worker.ui:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
