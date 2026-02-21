"""Gradio frontend for DC-Cox worker operations."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
import gradio as gr
import pandas as pd

from dccox.service.master.schemas import ProjectSchema

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
                        value="http://localhost:8000",
                        scale=3,
                    )
                    connect_btn = gr.Button("Connect to Master", variant="primary")
                connect_status = gr.Markdown("")

                gr.Markdown("---")
                gr.Markdown("### 🏢 Available Projects")
                refresh_btn = gr.Button("Refresh Projects")
                projects_table = gr.Dataframe(
                    headers=["ID", "Name", "Status", "Workers Count", "Created"],
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
                        value=",".join(fields["keep_feature_cols"].default),
                        label="keep_feature_cols",
                        info=fields["keep_feature_cols"].description
                        + " (comma separated)",
                    )
                    p_meta_cols = gr.Textbox(
                        value=",".join(fields["meta_cols"].default),
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
                        label="Your Worker Name", value="my-worker"
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
                    label="My Results: Coefficients Summary", interactive=False
                )

            # ── Tab 4: History ─────────────────────────────────────────
            with gr.TabItem("History", id="history"):
                history_refresh_btn = gr.Button("Refresh History")
                history_table = gr.Dataframe(
                    headers=["ID", "Name", "Created"],
                    interactive=False,
                )
                with gr.Row():
                    history_pid = gr.Textbox(label="Project ID")
                    history_wid = gr.Textbox(label="Worker Name/ID (optional)")
                    fetch_history_btn = gr.Button("Fetch Results")

                history_results = gr.JSON(label="Full Results")

        # ── Event Handlers ─────────────────────────────────────────────

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
            worker: DCCoxWorker | None, pid: str | None, dpath: str
        ) -> tuple[str, Any]:
            if worker is None or not pid:
                return "❌ Join a project first.", None
            if not dpath.strip():
                return "❌ Enter data path.", None

            try:
                surv = worker.run_local_pipeline(pid, dpath.strip(), poll_interval=2.0)
                summary_df = surv.summary
                return "✅ Local compute & global aggregation completed!", summary_df
            except Exception as e:
                return f"❌ Analysis failed: {e}", None

        run_btn.click(
            fn=handle_run,
            inputs=[worker_state, active_project_id, data_path],
            outputs=[workspace_status, summary_table],
        )

        def handle_poll_events(worker: DCCoxWorker | None, pid: str | None) -> str:
            if worker is None or not pid:
                return "Awaiting project connection..."
            try:
                events = worker.get_events(pid)
                lines = [f"[{e['time'][:19]}] {e['message']}" for e in events]
                return "\n".join(lines)
            except Exception:
                return "Polling events..."

        def handle_fetch_history(
            worker: DCCoxWorker | None, pid: str, wid: str
        ) -> str | dict:
            if worker is None:
                return "❌ Connect to master first"
            if not pid.strip() or not wid.strip():
                return "❌ Project ID and Worker ID are required"
            try:
                resp = worker._http.get(
                    f"/api/projects/{pid.strip()}/results/{wid.strip()}"
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {"error": str(e)}

        fetch_history_btn.click(
            fn=handle_fetch_history,
            inputs=[worker_state, history_pid, history_wid],
            outputs=[history_results],
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
        port=8001,
        reload=True,
    )
