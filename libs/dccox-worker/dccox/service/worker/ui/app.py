"""Gradio application for the DC-Cox worker UI.

Single Responsibility: Defines the UI layout (components) and wires each
component to the appropriate handler.  All business logic lives in
``handlers``, all styling in ``theme``, all utilities in ``helpers``.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
import gradio as gr

from dccox.service.master.schemas import ProjectSchema
from dccox.service.worker.config import settings
from dccox.service.worker.ui.handlers import (
    download_plot_path,
    download_table_csv,
    handle_connect,
    handle_create_project,
    handle_fetch_history,
    handle_join,
    handle_lock,
    handle_poll_events,
    handle_prediction,
    handle_refresh,
    handle_run,
    handle_start,
)
from dccox.service.worker.ui.prediction import (
    PREDICTION_DEFAULT_METHOD,
    PREDICTION_OPTIONS,
)
from dccox.service.worker.ui.theme import CUSTOM_CSS, THEME

logger = logging.getLogger(__name__)


def create_ui() -> gr.Blocks:
    """Create the Gradio Blocks application for worker interaction."""
    with gr.Blocks(title="DC-Cox Worker") as blocks_app:
        # ── Global State ───────────────────────────────────────────────
        worker_state = gr.State(value=None)
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
                gr.Markdown("### Available Projects")
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
                fields = ProjectSchema.model_fields

                with gr.Row():
                    p_name = gr.Textbox(
                        label="Project Name",
                        info=fields["name"].description,
                        scale=2,
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
                        label="Prediction Output",
                        interactive=False,
                        type="pandas",
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
                        label="Prediction Output",
                        interactive=False,
                        type="pandas",
                    )

        # ── Wire Event Handlers ────────────────────────────────────────

        connect_btn.click(
            fn=handle_connect,
            inputs=[master_url],
            outputs=[worker_state, connect_status],
        )

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

        refresh_btn.click(
            fn=handle_refresh, inputs=[worker_state], outputs=[projects_table]
        )
        history_refresh_btn.click(
            fn=handle_refresh, inputs=[worker_state], outputs=[history_table]
        )

        join_btn.click(
            fn=handle_join,
            inputs=[worker_state, join_project_id, worker_name_input, data_path],
            outputs=[active_project_id, workspace_status],
        )

        lock_btn.click(
            fn=handle_lock,
            inputs=[worker_state, active_project_id],
            outputs=[workspace_status],
        )

        start_btn.click(
            fn=handle_start,
            inputs=[worker_state, active_project_id],
            outputs=[workspace_status],
        )

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

        # ── CSV Downloads ──────────────────────────────────────────────

        summary_download.click(
            fn=lambda w, p: download_table_csv(w, p, "summary"),
            inputs=[worker_state, results_project_id],
            outputs=[summary_download],
        )
        baseline_cumhazards_download.click(
            fn=lambda w, p: download_table_csv(w, p, "baseline_cumhazards"),
            inputs=[worker_state, results_project_id],
            outputs=[baseline_cumhazards_download],
        )
        baseline_survival_download.click(
            fn=lambda w, p: download_table_csv(w, p, "baseline_survival"),
            inputs=[worker_state, results_project_id],
            outputs=[baseline_survival_download],
        )
        baseline_hazard_download.click(
            fn=lambda w, p: download_table_csv(w, p, "baseline_hazard"),
            inputs=[worker_state, results_project_id],
            outputs=[baseline_hazard_download],
        )
        history_summary_download.click(
            fn=lambda w, p: download_table_csv(w, p, "summary"),
            inputs=[worker_state, results_project_id],
            outputs=[history_summary_download],
        )
        history_baseline_cumhazards_download.click(
            fn=lambda w, p: download_table_csv(w, p, "baseline_cumhazards"),
            inputs=[worker_state, results_project_id],
            outputs=[history_baseline_cumhazards_download],
        )
        history_baseline_survival_download.click(
            fn=lambda w, p: download_table_csv(w, p, "baseline_survival"),
            inputs=[worker_state, results_project_id],
            outputs=[history_baseline_survival_download],
        )
        history_baseline_hazard_download.click(
            fn=lambda w, p: download_table_csv(w, p, "baseline_hazard"),
            inputs=[worker_state, results_project_id],
            outputs=[history_baseline_hazard_download],
        )

        # ── Plot Downloads ─────────────────────────────────────────────

        coef_plot_download.click(
            fn=lambda w, p: download_plot_path(w, p, "coefficients_plot"),
            inputs=[worker_state, results_project_id],
            outputs=[coef_plot_download],
        )
        history_coef_plot_download.click(
            fn=lambda w, p: download_plot_path(w, p, "history_coefficients_plot"),
            inputs=[worker_state, results_project_id],
            outputs=[history_coef_plot_download],
        )

        # ── History Fetch ──────────────────────────────────────────────

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
            return str(evt.value or "")

        history_table.select(fn=on_select_history, inputs=[], outputs=[history_pid])

        # ── Event Polling Timer ────────────────────────────────────────

        timer = gr.Timer(2)
        timer.tick(
            fn=handle_poll_events,
            inputs=[worker_state, active_project_id],
            outputs=[event_log_area],
        )

    blocks_app.theme = THEME
    blocks_app.css = CUSTOM_CSS
    return blocks_app


def create_app() -> FastAPI:
    """Create a FastAPI app to host the Gradio UI for uvicorn."""
    fastapi_app = FastAPI(title="DC-Cox Worker UI")
    gradio_app = create_ui()
    return gr.mount_gradio_app(fastapi_app, gradio_app, path="/")


# Module-level app for `uvicorn dccox.service.worker.ui:app`
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
