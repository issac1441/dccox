"""Gradio theme and CSS configuration for the DC-Cox worker UI."""

import gradio as gr

CUSTOM_CSS = """
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

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
)
