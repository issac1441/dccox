"""Shared utility functions for the DC-Cox worker UI.

Single Responsibility: Formatting, temp file creation, and plotting helpers
used by the UI handlers. No Gradio dependencies.
"""

from __future__ import annotations

import tempfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Temp File Helpers ───────────────────────────────────────────────────


def create_temp_file(
    data: str | bytes, *, prefix: str, suffix: str, binary: bool = False
) -> str:
    """Write data to a named temp file and return its path."""
    safe_prefix = (prefix or "export").replace(" ", "_")
    mode = "wb" if binary else "w"
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        prefix=f"{safe_prefix}_",
        mode=mode,
    ) as tmp:
        if binary:
            tmp.write(data.encode("utf-8") if isinstance(data, str) else data)
        else:
            tmp.write(data.decode("utf-8") if isinstance(data, bytes) else data)
        return tmp.name


def create_temp_csv(content: str, prefix: str) -> str:
    """Write CSV content to a temp file."""
    return create_temp_file(content, prefix=prefix, suffix=".csv")


# ── DataFrame Helpers ───────────────────────────────────────────────────


def default_prediction_df(headers: list[str]) -> pd.DataFrame | None:
    """Create a zero-filled DataFrame for the prediction input form."""
    if not headers:
        return None
    return pd.DataFrame([dict.fromkeys(headers, 0.0)])


def prepare_feature_df(value: pd.DataFrame | None, headers: list[str]) -> pd.DataFrame:
    """Validate and reorder a user-provided feature DataFrame."""
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


def format_prediction_output(
    result: pd.DataFrame | pd.Series | np.ndarray | float, method: str
) -> pd.DataFrame:
    """Normalize any prediction output into a DataFrame."""
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


# ── Plot Helpers ────────────────────────────────────────────────────────


def build_coef_plot(summary: pd.DataFrame) -> plt.Figure | None:
    """Create a coefficient forest plot from a summary DataFrame."""
    if summary is None or summary.empty:
        return None
    if "coef" not in summary.columns:
        return None

    df = summary.copy()
    features = (
        df["feature"].astype(str) if "feature" in df.columns else df.index.astype(str)
    )
    df = df.assign(_feature=features)
    df = df.sort_values("coef")

    lower_col = next((c for c in df.columns if c.startswith("coef lower")), None)
    upper_col = next((c for c in df.columns if c.startswith("coef upper")), None)
    if lower_col is None or upper_col is None:
        return None

    fig, ax = plt.subplots(figsize=(8, max(4.0, min(10.0, 1.5 + 0.35 * len(df)))))
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


# ── Tuple Builders for Gradio Outputs ───────────────────────────────────


def workspace_empty_outputs(
    message: str, pid: str | None, headers: list[str]
) -> tuple[Any, ...]:
    """Build an all-empty output tuple for the workspace run handler."""
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


def history_empty_outputs(
    message: str, pid: str | None, headers: list[str]
) -> tuple[Any, ...]:
    """Build an all-empty output tuple for the history fetch handler."""
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
