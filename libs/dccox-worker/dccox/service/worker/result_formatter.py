"""Result formatting utilities for DC-Cox worker outputs.

Single Responsibility: Transform raw SurvivalFunction results into
presentation-ready DataFrames.
"""

from __future__ import annotations

import pandas as pd

from dccox.cox import SurvivalFunction


def build_result_tables(surv: SurvivalFunction) -> dict[str, pd.DataFrame]:
    """Build formatted summary and baseline tables from a SurvivalFunction."""
    return {
        "summary": _format_summary(surv.summary),
        "baseline_cumhazards": _format_baseline_table(
            surv.baseline_cumhazards, "baseline cumhazards"
        ),
        "baseline_survival": _format_baseline_table(
            surv.baseline_survival, "baseline survival"
        ),
        "baseline_hazard": _format_baseline_table(
            surv.baseline_hazard, "baseline hazard"
        ),
    }


def _format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Reset index and normalize the feature column name."""
    summary = summary.reset_index()
    summary.rename(columns={summary.columns[0]: "feature"}, inplace=True)
    summary["feature"] = summary["feature"].apply(_format_feature)
    return summary


def _format_baseline_table(table: pd.DataFrame, value_label: str) -> pd.DataFrame:
    """Reset index and rename columns for baseline metric tables."""
    formatted = table.reset_index()
    time_col = formatted.columns[0]
    formatted.rename(columns={time_col: "timeline"}, inplace=True)
    value_cols = [col for col in formatted.columns if col != "timeline"]
    if value_cols:
        formatted.rename(columns={value_cols[0]: value_label}, inplace=True)
    return formatted


def _format_feature(value: pd.Index | tuple[str] | list[str]) -> str:
    """Normalize multi-level feature names to a flat string."""
    if isinstance(value, pd.Index):
        parts = value.tolist()
    elif isinstance(value, (tuple, list)):
        parts = value
    else:
        return str(value)
    return " + ".join(str(part) for part in parts)
