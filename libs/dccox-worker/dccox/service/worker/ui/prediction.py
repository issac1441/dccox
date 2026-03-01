"""Prediction method configuration for the DC-Cox worker UI.

Single Responsibility: Defines prediction method metadata used by both
the UI layout and handler logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionMethodSpec:
    """Specification for a single prediction method."""

    name: str
    label: str
    requires_time: bool
    requires_features: bool


PREDICTION_METHODS: list[PredictionMethodSpec] = [
    PredictionMethodSpec("hazard_at", "hazard_at (baseline hazard at t)", True, False),
    PredictionMethodSpec(
        "cumhazard_at", "cumhazard_at (baseline cumulative hazard at t)", True, False
    ),
    PredictionMethodSpec(
        "survival_at", "survival_at (baseline survival at t)", True, False
    ),
    PredictionMethodSpec(
        "predict_log_partial_hazard",
        "predict_log_partial_hazard (log hazard, features only)",
        False,
        True,
    ),
    PredictionMethodSpec(
        "predict_partial_hazard",
        "predict_partial_hazard (hazard ratio, features only)",
        False,
        True,
    ),
    PredictionMethodSpec(
        "predict_hazard_at", "predict_hazard_at (hazard at t)", True, True
    ),
    PredictionMethodSpec(
        "predict_cumhazard_at",
        "predict_cumhazard_at (cumulative hazard at t)",
        True,
        True,
    ),
    PredictionMethodSpec(
        "predict_survival_at",
        "predict_survival_at (survival probability at t)",
        True,
        True,
    ),
    PredictionMethodSpec(
        "predict_hazard", "predict_hazard (hazard over all times)", False, True
    ),
    PredictionMethodSpec(
        "predict_cumhazard",
        "predict_cumhazard (cumulative hazard over all times)",
        False,
        True,
    ),
    PredictionMethodSpec(
        "predict_survival", "predict_survival (survival curve)", False, True
    ),
    PredictionMethodSpec(
        "predict_expectation",
        "predict_expectation (expected survival time)",
        False,
        True,
    ),
]

PREDICTION_METHOD_MAP: dict[str, PredictionMethodSpec] = {
    m.name: m for m in PREDICTION_METHODS
}

PREDICTION_OPTIONS: list[tuple[str, str]] = [
    (m.label, m.name) for m in PREDICTION_METHODS
]

PREDICTION_DEFAULT_METHOD = "predict_survival_at"
