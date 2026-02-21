"""Pydantic schemas for the DC-Cox federated service."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ProjectSchema(BaseModel):
    """Configuration for a DC-Cox analysis project."""

    name: str = Field(..., description="Project name")
    use_case: Literal["horizontal"] = Field(
        default="horizontal",
        description="Use case type. Currently only 'horizontal' is supported.",
    )

    # Projection hyperparameters
    k: int = Field(default=20, ge=1, description="Latent dimension of Fdr")
    r: int = Field(default=100, ge=10, description="Pseudo-number of anchor samples")
    bs_prop: float = Field(
        default=0.6, gt=0, le=1, description="Bootstrap sample proportion"
    )
    bs_times: int = Field(
        default=20, ge=1, description="Number of bootstrap iterations"
    )
    bs_replace: bool = Field(
        default=False, description="Whether to sample with replacement"
    )

    # Regression hyperparameters
    alpha: float = Field(default=0.05, gt=0, lt=1, description="Significance level")
    step_size: float = Field(default=0.5, gt=0, description="Step size for convergence")
    var_thres: float = Field(
        default=1e-8, gt=0, description="Low-variance column threshold"
    )
    m_hat: int | None = Field(
        default=None, description="Dimension of transformed feature space"
    )

    # Data options
    centering: Literal["mean"] | None = Field(
        default=None,
        description="Centering strategy: None (paper) or 'mean' (lifelines)",
    )
    keep_feature_cols: list[str] | None = Field(
        default=["MYCN", "ALK", "TP53", "PHOX2B"],
        description="Feature columns to use (auto-detect if None)",
    )
    meta_cols: list[str] | None = Field(
        default=["sample-id"],
        description="Metadata columns to preserve",
    )


class WorkerJoinRequest(BaseModel):
    """Request payload when a worker joins a project."""

    worker_name: str = Field(..., description="Human-readable worker name")
    n_features: int = Field(..., ge=1, description="Number of feature columns")


class WorkerInfo(BaseModel):
    """Information about a registered worker."""

    worker_id: str
    worker_name: str
    n_features: int
    has_submitted_proxy: bool = False


class ProjectStatus(StrEnum):
    """Project lifecycle status."""

    CREATED = "created"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class ProjectDetail(BaseModel):
    """Full project details for API responses."""

    id: str
    config: ProjectSchema
    status: ProjectStatus
    workers: list[WorkerInfo]
    n_features: int | None = None
    error: str | None = None
    created_at: str


class ProjectCreateResponse(BaseModel):
    """Response after creating a project."""

    id: str
    name: str
    status: ProjectStatus


class ProxyDataPayload(BaseModel):
    """Proxy data submitted by a worker after local computation."""

    x_tilde: list[list[float]] | None = None
    xanc_tilde: list[list[float]] | None = None
    y: list[list[float]]
    feature_sum: list[float]


class WorkerJoinResponse(BaseModel):
    """Response after a worker joins a project."""

    worker_id: str
    total_workers: int


class WorkerResultResponse(BaseModel):
    """Per-worker results returned by the master."""

    coef: list[float]
    coef_var: list[list[float]]
    baseline_hazard: dict
    feature_mean: list[float]
