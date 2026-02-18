"""Utility functions for dccox."""

from __future__ import annotations

import logging

from pydantic import ConfigDict, validate_call

from dccox.config import env


def validate_methods(cls: type) -> type:
    """
    Validate all methods in a class.

    This decorator validates all methods in a class using pydantic's validate_call.
    It is useful for validating method inputs and outputs.

    Parameters
    ----------
    cls : type
        The class to validate.

    Returns
    -------
    type
        The validated class.
    """
    if not env.enable_validation:
        return cls

    for name, method in cls.__dict__.items():
        if callable(method):
            if isinstance(method, staticmethod):
                logging.warning(f"Skipping validation for staticmethod: {name}")
                continue
            setattr(
                cls,
                name,
                validate_call(method, config=ConfigDict(arbitrary_types_allowed=True)),
            )
    return cls
