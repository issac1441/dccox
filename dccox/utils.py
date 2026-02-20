"""Utility functions for dccox."""

from __future__ import annotations

import logging
from typing import Self, get_type_hints

from pydantic import ConfigDict, validate_call

from dccox.config import env


def _returns_self(method: callable) -> bool:
    """Check if a method's return annotation contains Self."""
    try:
        hints = get_type_hints(method)
        return_hint = hints.get("return")
        if return_hint is Self:
            return True
    except Exception:
        pass
    return False


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

            # Disable return validation for methods returning Self (Pydantic doesn't support it)
            validate_return = not _returns_self(method)

            setattr(
                cls,
                name,
                validate_call(
                    method,
                    config=ConfigDict(
                        arbitrary_types_allowed=True,
                        validate_return=validate_return,
                    ),
                ),
            )
    return cls
