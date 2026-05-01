"""Shared argparse helpers for HQSB benchmark command-line interfaces.

These helpers centralize argument validation so that every public CLI
rejects malformed numeric inputs with a stable, diagnostic
``argparse.ArgumentTypeError`` (which argparse turns into a non-zero exit
with a usage message) instead of failing deep inside benchmark code.
"""

from __future__ import annotations

import argparse


def positive_int(value: str) -> int:
    """Parse a CLI argument as a strictly positive (>= 1) integer.

    Args:
        value: The raw string value received from argparse.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not an integer or is
            less than 1.
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from None

    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"must be a positive integer (>= 1), got {parsed}"
        )

    return parsed


def non_negative_int(value: str) -> int:
    """Parse a CLI argument as a non-negative (>= 0) integer.

    Args:
        value: The raw string value received from argparse.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not an integer or is
            negative.
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from None

    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"must be a non-negative integer (>= 0), got {parsed}"
        )

    return parsed
