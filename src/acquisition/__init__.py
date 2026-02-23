"""Acquisition policies subpackage."""

from .policies import (
    distortion_loss,
    expected_information_gain,
    greedy_acquisition_policy
)

__all__ = [
    "distortion_loss",
    "expected_information_gain",
    "greedy_acquisition_policy"
]
