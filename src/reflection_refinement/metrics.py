"""Reference-based metrics for the synthetic teaching experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mae(prediction: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    """Return mean absolute pixel error between equal-shaped image arrays.

    Args:
        prediction: Estimated image.
        target: Reference image, usually the synthetic transmission layer.

    Returns:
        Mean absolute error across all pixels and channels. Lower is better.
    """
    return float(np.mean(np.abs(prediction - target)))


def psnr(prediction: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    """Return peak signal-to-noise ratio assuming a unit image range.

    Args:
        prediction: Estimated image with values represented on a unit scale.
        target: Reference image on the same scale and with the same shape.

    Returns:
        PSNR in decibels. Higher is better; identical arrays return infinity.
    """
    mse = float(np.mean((prediction - target) ** 2))
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(1.0 / mse))
