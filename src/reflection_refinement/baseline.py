"""A deliberately simple image-only initial estimate."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reflection_refinement.synthetic import gaussian_blur


def initial_estimate(
    observed: NDArray[np.float64],
    strength: float = 0.25,
    sigma: float = 5.0,
) -> NDArray[np.float64]:
    """Make a simple initial transmission estimate from an observed RGB image.

    The estimate subtracts a scaled blurred copy of the observation:
    ``clip(observed - strength * gaussian_blur(observed, sigma), 0, 1)``.
    It is intentionally weak and is used to give experiments a shared starting point,
    not as a competitive reflection-removal model.

    Args:
        observed: Float RGB image with shape ``(height, width, 3)`` and values in
            ``[0, 1]``.
        strength: Fraction of the low-frequency component to subtract.
        sigma: Standard deviation, in pixels, of the Gaussian blur.

    Returns:
        A new float RGB image with the same shape, clipped to ``[0, 1]``.

    Raises:
        ValueError: If the image shape or either parameter is invalid.
    """
    if observed.ndim != 3 or observed.shape[-1] != 3:
        raise ValueError("observed must be an HxWx3 RGB image.")
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1.")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative.")
    return np.clip(observed - strength * gaussian_blur(observed, sigma), 0, 1)
