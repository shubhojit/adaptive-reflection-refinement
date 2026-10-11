"""Local gradient-exclusion refinement and its analytic gradient."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.float64]


def _forward_difference(image: Image, axis: int) -> Image:
    """Compute forward finite differences along one spatial axis.

    The final row or column is zero because there is no forward neighbor at that edge.
    """
    difference = np.zeros_like(image)
    if axis == 0:
        difference[:-1] = image[1:] - image[:-1]
    else:
        difference[:, :-1] = image[:, 1:] - image[:, :-1]
    return difference


def _difference_adjoint(values: Image, axis: int) -> Image:
    """Apply the transpose (adjoint) of ``_forward_difference``."""
    result = np.zeros_like(values)
    if axis == 0:
        result[0] -= values[0]
        result[1:-1] += values[:-2] - values[1:-1]
        result[-1] += values[-2]
    else:
        result[:, 0] -= values[:, 0]
        result[:, 1:-1] += values[:, :-2] - values[:, 1:-1]
        result[:, -1] += values[:, -2]
    return result


def loss_and_gradient(
    transmission: Image,
    observed: Image,
    initial: Image,
    lambda_fidelity: float = 1.0,
    lambda_exclusion: float = 0.2,
) -> tuple[float, Image]:
    """Evaluate the fidelity-plus-exclusion objective and its analytic gradient.

    The exclusion term penalizes co-located, similarly oriented gradients in ``T`` and
    the residual ``I - T``. The returned gradient is with respect to every value in
    ``transmission`` and is used by the local gradient-descent update.

    Args:
        transmission: Current candidate transmission, shape ``(H, W, 3)``.
        observed: Mixed input image ``I``, same shape as ``transmission``.
        initial: Fixed starting estimate ``T0``, same shape as ``transmission``.
        lambda_fidelity: Nonnegative weight on mean squared distance from ``initial``.
        lambda_exclusion: Nonnegative weight on squared gradient overlap.

    Returns:
        A pair ``(loss, gradient)``. ``loss`` is a Python float; ``gradient`` is a
        float array with the same shape as ``transmission``.

    Raises:
        ValueError: If shapes, channel count, or loss weights are invalid.
    """
    if transmission.shape != observed.shape or transmission.shape != initial.shape:
        raise ValueError("transmission, observed, and initial must have equal shapes.")
    if transmission.ndim != 3 or transmission.shape[-1] != 3:
        raise ValueError("image arrays must have shape HxWx3.")
    if lambda_fidelity < 0 or lambda_exclusion < 0:
        raise ValueError("loss weights must be nonnegative.")

    height, width, channels = transmission.shape
    fidelity_scale = 1.0 / (height * width * channels)
    exclusion_scale = 1.0 / (height * width)

    # Since R = I - T, its derivatives are the observed derivatives minus T's.
    tx = _forward_difference(transmission, axis=1)
    ty = _forward_difference(transmission, axis=0)
    ix = _forward_difference(observed, axis=1)
    iy = _forward_difference(observed, axis=0)
    rx, ry = ix - tx, iy - ty
    # Sum x/y gradient products over color channels at each spatial location.
    overlap = np.sum(tx * rx + ty * ry, axis=-1)

    fidelity = float(np.mean((transmission - initial) ** 2))
    exclusion = float(np.mean(overlap**2))
    loss = lambda_fidelity * fidelity + lambda_exclusion * exclusion

    fidelity_gradient = (
        2 * lambda_fidelity * fidelity_scale * (transmission - initial)
    )
    overlap_gradient = 2 * lambda_exclusion * exclusion_scale * overlap[..., None]
    # d/dT [DT * (DI - DT)] = DI - 2*DT; apply difference adjoints to return
    # derivatives from edge space to pixel space.
    gx = overlap_gradient * (ix - 2 * tx)
    gy = overlap_gradient * (iy - 2 * ty)
    exclusion_gradient = _difference_adjoint(gx, axis=1) + _difference_adjoint(
        gy, axis=0
    )
    return loss, fidelity_gradient + exclusion_gradient


def refine_tile(
    transmission: Image,
    observed: Image,
    initial: Image,
    tile: tuple[int, int, int, int],
    learning_rate: float = 0.2,
    lambda_fidelity: float = 1.0,
    lambda_exclusion: float = 100_000.0,
) -> Image:
    """Apply one projected gradient-descent update inside a single tile.

    A one-pixel context halo supplies neighbors for finite differences. Halo values
    are reflect-padded at image boundaries. Pixels outside the tile are not changed.

    Args:
        transmission: Current full image estimate, shape ``(H, W, 3)``.
        observed: Mixed input image, same shape as ``transmission``.
        initial: Fixed estimate used by the fidelity term, same shape as input.
        tile: ``(y_start, y_stop, x_start, x_stop)`` with exclusive stop indices.
        learning_rate: Positive gradient-descent step size.
        lambda_fidelity: Nonnegative fidelity-term weight.
        lambda_exclusion: Nonnegative gradient-exclusion weight.

    Returns:
        A new full-size image. Values in the selected tile are updated and clipped to
        ``[0, 1]``; all other pixels are unchanged.

    Raises:
        ValueError: If the learning rate or tile coordinates are invalid.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    y0, y1, x0, x1 = tile
    height, width, _ = transmission.shape
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError("tile coordinates must be a non-empty in-bounds region.")

    crop_y0, crop_y1 = max(0, y0 - 1), min(height, y1 + 1)
    crop_x0, crop_x1 = max(0, x0 - 1), min(width, x1 + 1)
    # Always form a tile-plus-halo crop. At image edges, pad back to the same context
    # width so equally sized tiles incur the same number of processed pixels.
    padding = (
        (max(0, 1 - y0), max(0, y1 + 1 - height)),
        (max(0, 1 - x0), max(0, x1 + 1 - width)),
        (0, 0),
    )
    crop = (slice(crop_y0, crop_y1), slice(crop_x0, crop_x1))
    local_observed = np.pad(observed[crop], padding, mode="reflect")
    local_initial = np.pad(initial[crop], padding, mode="reflect")
    local_transmission = np.pad(transmission[crop], padding, mode="reflect")
    _, gradient = loss_and_gradient(
        local_transmission,
        local_observed,
        local_initial,
        lambda_fidelity=lambda_fidelity,
        lambda_exclusion=lambda_exclusion,
    )

    updated = transmission.copy()
    # The center of the halo-padded gradient corresponds to the tile interior.
    local_tile = (slice(1, y1 - y0 + 1), slice(1, x1 - x0 + 1))
    updated[y0:y1, x0:x1] = np.clip(
        transmission[y0:y1, x0:x1] - learning_rate * gradient[local_tile],
        0,
        1,
    )
    return updated
