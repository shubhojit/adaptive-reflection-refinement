"""Small deterministic synthetic transmission/reflection mixtures."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.float64]


def gaussian_blur(image: Image, sigma: float) -> Image:
    """Apply a separable Gaussian blur using NumPy only.

    Args:
        image: Two-dimensional or channel-last array to blur on its first two axes.
        sigma: Gaussian standard deviation in pixels. A nonpositive value disables
            blurring and returns a copy.

    Returns:
        Blurred array with the same shape as ``image``. Reflective padding avoids
        introducing a constant dark border.
    """
    if sigma <= 0:
        return image.copy()
    radius = max(1, int(np.ceil(3 * sigma)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(coordinates**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    blurred = image
    for axis in (0, 1):
        # Pad and convolve one spatial axis at a time; separability avoids a 2-D kernel.
        padded = np.pad(
            blurred,
            [(radius, radius) if dim == axis else (0, 0) for dim in range(3)],
            mode="reflect",
        )
        blurred = sum(
            weight
            * np.take(padded, range(offset, offset + blurred.shape[axis]), axis=axis)
            for offset, weight in enumerate(kernel)
        )
    return blurred


def _draw_rectangles(image: Image, rng: np.random.Generator, count: int) -> None:
    """Draw random filled RGB rectangles in-place on ``image``."""
    height, width, _ = image.shape
    for _ in range(count):
        y0 = int(rng.integers(0, max(1, height - 8)))
        x0 = int(rng.integers(0, max(1, width - 8)))
        rect_h = int(rng.integers(5, max(6, height // 3)))
        rect_w = int(rng.integers(5, max(6, width // 3)))
        color = rng.uniform(0.05, 0.95, size=(1, 1, 3))
        image[y0 : min(height, y0 + rect_h), x0 : min(width, x0 + rect_w)] = color


def make_synthetic_scene(
    seed: int,
    size: int = 128,
    reflection_strength: float = 0.28,
) -> dict[str, Image | float | int]:
    """Create a reproducible RGB transmission/reflection mixture.

    The transmission contains a smooth color gradient and sharp rectangles. The
    reflection contains colored stripes and rectangles, then receives blur and a
    circular spatial shift before being blended with the transmission.

    Args:
        seed: Random seed controlling rectangle placement, colors, and stripe phase.
        size: Height and width of the square RGB images. Must be at least 32.
        reflection_strength: Mixture coefficient in ``[0, 1]``; larger values give the
            reflection more influence in the observed image.

    Returns:
        A dictionary containing ``transmission``, ``reflection``, and ``observed``
        float arrays of shape ``(size, size, 3)`` and values in ``[0, 1]``, along with
        the ``reflection_strength`` and ``seed`` used to make the scene.

    Raises:
        ValueError: If the image size or mixture coefficient is invalid.
    """
    if size < 32:
        raise ValueError("size must be at least 32 pixels.")
    if not 0 <= reflection_strength <= 1:
        raise ValueError("reflection_strength must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.stack(
        (
            xx / (size - 1),
            yy / (size - 1),
            (xx + yy) / (2 * (size - 1)),
        ),
        axis=-1,
    ).astype(np.float64)
    transmission = 0.18 + 0.55 * base
    _draw_rectangles(transmission, rng, count=max(3, size // 24))
    transmission = np.clip(transmission, 0, 1)

    reflection = np.zeros_like(transmission)
    stripe_phase = rng.uniform(0, 2 * np.pi)
    stripe_period = rng.uniform(size / 9, size / 4)
    stripe_pattern = (np.sin((xx + yy * 0.35) * 2 * np.pi / stripe_period + stripe_phase) + 1) / 2
    reflection[..., 0] = stripe_pattern
    reflection[..., 1] = np.roll(stripe_pattern, size // 13, axis=1)
    reflection[..., 2] = np.flip(stripe_pattern, axis=0)
    _draw_rectangles(reflection, rng, count=max(2, size // 40))
    reflection = gaussian_blur(reflection, sigma=max(1.0, size / 64))
    # Circular shift is a simple, deterministic stand-in for layer misalignment.
    reflection = np.roll(reflection, shift=(size // 17, -size // 19), axis=(0, 1))
    reflection = np.clip(reflection, 0, 1)

    observed = np.clip(
        (1 - reflection_strength) * transmission + reflection_strength * reflection,
        0,
        1,
    )
    return {
        "transmission": transmission,
        "reflection": reflection,
        "observed": observed,
        "reflection_strength": reflection_strength,
        "seed": seed,
    }
