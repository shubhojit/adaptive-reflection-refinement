"""Fixed-budget adaptive and uniform spatial tile allocation."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from reflection_refinement.metrics import mae, psnr
from reflection_refinement.refinement import refine_tile

Image = NDArray[np.float64]
Policy = Literal["adaptive", "uniform"]


def make_tiles(
    image_shape: tuple[int, int, int], tile_size: int
) -> list[tuple[int, int, int, int]]:
    """Partition an image into non-overlapping tiles.

    Args:
        image_shape: Channel-last image shape ``(height, width, channels)``.
        tile_size: Maximum tile height and width in pixels.

    Returns:
        Row-major list of ``(y_start, y_stop, x_start, x_stop)`` coordinates. Stop
        indices are exclusive; tiles at the image boundary may be smaller.
    """
    height, width, _ = image_shape
    if tile_size < 4:
        raise ValueError("tile_size must be at least 4.")
    return [
        (y, min(y + tile_size, height), x, min(x + tile_size, width))
        for y in range(0, height, tile_size)
        for x in range(0, width, tile_size)
    ]


def tile_scores(
    initial: Image, observed: Image, tiles: list[tuple[int, int, int, int]]
) -> NDArray[np.float64]:
    """Calculate mean absolute gradient-overlap score for each tile.

    Args:
        initial: Initial transmission estimate ``T0``.
        observed: Mixed image ``I``.
        tiles: Regions in the coordinate format returned by :func:`make_tiles`.

    Returns:
        One nonnegative score per tile, in the same order as ``tiles``. Larger scores
        indicate more gradient overlap between ``T0`` and ``I - T0``; they are a
        heuristic for routing, not ground-truth reflection labels.
    """
    residual = observed - initial
    tx = np.diff(initial, axis=1, append=initial[:, -1:, :])
    ty = np.diff(initial, axis=0, append=initial[-1:, :, :])
    rx = np.diff(residual, axis=1, append=residual[:, -1:, :])
    ry = np.diff(residual, axis=0, append=residual[-1:, :, :])
    overlap = np.sum(tx * rx + ty * ry, axis=-1)
    return np.asarray(
        [float(np.mean(np.abs(overlap[y0:y1, x0:x1]))) for y0, y1, x0, x1 in tiles]
    )


def select_tiles(scores: NDArray[np.float64], count: int, policy: Policy) -> list[int]:
    """Choose tile indices for one refinement budget.

    Args:
        scores: One routing score per available tile.
        count: Number of tiles to select.
        policy: ``"adaptive"`` selects largest scores; ``"uniform"`` samples indices
            evenly across the row-major tile list without consulting score values.

    Returns:
        Selected tile indices. Adaptive results are ordered highest-score first.

    Raises:
        ValueError: If ``count`` is out of range or ``policy`` is unsupported.
    """
    if not 0 <= count <= len(scores):
        raise ValueError("count must be between zero and the number of tiles.")
    if policy == "adaptive":
        return np.argsort(-scores, kind="stable")[:count].tolist()
    if policy != "uniform":
        raise ValueError(f"unknown routing policy: {policy}")
    if count == 0:
        return []
    return np.rint(np.linspace(0, len(scores) - 1, count)).astype(int).tolist()


def run_policy(
    observed: Image,
    initial: Image,
    target: Image,
    tile_size: int,
    budget_tiles: int,
    policy: Policy,
    learning_rate: float = 0.2,
    lambda_fidelity: float = 1.0,
    lambda_exclusion: float = 100_000.0,
) -> dict[str, float | int | Image]:
    """Run one routing policy and evaluate its refined output.

    Args:
        observed: Mixed RGB image ``I``.
        initial: Common initial transmission estimate ``T0``.
        target: Clean synthetic transmission used only for evaluation, never routing.
        tile_size: Side length for square tiles (edge tiles may be smaller).
        budget_tiles: Number of tiles to update.
        policy: Adaptive or uniform tile-selection strategy.
        learning_rate: Step size passed to each tile update.
        lambda_fidelity: Weight on distance from ``initial``.
        lambda_exclusion: Weight on gradient overlap.

    Returns:
        A dictionary with:
        - ``output``: refined RGB image array.
        - ``selected_tiles``: number of tiles updated.
        - ``updated_pixels``: pixels in tile interiors that were changed.
        - ``processed_pixels``: pixels processed including one-pixel halos.
        - ``mae`` and ``psnr``: metrics against ``target``.
        - ``seconds``: elapsed wall-clock time including routing and refinement.
    """
    started = time.perf_counter()
    tiles = make_tiles(observed.shape, tile_size)
    scores = (
        tile_scores(initial, observed, tiles)
        if policy == "adaptive"
        else np.zeros(len(tiles), dtype=np.float64)
    )
    selected = select_tiles(scores, budget_tiles, policy)
    output = initial.copy()
    for tile_index in selected:
        output = refine_tile(
            output,
            observed,
            initial,
            tiles[tile_index],
            learning_rate=learning_rate,
            lambda_fidelity=lambda_fidelity,
            lambda_exclusion=lambda_exclusion,
        )
    elapsed = time.perf_counter() - started
    return {
        "output": output,
        "selected_tiles": len(selected),
        "updated_pixels": sum(
            (y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in (tiles[i] for i in selected)
        ),
        "processed_pixels": sum(
            (y1 - y0 + 2) * (x1 - x0 + 2)
            for y0, y1, x0, x1 in (tiles[i] for i in selected)
        ),
        "mae": mae(output, target),
        "psnr": psnr(output, target),
        "seconds": elapsed,
    }


def compare_policies(
    observed: Image,
    initial: Image,
    target: Image,
    tile_size: int,
    budget_tiles: int,
    learning_rate: float = 0.2,
    lambda_fidelity: float = 1.0,
    lambda_exclusion: float = 100_000.0,
) -> dict[str, dict[str, float | int | Image]]:
    """Run adaptive and uniform policies under a shared tile-count budget.

    Args:
        observed: Mixed RGB image ``I``.
        initial: Common initial transmission estimate ``T0``.
        target: Clean transmission used only for scoring returned results.
        tile_size: Tile side length in pixels.
        budget_tiles: Number of tiles each policy is allowed to update.
        learning_rate: Step size used for each local gradient-descent update.
        lambda_fidelity: Weight on the estimate-fidelity term.
        lambda_exclusion: Weight on the gradient-exclusion term.

    Returns:
        Mapping with ``"adaptive"`` and ``"uniform"`` keys. Each value is the result
        dictionary returned by :func:`run_policy`.
    """
    options = {
        "learning_rate": learning_rate,
        "lambda_fidelity": lambda_fidelity,
        "lambda_exclusion": lambda_exclusion,
    }
    return {
        "adaptive": run_policy(
            observed, initial, target, tile_size, budget_tiles, "adaptive", **options
        ),
        "uniform": run_policy(
            observed, initial, target, tile_size, budget_tiles, "uniform", **options
        ),
    }
