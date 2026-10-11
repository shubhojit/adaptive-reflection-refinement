"""Reusable runners for sanity checks and budget sweeps."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from reflection_refinement.baseline import initial_estimate
from reflection_refinement.metrics import mae, psnr
from reflection_refinement.routing import compare_policies, make_tiles
from reflection_refinement.synthetic import make_synthetic_scene

FIELDS = [
    "seed",
    "size",
    "reflection_strength",
    "budget_tiles",
    "policy",
    "selected_tiles",
    "updated_pixels",
    "processed_pixels",
    "initial_mae",
    "initial_psnr",
    "refined_mae",
    "refined_psnr",
    "seconds",
]
# Each result row is one seed x reflection strength x budget x routing policy.


def run_budget_sweep(
    image_count: int,
    size: int,
    tile_size: int,
    output_path: Path,
    budgets: list[int] | None = None,
    learning_rate: float = 0.2,
    lambda_exclusion: float = 100_000.0,
    reflection_strengths: list[float] | None = None,
) -> list[dict[str, float | int | str]]:
    """Run paired routing experiments and save all per-case results to CSV.

    Args:
        image_count: Number of deterministic synthetic seeds per reflection strength.
        size: Square image side length in pixels.
        tile_size: Side length of each square refinement tile.
        output_path: CSV destination; parent folders are created if needed.
        budgets: Optional tile counts to evaluate. Defaults to zero, approximately
            one-eighth, one-quarter, and all available tiles.
        learning_rate: Gradient-descent step size for each selected tile.
        lambda_exclusion: Weight on the gradient-exclusion loss.
        reflection_strengths: Mixture coefficients to evaluate. Defaults to ``[0.28]``.

    Returns:
        List of row dictionaries written to the CSV. Each row contains seed, scene
        settings, policy, work counts, initial/refined MAE and PSNR, and elapsed time.

    Raises:
        ValueError: If image count, tile geometry, budget, or reflection strengths are
            invalid.
    """
    if image_count < 1:
        raise ValueError("image_count must be positive.")
    tiles = make_tiles((size, size, 3), tile_size)
    if size % tile_size != 0:
        raise ValueError("size must be divisible by tile_size for an equal pixel budget.")
    if budgets is None:
        budgets = sorted({0, max(1, len(tiles) // 8), max(1, len(tiles) // 4), len(tiles)})
    if any(budget < 0 or budget > len(tiles) for budget in budgets):
        raise ValueError("all budgets must fit within the available tile count.")
    if reflection_strengths is None:
        reflection_strengths = [0.28]
    if any(not 0 <= strength <= 1 for strength in reflection_strengths):
        raise ValueError("reflection strengths must be between 0 and 1.")

    rows: list[dict[str, float | int | str]] = []
    for strength in reflection_strengths:
        for seed in range(image_count):
            # Keep the scene seed identical across policies and budgets for paired
            # comparisons; only its requested reflection strength changes.
            scene = make_synthetic_scene(
                seed=seed, size=size, reflection_strength=strength
            )
            observed = scene["observed"]
            target = scene["transmission"]
            assert isinstance(observed, np.ndarray) and isinstance(target, np.ndarray)
            initial = initial_estimate(observed)
            initial_mae = mae(initial, target)
            initial_psnr = psnr(initial, target)
            for budget in budgets:
                # Both policies receive the same initial estimate and update budget.
                results = compare_policies(
                    observed,
                    initial,
                    target,
                    tile_size=tile_size,
                    budget_tiles=budget,
                    learning_rate=learning_rate,
                    lambda_exclusion=lambda_exclusion,
                )
                for policy, result in results.items():
                    rows.append(
                        {
                            "seed": seed,
                            "size": size,
                            "reflection_strength": float(scene["reflection_strength"]),
                            "budget_tiles": budget,
                            "policy": policy,
                            "selected_tiles": int(result["selected_tiles"]),
                            "updated_pixels": int(result["updated_pixels"]),
                            "processed_pixels": int(result["processed_pixels"]),
                            "initial_mae": initial_mae,
                            "initial_psnr": initial_psnr,
                            "refined_mae": float(result["mae"]),
                            "refined_psnr": float(result["psnr"]),
                            "seconds": float(result["seconds"]),
                        }
                    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows
