#!/usr/bin/env python3
"""Generate one example and report initial/refined reconstruction metrics."""

from __future__ import annotations

import argparse

from reflection_refinement.baseline import initial_estimate
from reflection_refinement.metrics import mae, psnr
from reflection_refinement.routing import compare_policies, make_tiles
from reflection_refinement.synthetic import make_synthetic_scene


def main() -> None:
    """Run both routing policies on one generated scene and print their metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128, help="square image side in pixels")
    parser.add_argument("--tile-size", type=int, default=32, help="tile side in pixels")
    parser.add_argument("--seed", type=int, default=0, help="synthetic scene random seed")
    parser.add_argument(
        "--learning-rate", type=float, default=0.2, help="one-step refinement size"
    )
    parser.add_argument(
        "--lambda-exclusion",
        type=float,
        default=100_000.0,
        help="weight on the gradient-exclusion loss",
    )
    args = parser.parse_args()

    scene = make_synthetic_scene(seed=args.seed, size=args.size)
    # The synthetic generator returns a dict; these entries are RGB NumPy arrays.
    observed, target = scene["observed"], scene["transmission"]
    initial = initial_estimate(observed)
    tile_count = len(make_tiles(observed.shape, args.tile_size))
    # Use one quarter of all tiles so the check remains quick but nontrivial.
    budget = max(1, tile_count // 4)
    results = compare_policies(
        observed,
        initial,
        target,
        args.tile_size,
        budget,
        learning_rate=args.learning_rate,
        lambda_exclusion=args.lambda_exclusion,
    )

    print(f"CPU NumPy sanity check | size={args.size} | seed={args.seed}")
    print(
        f"Initial estimate: MAE={mae(initial, target):.5f}, "
        f"PSNR={psnr(initial, target):.2f} dB"
    )
    for policy, result in results.items():
        print(
            f"{policy:>8}: MAE={result['mae']:.5f}, PSNR={result['psnr']:.2f} dB, "
            f"tiles={result['selected_tiles']}, time={result['seconds']:.4f}s"
        )


if __name__ == "__main__":
    main()
