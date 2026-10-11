#!/usr/bin/env python3
"""Compare routing policies across controlled reflection strengths."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from reflection_refinement.experiment import run_budget_sweep


def main() -> None:
    """Run all routing budgets at several reflection strengths and print means."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=int, default=10, help="synthetic scenes per strength"
    )
    parser.add_argument("--size", type=int, default=128, help="square image side in pixels")
    parser.add_argument("--tile-size", type=int, default=32, help="tile side in pixels")
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[0.1, 0.28, 0.45, 0.65],
        help="reflection mixture coefficients between 0 and 1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/strength_sweep.csv"),
        help="destination for per-image CSV rows",
    )
    args = parser.parse_args()

    rows = run_budget_sweep(
        image_count=args.images,
        size=args.size,
        tile_size=args.tile_size,
        output_path=args.output,
        reflection_strengths=args.strengths,
    )
    # Group detailed per-image measurements for a compact summary by condition.
    groups: dict[tuple[float, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        groups[(float(row["reflection_strength"]), str(row["policy"]))].append(row)

    print(f"Saved {len(rows)} rows to {args.output}")
    print("strength policy     initial MAE  refined MAE  PSNR(dB)")
    for (strength, policy), group in sorted(groups.items()):
        # Group rows include every budget, so these means summarize all tested budgets.
        initial_mae = sum(float(row["initial_mae"]) for row in group) / len(group)
        refined_mae = sum(float(row["refined_mae"]) for row in group) / len(group)
        refined_psnr = sum(float(row["refined_psnr"]) for row in group) / len(group)
        print(
            f"{strength:>8.2f} {policy:>8} {initial_mae:>12.5f} "
            f"{refined_mae:>12.5f} {refined_psnr:>9.3f}"
        )


if __name__ == "__main__":
    main()
