#!/usr/bin/env python3
"""Compare adaptive and uniform allocation over equal tile-update budgets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from reflection_refinement.experiment import run_budget_sweep


def main() -> None:
    """Run the budget sweep and print mean metrics for each budget and policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=int, default=20, help="synthetic scenes to average per budget"
    )
    parser.add_argument("--size", type=int, default=128, help="square image side in pixels")
    parser.add_argument("--tile-size", type=int, default=32, help="tile side in pixels")
    parser.add_argument(
        "--learning-rate", type=float, default=0.2, help="one-step update size"
    )
    parser.add_argument(
        "--lambda-exclusion",
        type=float,
        default=100_000.0,
        help="weight on the gradient-exclusion loss",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/budget_sweep.csv"),
        help="destination for per-image CSV rows",
    )
    args = parser.parse_args()

    rows = run_budget_sweep(
        image_count=args.images,
        size=args.size,
        tile_size=args.tile_size,
        output_path=args.output,
        learning_rate=args.learning_rate,
        lambda_exclusion=args.lambda_exclusion,
    )
    # Each CSV row represents one scene, budget, and policy; aggregate rows here
    # for a concise console summary while preserving the detailed CSV.
    groups: dict[tuple[int, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["budget_tiles"]), str(row["policy"]))].append(row)

    print(f"Saved {len(rows)} rows to {args.output}")
    print("budget policy     MAE       PSNR(dB)  time(ms)")
    for (budget, policy), group in sorted(groups.items()):
        # Average per-image values; timings include routing and local refinement.
        mean_mae = sum(float(row["refined_mae"]) for row in group) / len(group)
        mean_psnr = sum(float(row["refined_psnr"]) for row in group) / len(group)
        mean_time = 1000 * sum(float(row["seconds"]) for row in group) / len(group)
        print(
            f"{budget:>6} {policy:>8} {mean_mae:>9.5f} "
            f"{mean_psnr:>9.3f} {mean_time:>9.3f}"
        )


if __name__ == "__main__":
    main()
