"""CPU-only teaching experiments for spatially adaptive reflection refinement."""

from reflection_refinement.baseline import initial_estimate
from reflection_refinement.refinement import refine_tile
from reflection_refinement.routing import compare_policies
from reflection_refinement.synthetic import make_synthetic_scene

__all__ = [
    "compare_policies",
    "initial_estimate",
    "make_synthetic_scene",
    "refine_tile",
]
