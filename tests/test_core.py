from __future__ import annotations

import unittest

import numpy as np

from reflection_refinement.baseline import initial_estimate
from reflection_refinement.refinement import loss_and_gradient, refine_tile
from reflection_refinement.routing import compare_policies, make_tiles, select_tiles
from reflection_refinement.synthetic import make_synthetic_scene


class SyntheticTests(unittest.TestCase):
    def test_generation_is_deterministic_and_bounded(self) -> None:
        """The same seed should reproduce valid unit-range RGB scene layers."""
        first = make_synthetic_scene(seed=4, size=64)
        second = make_synthetic_scene(seed=4, size=64)
        np.testing.assert_array_equal(first["observed"], second["observed"])
        for key in ("transmission", "reflection", "observed"):
            image = first[key]
            self.assertEqual(image.shape, (64, 64, 3))
            self.assertTrue(np.all((image >= 0) & (image <= 1)))


class RefinementTests(unittest.TestCase):
    def test_zero_exclusion_preserves_initial_at_stationary_point(self) -> None:
        """With no exclusion term, an estimate equal to its anchor has zero gradient."""
        rng = np.random.default_rng(2)
        observed = rng.random((12, 13, 3))
        initial = observed.copy()
        loss, gradient = loss_and_gradient(
            initial, observed, initial, lambda_exclusion=0
        )
        self.assertEqual(loss, 0)
        np.testing.assert_array_equal(gradient, np.zeros_like(initial))

    def test_analytic_gradient_matches_finite_difference(self) -> None:
        """Compare one analytic derivative with a central finite-difference estimate."""
        rng = np.random.default_rng(3)
        transmission = rng.random((5, 6, 3))
        observed = rng.random((5, 6, 3))
        initial = rng.random((5, 6, 3))
        _, gradient = loss_and_gradient(transmission, observed, initial)

        epsilon = 1e-6
        index = (2, 3, 1)
        plus, minus = transmission.copy(), transmission.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        loss_plus = loss_and_gradient(plus, observed, initial)[0]
        loss_minus = loss_and_gradient(minus, observed, initial)[0]
        numeric = (loss_plus - loss_minus) / (2 * epsilon)
        self.assertAlmostEqual(float(gradient[index]), numeric, places=7)

    def test_tile_refinement_changes_only_selected_tile(self) -> None:
        """A local update leaves pixels outside the selected tile unchanged."""
        scene = make_synthetic_scene(seed=1, size=32)
        observed = scene["observed"]
        initial = initial_estimate(observed)
        result = refine_tile(
            initial,
            observed,
            initial,
            (8, 16, 8, 16),
            lambda_exclusion=100_000,
        )
        outside = np.ones((32, 32), dtype=bool)
        outside[8:16, 8:16] = False
        np.testing.assert_array_equal(result[outside], initial[outside])
        self.assertTrue(np.all((result >= 0) & (result <= 1)))


class RoutingTests(unittest.TestCase):
    def test_selection_is_bounded_and_uniform_is_spread(self) -> None:
        """Adaptive ranking and uniform sampling choose the expected tile indices."""
        scores = np.arange(16, dtype=float)
        adaptive = select_tiles(scores, 4, "adaptive")
        uniform = select_tiles(scores, 4, "uniform")
        self.assertEqual(adaptive, [15, 14, 13, 12])
        self.assertEqual(uniform, [0, 5, 10, 15])

    def test_policies_use_equal_budget(self) -> None:
        """Both policies update the same tile interiors and process equal halos."""
        scene = make_synthetic_scene(seed=5, size=32)
        observed, target = scene["observed"], scene["transmission"]
        initial = initial_estimate(observed)
        results = compare_policies(observed, initial, target, 8, 3)
        self.assertEqual(results["adaptive"]["selected_tiles"], 3)
        self.assertEqual(results["uniform"]["selected_tiles"], 3)
        self.assertEqual(results["adaptive"]["updated_pixels"], 3 * 8 * 8)
        self.assertEqual(results["uniform"]["updated_pixels"], 3 * 8 * 8)
        self.assertEqual(results["adaptive"]["processed_pixels"], 3 * 10 * 10)
        self.assertEqual(results["uniform"]["processed_pixels"], 3 * 10 * 10)

    def test_tile_count_covers_image(self) -> None:
        """Non-overlapping tiles cover each image pixel once, including edge tiles."""
        tiles = make_tiles((33, 35, 3), tile_size=16)
        self.assertEqual(len(tiles), 9)
        covered = sum((y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in tiles)
        self.assertEqual(covered, 33 * 35)


if __name__ == "__main__":
    unittest.main()
