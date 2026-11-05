"""Collision-free link layering (pure NumPy)."""
import unittest

from jansen_biped import kinematics as K
from jansen_biped import layer_assignment as LA

UPSTREAM = dict(zip([b[2] for b in K.BARS], (0, 3, 0, 3, 0, 0, 3, 3, 0, 3, 0, -3)))
WIDTHS = {b[2]: 6.0 for b in K.BARS}
WIDTHS["crank"] = 14.1


class LayerTests(unittest.TestCase):
    def test_upstream_two_layer_pattern_has_same_layer_pivots(self):
        pattern = {k: v for k, v in UPSTREAM.items() if k != "fixed frame"}
        bad = LA.same_layer_collisions(pattern)
        self.assertEqual({node for node, _, _ in bad}, {"X", "W", "Y", "V", "U", "T"})

    def test_assignment_has_no_same_layer_pivot_at_several_sizes(self):
        for size in (1.0, 2.5, 4.0):
            la = LA.assign_layers(K.scaled_holy_numbers(size), WIDTHS)
            self.assertEqual(LA.same_layer_collisions(la.layers), [], f"size {size}")
            self.assertEqual(la.n_layers, 4)

    def test_crank_is_innermost(self):
        self.assertEqual(LA.assign_layers(K.scaled_holy_numbers(2.5), WIDTHS).layers["crank"], 0)

    def test_links_that_sweep_the_crank_axis_are_kept_off_the_shaft_hardware_layers(self):
        la = LA.assign_layers(K.scaled_holy_numbers(2.5), WIDTHS)
        self.assertEqual(set(la.near_axis_links), {"j", "k"})
        for name in la.near_axis_links:
            self.assertGreaterEqual(la.layers[name], la.axis_layers)

    def test_shaft_hardware_gap_is_positive_for_links_in_its_layers(self):
        hn = K.scaled_holy_numbers(2.5)
        la = LA.assign_layers(hn, WIDTHS)
        gap = LA.clearance_around_axis(hn, WIDTHS, 7.0, layers=la.layers)
        self.assertGreater(gap, 0.5)

    def test_j_and_k_really_do_cross_the_axis(self):
        gaps = LA.links_near_axis(K.scaled_holy_numbers(2.5), WIDTHS, radius=7.0, clearance=0.0)
        self.assertLess(gaps["j"], 0.0)
        self.assertLess(gaps["k"], 0.0)
        self.assertGreater(gaps["h"], 50.0)

    def test_point_segment_distance_handles_the_origin_case(self):
        import numpy as np
        d = LA._point_segment_distance(np.zeros(2), np.array([3.0, 4.0]), np.array([6.0, 8.0]))
        self.assertAlmostEqual(d, 5.0)


if __name__ == "__main__":
    unittest.main()
