"""Linkage-size scaling: proportions preserved, closure at every size, CLI-facing helpers."""
import unittest

import numpy as np

from jansen_biped import kinematics as K


class ScalingTests(unittest.TestCase):
    def test_default_size_is_2_5_times_upstream(self):
        self.assertEqual(K.DEFAULT_LINKAGE_SIZE, 2.5)
        for got, base in zip(K.HOLY_NUMBERS, K.BASE_HOLY_NUMBERS):
            self.assertAlmostEqual(got, base * 2.5, places=9)

    def test_every_dimension_is_scaled_including_offsets_and_crank(self):
        hn = K.scaled_holy_numbers(2.5)
        a, b, c, d, e, f, g, h, i, j, k, l, m = hn
        self.assertAlmostEqual(a, 95.0)
        self.assertAlmostEqual(l, 19.5)
        self.assertAlmostEqual(m, 37.5)
        self.assertAlmostEqual(h, 164.25)

    def test_foot_path_is_the_upstream_path_scaled(self):
        angles = np.arange(0, 360, 5.0)
        base = np.array([K.leg_nodes(float(t), K.BASE_HOLY_NUMBERS)["S"] for t in angles])
        for size in (0.5, 1.0, 2.5, 4.0):
            scaled = np.array([K.leg_nodes(float(t), K.scaled_holy_numbers(size))["S"] for t in angles])
            np.testing.assert_allclose(scaled, base * size, atol=1e-9)

    def test_all_bars_keep_length_over_full_cycle_at_several_sizes(self):
        for size in (1.0, 2.5, 3.7):
            hn = K.scaled_holy_numbers(size)
            lengths = K.bar_lengths(hn)
            for ang in np.arange(0, 360, 0.5):
                pts = K.leg_nodes(float(ang), hn)
                for n1, n2, name in K.BARS:
                    self.assertAlmostEqual(float(np.linalg.norm(pts[n2] - pts[n1])), lengths[name], delta=1e-9)
            ok, msg = K.validate_full_cycle(720, hn)
            self.assertTrue(ok, msg)

    def test_invalid_sizes_are_rejected(self):
        for bad in (0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                K.scaled_holy_numbers(bad)

    def test_pose_and_validation_accept_explicit_numbers(self):
        hn = K.scaled_holy_numbers(1.0)
        p1 = K.pose_plot_coordinates(30.0, 180.0, hn)["S"]
        p25 = K.pose_plot_coordinates(30.0, 180.0, K.scaled_holy_numbers(2.5))["S"]
        np.testing.assert_allclose(p25, p1 * 2.5, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
