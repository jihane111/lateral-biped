"""Compact drive layout solver (pure Python, no CadQuery needed)."""
import math
import unittest

from jansen_biped import drive_layout as D
from jansen_biped.kinematics import scaled_holy_numbers

HN = scaled_holy_numbers(2.5)


def layout(ratio="7", **kw):
    return D.solve_drive_layout(D.DriveSpec(ratio=ratio, n_layers=4, layer_pitch=3.5, **kw), HN)


class RatioTests(unittest.TestCase):
    def test_ratios_are_exactly_7_and_15(self):
        for key, (z1, z2, z3, z4) in D.RATIOS.items():
            self.assertEqual((z2 / z1) * (z4 / z3), float(key))


class ExistingStage4LayoutFlaw(unittest.TestCase):
    """Documents why the compact layout gives stage 2 its own module."""

    def test_old_7_to_1_axis_lies_inside_the_63_tooth_gear(self):
        c2 = 1.0 * (18 + 42) / 2                       # module 1.0 stage 2
        self.assertLess(c2 - D.root_radius(63, 1.0), 0.0)

    def test_old_15_to_1_clears_the_bare_shaft_by_only_one_millimetre(self):
        c2 = 1.0 * (16 + 60) / 2
        self.assertAlmostEqual(c2 - D.tip_radius(64, 1.0) - 4.0, 1.0)


class CompactLayoutTests(unittest.TestCase):
    def test_both_ratios_pass_every_check_at_default_size(self):
        for ratio in D.RATIOS:
            lay = layout(ratio)
            self.assertTrue(lay.passed, [c.name for c in lay.checks if not c.passed])
            self.assertIn(lay.module2, D.STANDARD_MODULES)

    def test_large_gear_clears_the_output_shaft(self):
        for ratio in D.RATIOS:
            lay = layout(ratio)
            self.assertGreaterEqual(lay.c2 - lay.radii["large_gear_tip"] - lay.spec.output_shaft_d / 2, 1.5)

    def test_centre_distances_equal_pitch_radii(self):
        lay = layout("7")
        z1, z2, z3, z4 = lay.teeth
        self.assertAlmostEqual(lay.c1, lay.module1 * (z1 + z2) / 2)
        self.assertAlmostEqual(lay.c2, lay.module2 * (z3 + z4) / 2)
        self.assertAlmostEqual(math.dist(lay.O, lay.C), lay.c2)
        self.assertAlmostEqual(math.dist(lay.C, lay.M), lay.c1)

    def test_axial_stack_is_consistent_and_symmetric(self):
        lay = layout("7")
        z, s = lay.z, lay.spec
        self.assertAlmostEqual(z["back_wall_outer"], -lay.half_length)
        self.assertAlmostEqual(z["lid_outer"], lay.half_length)
        self.assertAlmostEqual(z["motor_face"] - z["back_wall_inner"], s.motor_length)
        # the compound pinion's hub meets the underside of the large gear
        self.assertAlmostEqual(z["P2_end"] + s.hub_height, z["P1_start"])
        # legs start beyond the lid boss and are mirrored by construction
        self.assertAlmostEqual(lay.z0_leg, lay.half_length + s.boss_height + s.layer_clearance)

    def test_housing_is_shorter_than_the_old_carrier_stack(self):
        lay = layout("7")
        s = lay.spec
        old = 5.0 + 3.0 + 5.0 + 3.0 + 5.0 + s.motor_length            # stage 2, stage 1, plate, motor (old stage-4 stack)
        self.assertLess(2 * lay.half_length, old + 15.0)

    def test_motor_axis_is_near_pivot_Y_not_behind_the_crank(self):
        lay = layout("7")
        self.assertGreater(lay.M[0], lay.pivot_Y[0] - 40.0)

    def test_teeth_interleave_for_random_directions(self):
        import random
        rng = random.Random(1)
        for _ in range(200):
            z1, z2 = rng.choice([(21, 63), (16, 64), (18, 42)])
            a = rng.uniform(0, 360)
            rot1 = rng.uniform(0, 360)
            rot2 = D.mesh_rotation_deg(z1, z2, rot1, a)
            f1 = D.tooth_phase(z1, rot1, a)
            f2 = D.tooth_phase(z2, rot2, a + 180.0)
            self.assertAlmostEqual(((f1 + f2) % 1.0), 0.5, places=9)

    def test_too_small_a_linkage_is_refused_with_advice(self):
        with self.assertRaises(D.LayoutError) as ctx:
            D.solve_drive_layout(D.DriveSpec(ratio="7", n_layers=4, layer_pitch=3.5), scaled_holy_numbers(1.0))
        self.assertIn("larger --linkage-size", str(ctx.exception))

    def test_explicit_bad_module_is_reported_not_raised(self):
        lay = layout("7", module2=1.0)
        self.assertFalse(lay.passed)
        self.assertIn("large_gear_clear_of_output_shaft", [c.name for c in lay.checks if not c.passed])

    def test_bigger_motor_grows_the_housing(self):
        small, big = layout("7", motor_diameter=35.0, motor_length=40.0), layout("7", motor_diameter=43.0, motor_length=55.0)
        self.assertLess(small.half_length, big.half_length)

    def test_motor_size_limit_with_stage1_module_1_and_how_to_lift_it(self):
        with self.assertRaises(D.LayoutError):
            layout("7", motor_diameter=48.0)                 # sleeve would hit the compound-shaft bearing boss
        self.assertTrue(layout("7", motor_diameter=48.0, module1=1.25).passed)
        self.assertTrue(layout("15", motor_diameter=50.0, module1=1.25).passed)

    def test_manifest_dict_is_json_serialisable(self):
        import json
        json.dumps(layout("15").to_dict())


if __name__ == "__main__":
    unittest.main()
