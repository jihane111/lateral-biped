"""MuJoCo model generator, offline verifier and run script (no MuJoCo required)."""
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jansen_biped import mujoco_model as MM
from jansen_biped import mujoco_verify as MV
from jansen_biped.kinematics import scaled_holy_numbers

_CACHE = {}


def built(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _CACHE:
        _CACHE[key] = MM.build_mujoco_model(MM.MujocoParams(**kw))
    return _CACHE[key]


def verify(b, size=2.5):
    return MV.verify_model(b["xml"], b["sweep"], scaled_holy_numbers(size), b["report"])


class GeneratorTests(unittest.TestCase):
    def test_default_model_passes_every_offline_check(self):
        b = built()
        failed = [c.name for c in verify(b) if not c.passed]
        self.assertEqual(failed, [])

    def test_other_ratios_sizes_and_options_also_pass(self):
        for kw, size in (({"ratio": "15"}, 2.5), ({"linkage_size": 3.0}, 3.0), ({"linkage_size": 1.6}, 1.6),
                         ({"actuator": "torque", "payload_mass_kg": 0.2, "foot_halfwidth_mm": 0.0, "start_angle_deg": 40.0}, 2.5)):
            b = built(**kw)
            self.assertEqual([c.name for c in verify(b, size) if not c.passed], [], kw)

    def test_counts_match_the_mechanism(self):
        c = built()["report"]["counts"]
        self.assertEqual((c["expected_nq"], c["expected_nv"], c["connect_constraints"], c["expected_equality_rows"]), (28, 27, 10, 30))
        xml = built()["xml"]
        self.assertEqual(len(re.findall(r"<connect ", xml)), 10)
        self.assertEqual(len(re.findall(r"<joint name", xml)), 21)

    def test_single_driven_joint_and_180_degree_phase(self):
        xml = built()["xml"]
        self.assertEqual(len(re.findall(r"<(velocity|motor) ", xml)), 1)
        m = MV.Model(xml)
        # the two crank pins are exactly opposite each other about the shaft
        pl, pr = m.bodies["j_L"]["pos"], m.bodies["j_R"]["pos"]
        self.assertAlmostEqual(pl[0], -pr[0], places=9)
        self.assertAlmostEqual(pl[2], -pr[2], places=9)
        self.assertGreater(pl[1], 0)       # left leg at +Y
        self.assertLess(pr[1], 0)

    def test_legs_use_the_collision_free_layers(self):
        m = MV.Model(built()["xml"])
        ys = {n: b["pos"][1] for n, b in m.bodies.items() if n.endswith("_L") and n[0] in "bcdefghijk" and len(n) == 3}
        by_node = {}
        # bars sharing a pivot must sit at different lateral offsets (different layers)
        for a, b in (("b", "d"), ("b", "c"), ("d", "c"), ("j", "k")):
            self.assertNotAlmostEqual(ys[f"{a}_L"], ys[f"{b}_L"], places=6, msg=f"{a},{b} share a layer")

    def test_masses_are_plausible_and_consistent(self):
        r = built()["report"]
        self.assertAlmostEqual(r["mass_kg"]["total"], sum(p["mass_kg"] for p in r["mass_parts"]), places=9)
        self.assertGreater(r["mass_kg"]["total"], 0.3)
        self.assertLess(r["mass_kg"]["total"], 3.0)
        self.assertAlmostEqual(r["mass_kg"]["by_group"]["motor"], 0.25)
        self.assertTrue(r["com_x_inside_stance_x_range"])

    def test_payload_adds_mass(self):
        self.assertAlmostEqual(built(payload_mass_kg=0.3)["report"]["mass_kg"]["total"] - built()["report"]["mass_kg"]["total"], 0.3, places=9)

    def test_reflected_inertia_scales_with_the_square_of_the_ratio(self):
        r7, r15 = built()["report"]["gearbox"], built(ratio="15")["report"]["gearbox"]
        self.assertAlmostEqual(r7["ratio"], 7.0)
        self.assertGreater(r15["reflected_inertia_on_crank_kgm2"] / r7["reflected_inertia_on_crank_kgm2"], (15 / 7) ** 2 * 0.8)

    def test_feet_start_nearly_level_and_just_above_the_floor(self):
        r = built()["report"]
        self.assertLess(r["foot_height_difference_at_start_mm"], 2.0)

    def test_generation_is_deterministic(self):
        a = MM.build_mujoco_model(MM.MujocoParams())
        self.assertEqual(a["xml"], built()["xml"])

    def test_sweep_starts_at_zero_and_covers_a_full_turn(self):
        sw = built()["sweep"]
        self.assertEqual(sw["crank_offset_deg"][0], 0.0)
        self.assertEqual(sw["crank_offset_deg"][-1], 360.0)
        self.assertTrue(all(abs(x) < 1e-12 for x in sw["hinge_angles_rad"][0]))

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            MM.build_mujoco_model(MM.MujocoParams(ratio="9"))
        with self.assertRaises(ValueError):
            MM.build_mujoco_model(MM.MujocoParams(actuator="position"))
        with self.assertRaises(ValueError):
            MM.build_mujoco_model(MM.MujocoParams(timestep=0.0))

    def test_cli_writes_the_three_files_and_refuses_bad_input(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            MM.main(["--out", tmp])
            for f in ("jansen_biped.xml", "jansen_biped_mujoco_report.json", "crank_sweep.json"):
                self.assertTrue((Path(tmp) / f).exists(), f)
            with self.assertRaises(SystemExit) as ctx:
                MM.main(["--out", tmp, "--linkage-size", "-2"])
            self.assertEqual(ctx.exception.code, 2)
            with self.assertRaises(SystemExit) as ctx:
                MM.main(["--out", tmp, "--linkage-size", "1.0"])           # too small for the motor
            self.assertIn("larger --linkage-size", str(ctx.exception))


class VerifierCanFail(unittest.TestCase):
    """A verifier that cannot fail proves nothing."""

    def run_checks(self, xml):
        b = built()
        return {c.name: c for c in MV.verify_model(xml, b["sweep"], scaled_holy_numbers(2.5), b["report"])}

    def test_swapped_loop_partner_breaks_closure(self):
        # The Jansen linkage contains two rigid triangles, (b, d, e) at Y and (g, h, i) at U, so swapping
        # members of the same triangle would NOT change the mechanism. j and k both pivot on the crank pin but
        # have no link between their far ends, so they do turn relative to each other: joining c to j instead
        # of k at U must break loop U.
        xml = built()["xml"].replace('name="loop_L_U" body1="c_L" body2="k_L"', 'name="loop_L_U" body1="c_L" body2="j_L"')
        self.assertNotEqual(xml, built()["xml"])
        self.assertFalse(self.run_checks(xml)["loop_closure_over_full_crank_cycle"].passed)

    def test_the_two_rigid_triangles_are_really_rigid(self):
        import numpy as np
        sw = built()["sweep"]
        names = sw["joint_names"]
        rows = np.array(sw["hinge_angles_rad"])
        col = lambda n: rows[:, names.index(n)]
        self.assertLess(np.abs(col("b_L_pin") - col("d_L_pin")).max(), 1e-9)      # b, d turn together about Y
        self.assertLess(np.abs(col("e_L_pin")).max(), 1e-9)                       # e is fixed relative to b
        self.assertLess(np.abs(col("h_L_pin")).max(), 1e-9)                       # h is fixed relative to g
        self.assertGreater(np.abs(col("f_L_pin")).max(), 0.1)                     # f does turn relative to d
        self.assertGreater(np.abs(col("i_L_pin")).max(), 0.1)                     # i does turn relative to k

    def test_moved_anchor_is_caught(self):
        xml = re.sub(r'(name="loop_R_S"[^>]*anchor=")(\S+)', lambda m: m.group(1) + "9" + m.group(2)[1:], built()["xml"], count=1)
        res = self.run_checks(xml)
        self.assertFalse(res["loop_closure_over_full_crank_cycle"].passed and res["bar_lengths_match_kinematics"].passed)

    def test_shifted_body_changes_a_bar_length(self):
        def shift(m):
            x, rest = m.group(1), m.group(2)
            return f'<body name="e_L" pos="{float(x) + 0.01:.9g} {rest}'
        xml = re.sub(r'<body name="e_L" pos="(\S+) (\S+ \S+)"', lambda m: shift(m) + '"', built()["xml"], count=1)
        self.assertNotEqual(xml, built()["xml"])
        res = self.run_checks(xml)
        self.assertFalse(res["bar_lengths_match_kinematics"].passed)

    def test_duplicate_name_and_unknown_attribute_are_caught(self):
        xml = built()["xml"].replace('<body name="d_L"', '<body name="b_L"', 1)
        self.assertFalse(self.run_checks(xml)["names_unique_within_type"].passed)
        xml = built()["xml"].replace('<compiler angle="radian"/>', '<compiler angle="radian" bogus="1"/>')
        self.assertFalse(self.run_checks(xml)["only_known_elements_and_attributes"].passed)

    def test_rotated_body_frame_is_caught(self):
        xml = built()["xml"].replace('<body name="crankshaft" pos="0 0 0">', '<body name="crankshaft" pos="0 0 0" euler="0 0.1 0">')
        self.assertFalse(self.run_checks(xml)["body_frames_are_world_aligned"].passed)


class RunScriptWithFakeMujoco(unittest.TestCase):
    """Executes jansen_biped.mujoco_run end to end against tests/fake_mujoco.py.

    This exercises the script's own logic only; real MuJoCo behaviour is not tested here."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with contextlib.redirect_stdout(io.StringIO()):
            MM.main(["--out", cls.tmp.name])
        cls.model = str(Path(cls.tmp.name) / "jansen_biped.xml")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import fake_mujoco
        cls.fake = fake_mujoco

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_check_command_passes_on_the_kinematic_replay(self):
        from jansen_biped import mujoco_run
        buf = io.StringIO()
        with mock.patch.dict(sys.modules, {"mujoco": self.fake}), contextlib.redirect_stdout(buf):
            code = mujoco_run.main(["check", "--model", self.model])
        self.assertEqual(code, 0, buf.getvalue())
        self.assertIn("nq=28", buf.getvalue())
        self.assertIn("loop closure PASS", buf.getvalue())

    def test_run_command_writes_log_and_summary(self):
        from jansen_biped import mujoco_run
        out = Path(self.tmp.name) / "run_out"
        buf = io.StringIO()
        with mock.patch.dict(sys.modules, {"mujoco": self.fake}), contextlib.redirect_stdout(buf):
            code = mujoco_run.main(["run", "--model", self.model, "--seconds", "0.5", "--out", str(out)])
        self.assertEqual(code, 0, buf.getvalue())
        summary = json.loads((out / "run_summary.json").read_text())
        self.assertAlmostEqual(summary["simulated_s"], 0.5, places=3)
        self.assertLess(summary["max_loop_closure_gap_m"], 1e-3)
        self.assertTrue((out / "run_log.csv").exists())
        self.assertIsNotNone(summary["peak_motor_side_torque_Nm"])

    def test_check_fails_when_the_report_disagrees_with_the_model(self):
        from jansen_biped import mujoco_run
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            MM.main(["--out", tmp])
            rp = Path(tmp) / "jansen_biped_mujoco_report.json"
            rep = json.loads(rp.read_text())
            rep["counts"]["expected_nq"] = 99
            rp.write_text(json.dumps(rep))
            with mock.patch.dict(sys.modules, {"mujoco": self.fake}):
                self.assertEqual(mujoco_run.main(["check", "--model", str(Path(tmp) / "jansen_biped.xml")]), 1)


try:
    import mujoco as _real_mujoco
except ImportError:
    _real_mujoco = None


@unittest.skipUnless(_real_mujoco is not None and not str(getattr(_real_mujoco, "__version__", "")).startswith("fake"),
                     "real mujoco not installed")
class RealMujoco(unittest.TestCase):
    def test_model_compiles_closes_and_runs(self):
        from jansen_biped import mujoco_run
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            MM.main(["--out", tmp])
            self.assertEqual(mujoco_run.main(["check", "--model", str(Path(tmp) / "jansen_biped.xml")]), 0)
            self.assertEqual(mujoco_run.main(["run", "--model", str(Path(tmp) / "jansen_biped.xml"), "--seconds", "2",
                                              "--out", str(Path(tmp) / "run")]), 0)


if __name__ == "__main__":
    unittest.main()
