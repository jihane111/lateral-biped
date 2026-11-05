"""Environment, run loop, manual-end behaviour and the RL / MPC demos.

Everything here runs against tests/fake_mujoco.py (a kinematic replay, not physics), so it checks the
Python logic of the environment and demos: spaces, termination, bookkeeping, state save/restore,
ending rules, training plumbing. Real MuJoCo dynamics are NOT tested here."""
import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fake_mujoco as fake  # noqa: E402

from jansen_biped import mujoco_model as MM  # noqa: E402
from jansen_biped import mpc_demo, mujoco_env, mujoco_run, rl_demo  # noqa: E402
from jansen_biped.mujoco_env import EnvConfig, JansenBipedEnv, evaluate, run_loop  # noqa: E402

_TMP = tempfile.TemporaryDirectory()
with contextlib.redirect_stdout(io.StringIO()):
    MM.main(["--out", _TMP.name])
MODEL = str(Path(_TMP.name) / "jansen_biped.xml")


def tearDownModule():
    _TMP.cleanup()


def patched():
    return mock.patch.dict(sys.modules, {"mujoco": fake, "mujoco.viewer": fake.viewer})


def make(render_mode=None, **kw):
    cfg = dict(model_path=MODEL)
    cfg.update(kw)
    return JansenBipedEnv(EnvConfig(**cfg), render_mode=render_mode)


def tilt(env, deg=70.0):
    h = math.radians(deg) / 2
    env.data.qpos[3:7] = [math.cos(h), math.sin(h), 0.0, 0.0]


class EnvBasics(unittest.TestCase):
    def setUp(self):
        self.p = patched()
        self.p.start()
        self.addCleanup(self.p.stop)

    def test_spaces_reset_and_step_shapes(self):
        env = make()
        obs, info = env.reset(seed=1)
        self.assertEqual(obs.shape, (16,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertEqual(env.action_space.shape, (1,))
        self.assertEqual(len(mujoco_env.OBS_NAMES), env.observation_space.shape[0])
        out = env.step(np.array([0.2], dtype=np.float32))
        self.assertEqual(len(out), 5)
        for key in ("x", "y", "z", "roll", "pitch", "forward_velocity", "forward_distance", "crank_torque", "fell",
                    "nonfinite", "reward_terms", "time"):
            self.assertIn(key, out[4])

    def test_upright_observation(self):
        env = make(randomize_crank_phase=False, orientation_noise_deg=0.0, velocity_noise=0.0)
        obs, _ = env.reset(seed=0)
        names = mujoco_env.OBS_NAMES
        np.testing.assert_allclose(obs[names.index("gravity_body_z")], -1.0, atol=1e-6)
        self.assertAlmostEqual(float(obs[names.index("height_error")]), 0.0, places=6)

    def test_action_maps_to_crank_speed_and_is_clipped(self):
        env = make()
        env.reset(seed=0)
        omega_max = 60 * 2 * math.pi / 60
        env.step([0.5])
        self.assertAlmostEqual(float(env.data.ctrl[0]), 0.5 * omega_max, places=6)
        env.step([-9.0])
        self.assertAlmostEqual(float(env.data.ctrl[0]), -omega_max, places=6)
        with self.assertRaises(ValueError):
            env.step([0.1, 0.2])

    def test_seeded_reset_is_reproducible(self):
        env = make()
        a, _ = env.reset(seed=7)
        b, _ = env.reset(seed=7)
        c, _ = env.reset(seed=8)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_random_crank_phase_keeps_the_loops_closed(self):
        env = JansenBipedEnv(EnvConfig(model_path=None))       # builds in memory with a 1 degree sweep
        worst = 0.0
        for ph in (0.0, 37.5, 111.1, 200.0, 301.7):
            env.reset(seed=1, options={"crank_phase_deg": ph})
            worst = max(worst, max(mujoco_run.connect_gaps(fake, env.model, env.data).values()))
        self.assertLess(worst, 3e-4)

    def test_without_sweep_file_the_reference_pose_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "jansen_biped.xml").write_text(Path(MODEL).read_text())
            env = JansenBipedEnv(EnvConfig(model_path=str(Path(tmp) / "jansen_biped.xml")))
            env.reset(seed=0)
            self.assertAlmostEqual(float(env.data.qpos[env._qadr]), 0.0)

    def test_unknown_config_option_is_rejected(self):
        with self.assertRaises(TypeError):
            JansenBipedEnv(model_path=MODEL, bogus=1)

    def test_fall_terminates_and_penalises(self):
        env = make()
        env.reset(seed=0)
        tilt(env, 70)
        _, r, term, trunc, info = env.step([0.0])
        self.assertTrue(term and info["fell"])
        self.assertAlmostEqual(info["reward_terms"]["fall"], -5.0)

    def test_terminate_on_fall_false_keeps_going_without_penalty(self):
        env = make(terminate_on_fall=False, max_episode_steps=None)
        env.reset(seed=0)
        tilt(env, 70)
        _, r, term, trunc, info = env.step([0.0])
        self.assertFalse(term)
        self.assertTrue(info["fell"])
        self.assertEqual(info["reward_terms"]["fall"], 0.0)

    def test_time_limit_truncates(self):
        env = make(max_episode_steps=3)
        env.reset(seed=0)
        flags = [env.step([0.1])[3] for _ in range(3)]
        self.assertEqual(flags, [False, False, True])

    def test_nonfinite_state_terminates_cleanly(self):
        env = make()
        env.reset(seed=0)
        env.data.qpos[0] = float("nan")
        obs, r, term, trunc, info = env.step([0.1])
        self.assertTrue(term and info["nonfinite"])
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_reward_terms_sum_and_forward_term(self):
        env = make()
        env.reset(seed=0)
        env.data.qvel[0] = 0.15
        _, r, _, _, info = env.step([0.0])
        self.assertAlmostEqual(sum(info["reward_terms"].values()), r, places=9)
        self.assertAlmostEqual(info["reward_terms"]["forward"], 1.0, places=2)

    def test_save_and_load_state_roundtrip(self):
        env = make()
        env.reset(seed=3)
        for _ in range(3):
            env.step([0.3])
        s = env.save_state()
        a = [env.step([-0.7])[0] for _ in range(4)]
        env.load_state(s)
        b = [env.step([-0.7])[0] for _ in range(4)]
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        env.load_state(s)
        self.assertEqual(env._steps, s["steps"])

    def test_render_modes(self):
        env = make(render_mode="rgb_array")
        env.reset(seed=0)
        self.assertEqual(env.render().shape, (480, 640, 3))
        env.close()
        env = make(render_mode="human")
        env.reset(seed=0)
        env.render()
        self.assertTrue(env.viewer_running())
        env.close()


class RunLoopEndings(unittest.TestCase):
    def setUp(self):
        self.p = patched()
        self.p.start()
        self.addCleanup(self.p.stop)
        fake.viewer.lifetime = 40
        self.addCleanup(setattr, fake.viewer, "lifetime", 25)

    def knock_over_at(self, env, step_no):
        count = {"n": 0}

        def act(obs):
            count["n"] += 1
            if count["n"] == step_no:
                tilt(env)
            return np.array([0.3], dtype=np.float32)
        return act

    def test_default_stops_when_the_robot_falls(self):
        env = make()
        res = run_loop(env, self.knock_over_at(env, 5), realtime=False, log=lambda *_: None)
        self.assertEqual(res["ended_by"], "robot fell")
        self.assertEqual(res["steps"], 5)

    def test_time_limit_ends_the_default_run(self):
        env = make(max_episode_steps=6)
        res = run_loop(env, lambda o: [0.2], realtime=False, log=lambda *_: None)
        self.assertEqual(res["ended_by"], "time limit")

    def test_manual_end_continues_after_a_fall_until_the_viewer_is_closed(self):
        env = make(terminate_on_fall=False, max_episode_steps=None, render_mode="human")
        res = run_loop(env, self.knock_over_at(env, 5), manual_end=True, realtime=False, log=lambda *_: None)
        self.assertEqual(res["ended_by"], "viewer closed by user")
        self.assertEqual(res["falls"], 1)
        self.assertGreater(res["steps"], 30)

    def test_manual_end_ignores_a_step_limit_only_if_none_is_given(self):
        env = make(terminate_on_fall=False, max_episode_steps=None, render_mode="human")
        res = run_loop(env, lambda o: [0.2], manual_end=True, max_steps=12, realtime=False, log=lambda *_: None)
        self.assertEqual(res["ended_by"], "step limit")
        self.assertEqual(res["steps"], 12)

    def test_reset_on_fall_keeps_going(self):
        env = make()
        res = run_loop(env, self.knock_over_at(env, 5), reset_on_fall=True, max_steps=30, realtime=False, log=lambda *_: None)
        self.assertEqual(res["ended_by"], "step limit")
        self.assertEqual(res["steps"], 30)
        self.assertGreaterEqual(res["falls"], 1)

    def test_ctrl_c_ends_cleanly(self):
        env = make()
        n = {"i": 0}

        def act(obs):
            n["i"] += 1
            if n["i"] == 4:
                raise KeyboardInterrupt
            return [0.2]
        res = run_loop(env, act, realtime=False, log=lambda *_: None)
        self.assertEqual(res["ended_by"], "Ctrl-C")
        self.assertEqual(res["steps"], 3)

    def test_evaluate_returns_per_episode_metrics(self):
        env = make(max_episode_steps=4)
        rows = evaluate(env, lambda o: [0.4], episodes=3, seed=5)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["steps"] == 4 and not r["fell"] for r in rows))


class MujocoRunManualEnd(unittest.TestCase):
    def setUp(self):
        self.p = patched()
        self.p.start()
        self.addCleanup(self.p.stop)
        self.addCleanup(setattr, fake, "TILT_AT", None)
        self.addCleanup(setattr, fake.viewer, "lifetime", 25)

    def run_cmd(self, *extra):
        out = Path(_TMP.name) / "mr"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = mujoco_run.main(["run", "--model", MODEL, "--out", str(out), *extra])
        return code, json.loads((out / "run_summary.json").read_text()), buf.getvalue()

    def test_default_stops_at_the_time_limit(self):
        code, s, _ = self.run_cmd("--seconds", "0.3")
        self.assertEqual((code, s["ended_by"], s["manual_end"]), (0, "time limit", False))

    def test_default_stops_when_the_robot_falls(self):
        fake.TILT_AT = 0.2
        _, s, _ = self.run_cmd("--seconds", "5")
        self.assertEqual(s["ended_by"], "robot fell")
        self.assertLess(s["simulated_s"], 1.0)

    def test_manual_end_keeps_running_after_a_fall_until_the_viewer_closes(self):
        fake.TILT_AT = 0.05
        fake.viewer.lifetime = 400                     # 400 steps = 0.4 s of simulated time
        _, s, text = self.run_cmd("--seconds", "0.1", "--manual-end", "--viewer")
        self.assertEqual(s["ended_by"], "viewer closed by user")
        self.assertTrue(s["manual_end"])
        self.assertIsNotNone(s["fell_at_s"])
        self.assertGreater(s["simulated_s"], 0.3)       # --seconds 0.1 was ignored
        self.assertIn("continuing because of --manual-end", text)

    def test_ctrl_c_writes_the_log_and_summary(self):
        calls = {"n": 0}
        real = fake.mj_step

        def interrupting(m, d):
            calls["n"] += 1
            if calls["n"] > 50:
                raise KeyboardInterrupt
            real(m, d)
        with mock.patch.object(fake, "mj_step", interrupting):
            _, s, _ = self.run_cmd("--manual-end")
        self.assertEqual(s["ended_by"], "Ctrl-C")
        self.assertTrue((Path(_TMP.name) / "mr" / "run_log.csv").exists())


class ARSAndPolicies(unittest.TestCase):
    def setUp(self):
        self.p = patched()
        self.p.start()
        self.addCleanup(self.p.stop)

    def test_running_stat_matches_numpy(self):
        rng = np.random.default_rng(0)
        x = rng.normal(2.0, 3.0, (200, 4))
        st = rl_demo.RunningStat(4)
        for row in x:
            st.update(row)
        np.testing.assert_allclose(st.mean, x.mean(axis=0))
        np.testing.assert_allclose(st.std, x.std(axis=0), rtol=1e-9)

    def test_linear_policy_round_trip_and_bounds(self):
        pol = rl_demo.LinearPolicy(np.full((1, 16), 5.0), np.zeros(16), np.ones(16))
        a = pol.act(np.ones(16))
        self.assertEqual(a.shape, (1,))
        self.assertEqual(float(a[0]), 1.0)                       # clipped
        with tempfile.TemporaryDirectory() as tmp:
            pol.save(Path(tmp) / "p.npz")
            back = rl_demo.LinearPolicy.load(Path(tmp) / "p.npz")
        np.testing.assert_array_equal(back.W, pol.W)
        self.assertEqual(float(rl_demo.LinearPolicy.zeros(16).act(np.ones(16))[0]), 0.0)

    def test_ars_trains_saves_and_is_deterministic(self):
        def run(out):
            env = make(max_episode_steps=6)
            rl_demo.train_ars(env, iterations=2, directions=2, top=1, seed=11, out_dir=out, log=lambda *_: None)
            return rl_demo.LinearPolicy.load(Path(out) / "ars_policy.npz")
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            a, b = run(t1), run(t2)
            np.testing.assert_array_equal(a.W, b.W)
            rows = (Path(t1) / "training_log.csv").read_text().strip().splitlines()
            self.assertEqual(len(rows), 3)                      # header + 2 iterations
            self.assertTrue((Path(t1) / "ars_policy_last.npz").exists())

    def test_load_policy_variants_and_errors(self):
        self.assertAlmostEqual(float(rl_demo.load_policy("constant:0.7").act(None)[0]), 0.7, places=6)
        self.assertEqual(rl_demo.load_policy("random", 1).act(None).shape, (1,))
        with self.assertRaises(SystemExit):
            rl_demo.load_policy("nope.npz")
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "x.pkl").write_text("x")
            with self.assertRaises(SystemExit):
                rl_demo.load_policy(str(Path(tmp) / "x.pkl"))
            (Path(tmp) / "m.zip").write_text("x")
            with mock.patch.dict(sys.modules, {"stable_baselines3": None}), self.assertRaises(SystemExit) as ctx:
                rl_demo.load_policy(str(Path(tmp) / "m.zip"))
            self.assertIn("stable-baselines3", str(ctx.exception))

    def test_ppo_without_stable_baselines_gives_a_clear_message(self):
        with mock.patch.dict(sys.modules, {"stable_baselines3": None}), self.assertRaises(SystemExit) as ctx, \
                contextlib.redirect_stdout(io.StringIO()):
            rl_demo.main(["train", "--algo", "ppo", "--model", MODEL, "--out", str(Path(_TMP.name) / "ppo")])
        self.assertIn("--algo ars", str(ctx.exception))

    def test_cli_train_eval_deploy_manual_end(self):
        out = Path(_TMP.name) / "cli_ars"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(rl_demo.main(["train", "--algo", "ars", "--model", MODEL, "--out", str(out), "--iterations", "2",
                                           "--directions", "2", "--top", "1", "--max-episode-steps", "5"]), 0)
            self.assertEqual(rl_demo.main(["eval", "--policy", str(out / "ars_policy.npz"), "--model", MODEL, "--episodes", "2",
                                           "--max-episode-steps", "5", "--json", str(out / "eval.json")]), 0)
            self.assertEqual(rl_demo.main(["eval", "--policy", "constant:0.5", "--model", MODEL, "--episodes", "1",
                                           "--max-episode-steps", "5"]), 0)
            fake.viewer.lifetime = 30
            self.assertEqual(rl_demo.main(["deploy", "--policy", "constant:0.5", "--model", MODEL, "--manual-end", "--fast"]), 0)
        text = buf.getvalue()
        self.assertEqual(len(json.loads((out / "eval.json").read_text())), 2)
        self.assertIn("Manual-end mode", text)
        self.assertIn("viewer closed by user", text)
        fake.viewer.lifetime = 25


class StubEnv:
    """Deterministic toy model: reward is highest when the action is 0.8."""

    def load_state(self, s):
        pass

    def step(self, a):
        return None, -(float(a) - 0.8) ** 2, False, False, {}


class MPCTests(unittest.TestCase):
    def test_mppi_converges_on_a_known_optimum(self):
        mpc = mpc_demo.SamplingMPC(StubEnv(), horizon=5, samples=32, sigma=0.3, temperature=0.5, init_action=-0.5, seed=0)
        a = None
        for _ in range(40):
            a = mpc.plan({})
        self.assertAlmostEqual(a, 0.8, delta=0.15)

    def test_actions_stay_in_bounds_and_planning_is_deterministic(self):
        a = mpc_demo.SamplingMPC(StubEnv(), horizon=4, samples=8, sigma=2.0, seed=3)
        b = mpc_demo.SamplingMPC(StubEnv(), horizon=4, samples=8, sigma=2.0, seed=3)
        xs = [a.plan({}) for _ in range(10)]
        ys = [b.plan({}) for _ in range(10)]
        self.assertEqual(xs, ys)
        self.assertTrue(all(-1.0 <= x <= 1.0 for x in xs))

    def test_invalid_sizes_rejected(self):
        with self.assertRaises(ValueError):
            mpc_demo.SamplingMPC(StubEnv(), horizon=0)

    def test_planning_does_not_disturb_the_real_robot(self):
        with patched():
            real = make()
            real.reset(seed=0)
            args = mpc_demo.argparse.Namespace(model=MODEL, linkage_size=2.5, ratio="7", foot_width=3.0, payload_mass=0.0,
                                               floor_friction=0.8, frame_skip=10, max_episode_steps=1000, max_crank_rpm=60.0,
                                               seed=0, horizon=3, samples=3, sigma=0.4, temperature=0.5, init_action=0.5)
            mpc = mpc_demo.build_controller(real, args)
            before = real.save_state()
            mpc.plan(before)
            after = real.save_state()
        np.testing.assert_array_equal(before["qpos"], after["qpos"])
        self.assertEqual(before["time"], after["time"])

    def test_cli_runs_and_manual_end_needs_the_viewer_closed(self):
        with patched():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(mpc_demo.main(["--model", MODEL, "--steps", "3", "--samples", "3", "--horizon", "2", "--fast"]), 0)
                fake.viewer.lifetime = 8
                self.assertEqual(mpc_demo.main(["--model", MODEL, "--samples", "3", "--horizon", "2", "--fast", "--viewer",
                                                "--manual-end"]), 0)
            fake.viewer.lifetime = 25
        text = buf.getvalue()
        self.assertIn("control_steps", text)
        self.assertIn("viewer closed by user", text)


class MissingMujoco(unittest.TestCase):
    def test_demos_exit_with_install_advice_instead_of_a_traceback(self):
        with mock.patch.dict(sys.modules, {"mujoco": None}):
            for main, argv in ((rl_demo.main, ["eval", "--policy", "constant:0.5", "--model", MODEL]),
                               (mpc_demo.main, ["--model", MODEL, "--steps", "1"])):
                with self.assertRaises(SystemExit) as ctx:
                    main(argv)
                self.assertIn("pip install", str(ctx.exception))


class GymnasiumFallback(unittest.TestCase):
    def test_box_fallback_when_gymnasium_is_missing(self):
        if mujoco_env.HAVE_GYMNASIUM:
            self.skipTest("gymnasium is installed; the stand-in is not used")
        box = mujoco_env.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.assertTrue(box.contains(box.sample()))
        self.assertFalse(box.contains(np.array([3.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
