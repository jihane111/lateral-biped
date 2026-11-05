"""Reinforcement-learning demo for the Jansen biped environment.

    # train (pick one)
    python -m jansen_biped.rl_demo train --algo ars --out runs/ars --iterations 30     # numpy only
    python -m jansen_biped.rl_demo train --algo ppo --out runs/ppo --timesteps 200000   # needs stable-baselines3

    # measure (headless) and compare with baselines
    python -m jansen_biped.rl_demo eval --policy runs/ars/ars_policy.npz --episodes 5
    python -m jansen_biped.rl_demo eval --policy constant:0.5            # the open-loop gait the mechanism is built for
    python -m jansen_biped.rl_demo eval --policy random

    # deploy: run the saved policy live in the viewer
    python -m jansen_biped.rl_demo deploy --policy runs/ars/ars_policy.npz --manual-end

THIS IS A DEMONSTRATION OF THE PIPELINE, NOT A WORKING WALKER. The robot has one actuator, thin
3 mm feet and two legs, so these policies are expected to fail (fall over). What the demo shows is
how to train, save, load, evaluate and deploy a controller against this environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from .mujoco_env import OBS_NAMES, add_env_args, evaluate, make_env, run_loop


# ----------------------------------------------------------------------------
# policies: anything with .act(obs) -> array of shape (1,) in [-1, 1]
# ----------------------------------------------------------------------------

class ConstantPolicy:
    """Always command the same crank speed (fraction of max_crank_rpm). The open-loop gait."""

    def __init__(self, value: float = 0.5):
        self.value = float(np.clip(value, -1.0, 1.0))

    def act(self, obs):
        return np.array([self.value], dtype=np.float32)


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return self.rng.uniform(-1.0, 1.0, 1).astype(np.float32)


class LinearPolicy:
    """a = clip(W @ ((obs - mean) / std), -1, 1). Small, fast, and deployable as plain numpy."""

    def __init__(self, W, mean, std):
        self.W = np.asarray(W, dtype=np.float64)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    @classmethod
    def zeros(cls, obs_dim: int, act_dim: int = 1):
        return cls(np.zeros((act_dim, obs_dim)), np.zeros(obs_dim), np.ones(obs_dim))

    def act(self, obs):
        z = (np.asarray(obs, dtype=np.float64) - self.mean) / (self.std + 1e-8)
        return np.clip(self.W @ z, -1.0, 1.0).astype(np.float32)

    def save(self, path):
        np.savez(path, kind="linear", W=self.W, mean=self.mean, std=self.std, obs_names=np.array(OBS_NAMES))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["W"], z["mean"], z["std"])


class SB3Policy:
    """Wraps a saved Stable-Baselines3 model (deterministic actions)."""

    def __init__(self, path):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise SystemExit("Loading a .zip policy needs stable-baselines3: pip install stable-baselines3") from exc
        self.model = PPO.load(str(path), device="cpu")

    def act(self, obs):
        action, _ = self.model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)


def load_policy(spec: str, seed: int = 0):
    """'constant:0.5' | 'random' | path to .npz (ARS) | path to .zip (PPO)."""
    if spec == "random":
        return RandomPolicy(seed)
    if spec.startswith("constant"):
        return ConstantPolicy(float(spec.split(":")[1]) if ":" in spec else 0.5)
    p = Path(spec)
    if not p.exists():
        raise SystemExit(f"policy file not found: {spec}")
    if p.suffix == ".npz":
        return LinearPolicy.load(p)
    if p.suffix == ".zip":
        return SB3Policy(p)
    raise SystemExit(f"don't know how to load {spec!r} (expected .npz, .zip, 'random' or 'constant:<x>')")


# ----------------------------------------------------------------------------
# ARS: Augmented Random Search (Mania et al. 2018), a tiny gradient-free RL method
# ----------------------------------------------------------------------------

class RunningStat:
    """Welford running mean and standard deviation of observations."""

    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)

    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def std(self):
        return np.sqrt(self.m2 / self.n) if self.n > 1 else np.ones_like(self.mean)


def _rollout(env, W, mean, std, seed, stat=None):
    obs, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        if stat is not None:
            stat.update(obs)
        a = np.clip(W @ ((obs - mean) / (std + 1e-8)), -1.0, 1.0)
        obs, r, term, trunc, info = env.step(a.astype(np.float32))
        total += r
        steps += 1
        if term or trunc:
            return total, steps, info


def train_ars(env, *, iterations=30, directions=8, top=4, noise=0.05, step_size=0.02, seed=0,
              out_dir=Path("runs/ars"), log=print):
    """Train a linear policy with ARS and write ars_policy.npz and training_log.csv to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    obs_dim = int(env.observation_space.shape[0])
    W = np.zeros((1, obs_dim))
    stat = RunningStat(obs_dim)
    best = (-np.inf, W.copy(), stat.mean.copy(), stat.std.copy())
    rows = []
    t0 = time.time()
    for it in range(1, iterations + 1):
        mean, std = stat.mean.copy(), stat.std.copy()
        deltas = rng.normal(size=(directions, *W.shape))
        seeds = rng.integers(0, 2 ** 31 - 1, size=directions)
        r_pos, r_neg = np.zeros(directions), np.zeros(directions)
        for k in range(directions):                     # same episode seed for + and - to cut variance
            r_pos[k], _, _ = _rollout(env, W + noise * deltas[k], mean, std, int(seeds[k]), stat)
            r_neg[k], _, _ = _rollout(env, W - noise * deltas[k], mean, std, int(seeds[k]), stat)
        order = np.argsort(-np.maximum(r_pos, r_neg))[:top]
        sigma_r = np.concatenate([r_pos[order], r_neg[order]]).std() + 1e-8
        W = W + step_size / (top * sigma_r) * sum((r_pos[k] - r_neg[k]) * deltas[k] for k in order)
        ret, steps, info = _rollout(env, W, stat.mean.copy(), stat.std.copy(), int(rng.integers(0, 2 ** 31 - 1)))
        rows.append({"iteration": it, "eval_return": ret, "eval_steps": steps, "eval_forward_distance_m": info["forward_distance"],
                     "mean_perturbed_return": float(np.concatenate([r_pos, r_neg]).mean()), "seconds": time.time() - t0})
        if ret > best[0]:
            best = (ret, W.copy(), stat.mean.copy(), stat.std.copy())
        log(f"[ars] it {it:3d}/{iterations}  eval return {ret:8.2f}  steps {steps:4d}  "
            f"distance {1000 * info['forward_distance']:7.1f} mm  perturbed mean {rows[-1]['mean_perturbed_return']:8.2f}")
    LinearPolicy(best[1], best[2], best[3]).save(out_dir / "ars_policy.npz")
    LinearPolicy(W, stat.mean, stat.std).save(out_dir / "ars_policy_last.npz")
    with (out_dir / "training_log.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log(f"[ars] saved best policy (eval return {best[0]:.2f}) to {out_dir / 'ars_policy.npz'}")
    return out_dir / "ars_policy.npz"


# ----------------------------------------------------------------------------
# PPO via Stable-Baselines3
# ----------------------------------------------------------------------------

def train_ppo(args, out_dir: Path, log=print):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        raise SystemExit("PPO needs Stable-Baselines3 and PyTorch: pip install stable-baselines3 gymnasium\n"
                         "(or use --algo ars, which needs only numpy)") from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    vec = make_vec_env(lambda: make_env(args), n_envs=args.n_envs, seed=args.seed)
    model = PPO("MlpPolicy", vec, seed=args.seed, verbose=1, n_steps=256, batch_size=256, learning_rate=3e-4,
                gamma=0.99, tensorboard_log=str(out_dir / "tb") if args.tensorboard else None)
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    path = out_dir / "ppo_policy"
    model.save(str(path))
    log(f"[ppo] saved {path}.zip")
    return Path(str(path) + ".zip")


# ----------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------

def _summarise(rows, log=print):
    keys = ("return", "seconds", "forward_distance_m", "peak_crank_torque_Nm")
    for r in rows:
        log(f"  episode {r['episode']}: return {r['return']:8.2f}  lasted {r['seconds']:5.2f} s  "
            f"moved {1000 * r['forward_distance_m']:7.1f} mm  {'FELL' if r['fell'] else 'stayed up'}  "
            f"peak crank torque {r['peak_crank_torque_Nm']:.2f} N m")
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    log(f"  mean: return {mean['return']:.2f}, {mean['seconds']:.2f} s, {1000 * mean['forward_distance_m']:.1f} mm; "
        f"fell in {sum(r['fell'] for r in rows)}/{len(rows)} episodes")
    return mean


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RL demo for the Jansen biped (expected to fail; shows the pipeline).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a policy")
    add_env_args(t)
    t.add_argument("--algo", choices=["ars", "ppo"], default="ars")
    t.add_argument("--out", default="runs/demo")
    t.add_argument("--iterations", type=int, default=30, help="ARS iterations")
    t.add_argument("--directions", type=int, default=8, help="ARS random directions per iteration")
    t.add_argument("--top", type=int, default=4, help="ARS directions used for the update")
    t.add_argument("--noise", type=float, default=0.05)
    t.add_argument("--step-size", type=float, default=0.02)
    t.add_argument("--timesteps", type=int, default=100_000, help="PPO total environment steps")
    t.add_argument("--n-envs", type=int, default=4, help="PPO parallel environments")
    t.add_argument("--tensorboard", action="store_true")

    e = sub.add_parser("eval", help="headless evaluation over several episodes")
    add_env_args(e)
    e.add_argument("--policy", required=True, help="path to .npz/.zip, 'random', or 'constant:<speed>'")
    e.add_argument("--episodes", type=int, default=5)
    e.add_argument("--json", default=None, help="also write the per-episode metrics here")

    d = sub.add_parser("deploy", help="run a saved policy live (viewer)")
    add_env_args(d)
    d.add_argument("--policy", required=True)
    d.add_argument("--no-viewer", action="store_true", help="run headless (still honours --manual-end: Ctrl-C to stop)")
    d.add_argument("--manual-end", action="store_true",
                   help="never end by itself: keep going after falls, no time limit; close the viewer or press Ctrl-C")
    d.add_argument("--reset-on-fall", action="store_true", help="after a fall, reset the robot and try again")
    d.add_argument("--steps", type=int, default=None, help="stop after this many control steps")
    d.add_argument("--fast", action="store_true", help="do not slow the viewer down to real time")

    a = ap.parse_args(argv)
    if a.cmd == "train":
        env = make_env(a)
        out = Path(a.out)
        if a.algo == "ars":
            train_ars(env, iterations=a.iterations, directions=a.directions, top=a.top, noise=a.noise,
                      step_size=a.step_size, seed=a.seed, out_dir=out)
        else:
            env.close()
            train_ppo(a, out)
        return 0
    policy = load_policy(a.policy, a.seed)
    if a.cmd == "eval":
        env = make_env(a)
        rows = evaluate(env, policy.act, episodes=a.episodes, seed=a.seed)
        print(f"Policy {a.policy}:")
        _summarise(rows)
        if a.json:
            Path(a.json).write_text(json.dumps(rows, indent=2))
        return 0
    # deploy
    viewer = not a.no_viewer
    env = make_env(a, render=viewer, manual_end=a.manual_end)
    if a.manual_end:
        print("Manual-end mode: the run never stops by itself. To end it, " +
              ("close the viewer window." if viewer else "press Ctrl-C."))
    res = run_loop(env, policy.act, max_steps=a.steps, manual_end=a.manual_end, reset_on_fall=a.reset_on_fall,
                   realtime=not a.fast, seed=a.seed)
    env.close()
    print(f"Ended by: {res['ended_by']}. {res['steps']} control steps ({res['control_seconds']:.1f} s), "
          f"falls: {res['falls']}, moved {1000 * res['forward_distance_m']:.0f} mm in the last episode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
