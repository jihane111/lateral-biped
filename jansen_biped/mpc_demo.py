"""Model-based control demo: sampling-based MPC (MPPI-style) with the MuJoCo model as the plant model.

    python -m jansen_biped.mpc_demo --steps 300
    python -m jansen_biped.mpc_demo --viewer --manual-end            # watch it; close the window to stop
    python -m jansen_biped.mpc_demo --samples 48 --horizon 20        # plan harder (slower)

How it works. At every control step the controller
  1. copies the real robot's full simulator state into a second, identical "model" environment,
  2. samples many candidate crank-speed sequences around its current best guess,
  3. rolls each one out in the model environment and adds up the environment's own reward,
  4. blends the candidates with weights that favour high reward (MPPI update),
  5. sends only the first action to the real robot and shifts the plan forward.
There is nothing to train: the "model" is the simulator, so this is the best a one-step-at-a-time
planner could do with a perfect model, and it shows how to use ``save_state``/``load_state`` for planning.

THIS IS A DEMONSTRATION, NOT A WORKING WALKER. One actuator, thin feet and a short horizon mean it is
expected to fall over. Planning is slow: each control step costs samples x horizon x frame_skip physics steps.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .mujoco_env import add_env_args, make_env, run_loop


class SamplingMPC:
    """MPPI-style sampling MPC over a 1-D action sequence (crank speed command in [-1, 1])."""

    def __init__(self, model_env, horizon: int = 12, samples: int = 24, sigma: float = 0.4,
                 temperature: float = 0.5, init_action: float = 0.5, seed: int = 0):
        if horizon < 1 or samples < 2:
            raise ValueError("horizon must be >= 1 and samples >= 2")
        self.env = model_env
        self.horizon, self.samples, self.sigma, self.temperature = horizon, samples, sigma, temperature
        self.rng = np.random.default_rng(seed)
        self.plan_ = np.full(horizon, float(np.clip(init_action, -1.0, 1.0)))
        self.last_scores = np.zeros(samples)
        self.last_plan_seconds = 0.0

    def plan(self, state: dict) -> float:
        """Return the first action of the improved plan for the robot in ``state`` (an ``env.save_state()``)."""
        t0 = time.perf_counter()
        K, H = self.samples, self.horizon
        eps = self.rng.normal(0.0, self.sigma, (K, H))
        eps[0] = 0.0                                              # candidate 0 is the current plan itself
        cand = np.clip(self.plan_[None, :] + eps, -1.0, 1.0)
        scores = np.zeros(K)
        for k in range(K):
            self.env.load_state(state)
            total = 0.0
            for h in range(H):
                _, r, term, trunc, info = self.env.step(cand[k, h])
                total += r
                if term or trunc:
                    break
            scores[k] = total
        z = (scores - scores.mean()) / (scores.std() + 1e-8)
        w = np.exp((z - z.max()) / self.temperature)
        w /= w.sum()
        self.plan_ = np.clip(self.plan_ + (w[:, None] * (cand - self.plan_[None, :])).sum(axis=0), -1.0, 1.0)
        action = float(self.plan_[0])
        self.plan_ = np.concatenate([self.plan_[1:], self.plan_[-1:]])    # receding horizon: shift and repeat the last action
        self.last_scores = scores
        self.last_plan_seconds = time.perf_counter() - t0
        return action


def build_controller(real_env, args) -> SamplingMPC:
    """Second environment used only for planning: same model, no random resets, falls end rollouts."""
    model_env = make_env(args, randomize_crank_phase=False, max_episode_steps=None, terminate_on_fall=True)
    model_env.reset(seed=args.seed)
    return SamplingMPC(model_env, horizon=args.horizon, samples=args.samples, sigma=args.sigma,
                       temperature=args.temperature, init_action=args.init_action, seed=args.seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sampling-MPC demo for the Jansen biped (expected to fail; shows the pipeline).")
    add_env_args(ap)
    ap.add_argument("--horizon", type=int, default=12, help="planning horizon in control steps")
    ap.add_argument("--samples", type=int, default=24, help="candidate sequences per control step")
    ap.add_argument("--sigma", type=float, default=0.4, help="exploration noise (action units)")
    ap.add_argument("--temperature", type=float, default=0.5, help="lower = greedier weighting of the best candidates")
    ap.add_argument("--init-action", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=300, help="control steps to run (ignored with --manual-end)")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--manual-end", action="store_true",
                    help="never end by itself: keep going after falls, no step limit; close the viewer or press Ctrl-C")
    ap.add_argument("--reset-on-fall", action="store_true")
    ap.add_argument("--fast", action="store_true", help="do not slow the viewer to real time")
    ap.add_argument("--json", default=None, help="write the run summary here")
    a = ap.parse_args(argv)

    env = make_env(a, render=a.viewer, manual_end=a.manual_end)
    mpc = build_controller(env, a)
    plan_times = []

    def act(obs):
        action = mpc.plan(env.save_state())
        plan_times.append(mpc.last_plan_seconds)
        return np.array([action], dtype=np.float32)

    if a.manual_end:
        print("Manual-end mode: the run never stops by itself. To end it, " +
              ("close the viewer window." if a.viewer else "press Ctrl-C."))
    res = run_loop(env, act, max_steps=None if a.manual_end else a.steps, manual_end=a.manual_end,
                   reset_on_fall=a.reset_on_fall, realtime=not a.fast, seed=a.seed)
    env.close()
    summary = {"ended_by": res["ended_by"], "control_steps": res["steps"], "control_seconds": res["control_seconds"],
               "falls": res["falls"], "forward_distance_m": res["forward_distance_m"],
               "mean_planning_seconds_per_step": float(np.mean(plan_times)) if plan_times else 0.0,
               "horizon": a.horizon, "samples": a.samples}
    print(json.dumps(summary, indent=2))
    if a.json:
        open(a.json, "w").write(json.dumps(summary, indent=2))
    print(f"\nEnded by: {res['ended_by']}. Planned {res['steps']} steps; mean planning time "
          f"{summary['mean_planning_seconds_per_step']:.2f} s per control step (real time is {env.dt:.2f} s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
