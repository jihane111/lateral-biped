# Jansen biped MuJoCo environment: user guide

This guide covers everything needed to use the simulated robot as a control or learning problem:
installing, creating the environment, what the agent sees and controls, how episodes end, how to end a
run yourself, how to train and deploy the two demo controllers, and how to write your own.

> **Status, please read first.**
> * The demos (`rl_demo`, `mpc_demo`) are **demonstrations of the pipeline, not working walkers**. The robot
>   has one actuator, two thin-footed legs and no balance mechanism, so they are expected to fall over.
>   A demo "succeeds" if it trains, saves, loads, runs and ends the way you asked it to.
> * The code was written and unit-tested **without MuJoCo, Gymnasium or Stable-Baselines3 installed**.
>   The tests run against a fake kinematic stand-in (`tests/fake_mujoco.py`). Run
>   `python -m jansen_biped.mujoco_run check` first; it compiles the model with real MuJoCo and tells you
>   whether everything lines up. Report anything it flags.
> * Reward weights, observation scales and PPO settings are reasonable first guesses, not tuned values.

---

## 1. Install

```bash
pip install -r requirements-mujoco.txt     # mujoco, gymnasium, stable-baselines3 (+ PyTorch)
```

* Minimum to generate/check/run the model: `pip install mujoco` (a recent 3.x; the model uses the `implicitfast` integrator).
* The environment works without `gymnasium` (a minimal stand-in is used), and `rl_demo train --algo ars` needs only NumPy.
  You need `gymnasium` for `gymnasium.make("JansenBiped-v0")` and `stable-baselines3` for `--algo ppo`.
* macOS: the live viewer must be started with `mjpython` instead of `python`, for example
  `mjpython -m jansen_biped.rl_demo deploy ...`.
* Headless rendering (`render_mode="rgb_array"`) on Linux needs `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa`.

Check the model on your machine before anything else:

```bash
python -m jansen_biped.mujoco_model --out output_mujoco
python -m jansen_biped.mujoco_run check --model output_mujoco/jansen_biped.xml
```

`check` should end with `ALL CHECKS PASSED`. It compares MuJoCo's model sizes (28 coordinates, 27 velocities,
10 connect constraints) and mass with the generator's report, measures the loop-closure gap over a full crank
turn, and runs one second of simulation looking for NaNs.

## 2. The robot as a control problem

* **One actuator**: the shared crankshaft. Both cranks are on it, 180 degrees apart, so one number drives both legs.
  Everything else (20 pin joints and 10 loop-closure constraints) is passive mechanism.
* **Frame**: x forward, y left, z up, SI units. A positive crank rotation walks the robot toward **+x**. The left leg is at +y.
* **Gait comes from the mechanism.** Turning the crank at a steady speed produces the Jansen foot motion. A controller can
  only change *how fast* (and in which direction) the crank turns over time. It cannot place feet or shift weight.
* A constant crank speed is therefore the natural baseline: `rl_demo eval --policy constant:0.5`.

## 3. Quick start

```python
from jansen_biped.mujoco_env import JansenBipedEnv

env = JansenBipedEnv()                              # builds the default model in memory (a few seconds)
obs, info = env.reset(seed=0)
for _ in range(500):
    obs, reward, terminated, truncated, info = env.step([0.5])      # half speed, forward
    if terminated or truncated:
        break
print(info["forward_distance"], info["fell"])
env.close()
```

Faster start-up when you create many environments: generate the model once and point every environment at it.

```python
env = JansenBipedEnv(model_path="output_mujoco/jansen_biped.xml")
```

Gymnasium registry (needs `gymnasium`):

```python
import gymnasium as gym, jansen_biped.mujoco_env          # importing registers JansenBiped-v0
env = gym.make("JansenBiped-v0", render_mode="human")
```

## 4. Creating the environment

`JansenBipedEnv(config=None, render_mode=None, **overrides)`. Options live in `EnvConfig`; pass an `EnvConfig` or keyword overrides
(`JansenBipedEnv(max_episode_steps=2000, terminate_on_fall=False)`). Unknown names raise `TypeError`.

| Option | Default | Meaning |
|---|---|---|
| `model_path` | `None` | A generated `jansen_biped.xml`. `None` builds from `params` in memory. If a `crank_sweep.json` and report sit next to the XML they are used. |
| `params` | `None` | A `MujocoParams` (geometry, masses, friction; see README.md, section "Model parameters"). Used only when `model_path` is `None`. |
| `frame_skip` | 10 | Physics steps per control step. Control period = `frame_skip` x timestep = 10 ms by default. |
| `max_episode_steps` | 1000 | Truncation after this many control steps (10 s). `None` = never. |
| `max_crank_rpm` | 60 | Action +1 commands this crank speed. |
| `terminate_on_fall` | `True` | `False`: falling never ends the episode (see section 8). |
| `fall_roll_deg`, `fall_pitch_deg` | 45 | Fallen if the chassis tilts further than this. |
| `min_height_ratio` | 0.6 | Fallen if the chassis height drops below this fraction of its starting height. |
| `randomize_crank_phase` | `True` | Start at a random point of the gait cycle (needs the crank sweep). |
| `orientation_noise_deg` | 1.0 | Random roll/pitch at reset. |
| `velocity_noise` | 0.01 | Random initial velocity (m/s, rad/s). |
| `reset_height_margin_m` | 0.003 | Extra drop height so no start pose is inside the floor. |
| `lin_vel_scale`, `ang_vel_scale`, `height_scale` | 5, 0.5, 20 | Observation scaling (section 6). |
| `contact_threshold_n` | 0.05 | Touch force above which a foot counts as in contact. |
| `forward_speed_ref` | 0.15 | Forward speed (m/s) that scores 1 in the reward. |
| `w_forward`, `w_alive`, `w_tilt`, `w_lateral`, `w_energy`, `w_smooth`, `fall_penalty` | 1, 0.05, 0.5, 0.5, 0.002, 0.01, 5 | Reward weights (section 7). |
| `render_width`, `render_height`, `render_camera` | 640, 480, `"side"` | For `rgb_array` rendering. |

**Changing the robot** is done through `MujocoParams` (or the `--linkage-size`, `--ratio`, `--foot-width`, `--payload-mass`,
`--floor-friction` flags of the demos):

```python
from jansen_biped.mujoco_model import MujocoParams
from jansen_biped.mujoco_env import JansenBipedEnv
env = JansenBipedEnv(params=MujocoParams(foot_halfwidth_mm=20.0, payload_mass_kg=0.3))   # 40 mm wide feet, battery
```

## 5. Action space

`Box(-1, 1, shape=(1,), float32)`: the commanded crank speed as a fraction of `max_crank_rpm`.
`+1` = full speed forward, `0` = hold still, `-1` = full speed backward. Values outside [-1, 1] are clipped; a wrong
shape raises `ValueError`. With the default velocity-servo model the number sets the servo target; if you generated the model
with `--actuator torque` the action scales the torque limit instead.

## 6. Observation space

`Box(-inf, inf, shape=(16,), float32)`. Names are in `mujoco_env.OBS_NAMES`.

| Index | Name | Meaning | Scaling |
|---|---|---|---|
| 0, 1 | `sin_crank`, `cos_crank` | Crank angle (gait phase) | none |
| 2 | `crank_speed` | Crank speed | divided by the max speed |
| 3 | `height_error` | Chassis height minus its start height | x `height_scale` (20, i.e. 5 cm = 1) |
| 4 to 6 | `gravity_body_*` | World "down" in the chassis frame (upright = 0, 0, -1) | none |
| 7 to 9 | `lin_vel_body_*` | Chassis velocity in its own frame | x `lin_vel_scale` |
| 10 to 12 | `ang_vel_body_*` | Chassis angular velocity in its own frame | x `ang_vel_scale` |
| 13, 14 | `contact_left`, `contact_right` | Foot touching the floor (0/1) | none |
| 15 | `previous_action` | The last action sent | none |

On a physical robot these would come from an encoder (crank), an IMU (gravity, angular velocity), a state estimator
(velocity, height) and foot contact sensors. Several of them have no real-world sensor yet; see section 13.

## 7. Reward

Per control step, `reward = sum of terms`, and the individual terms are returned in `info["reward_terms"]`:

| Term | Formula | Default weight |
|---|---|---|
| `forward` | `w_forward * clip(vx / forward_speed_ref, -1, 2)` (world-frame forward velocity) | 1.0 |
| `alive` | `w_alive` every step | 0.05 |
| `tilt` | `-w_tilt * (roll^2 + pitch^2)` (radians) | 0.5 |
| `lateral` | `-w_lateral * abs(vy) / forward_speed_ref` | 0.5 |
| `energy` | `-w_energy * mean(abs(crank_torque * crank_speed))` over the control step (watts) | 0.002 |
| `smooth` | `-w_smooth * (action - previous_action)^2` | 0.01 |
| `fall` | `-fall_penalty` once, only when a fall terminates the episode | 5.0 |

`forward_speed_ref` (0.15 m/s) is roughly the speed of a robot that advances one foot-path length per crank turn at 60 rpm
(about 0.17 m per turn at the default size, with no slip). To change the objective, change the weights, or subclass the environment
and override `step`.

## 8. Episodes: reset, termination, and ending a run yourself

**Reset** (`env.reset(seed=None, options=None)`) returns `(obs, info)`. By default it starts the gait at a random phase (all 20 pin
angles are set consistently from the stored crank sweep, so the loops are closed), adds small tilt and velocity noise, and drops the
robot from a few millimetres. `options={"crank_phase_deg": 90}` picks the phase, for reproducible demos. A given `seed` always
produces the same start.

**Ending an episode.**
* `terminated=True`: the robot fell (tilt beyond the limits, or height below `min_height_ratio`), or the physics produced
  non-finite numbers (`info["nonfinite"]`).
* `truncated=True`: `max_episode_steps` reached.
* `info["fell"]` is set as soon as a fall has happened, whether or not it terminates the episode.

**Ending a run yourself ("manual end").** By default a run stops by itself when the robot falls or time runs out. With manual end
it never does: falling is recorded but does not stop anything, there is no time limit, and the run ends only when **you close the
viewer window or press Ctrl-C** (or if the physics goes non-finite, which cannot be continued). The same flag exists everywhere:

| Where | How |
|---|---|
| `mujoco_run run` | `--manual-end` (ignores `--seconds`; log and summary are still written on exit; `ended_by` in the summary says why it ended) |
| `rl_demo deploy` | `--manual-end`, optionally `--no-viewer` (then Ctrl-C ends it) |
| `mpc_demo` | `--manual-end` |
| Your own code | `JansenBipedEnv(terminate_on_fall=False, max_episode_steps=None)` and `run_loop(env, act_fn, manual_end=True)` |

Related option: `--reset-on-fall` (deploy and MPC) resets the robot after each fall and keeps going, useful for watching many attempts.

`run_loop(env, act_fn, *, max_steps=None, manual_end=False, reset_on_fall=False, realtime=True, seed=0)` is the helper all demos use.
It returns a dict with `ended_by` (`"robot fell"`, `"time limit"`, `"step limit"`, `"viewer closed by user"`, `"Ctrl-C"` or
`"non-finite state ..."`), the number of steps, falls, returns and the final `info`.

## 9. Saving, restoring and planning with the state

```python
s = env.save_state()          # dict of copies: qpos, qvel, solver warm start, time, control, step counter, last action
...                           # step the environment however you like
env.load_state(s)             # exact return to that moment
```

This is what makes model-based control possible: copy the real robot's state into a second environment, try action sequences there,
and keep only the best first action. See `jansen_biped/mpc_demo.py`. Rollouts are deterministic for a given state and action sequence.

## 10. Rendering

* `render_mode="human"`: opens the MuJoCo passive viewer; call `env.render()` after each step. `env.viewer_running()` turns `False`
  when the user closes the window (this is how manual end detects it).
* `render_mode="rgb_array"`: `env.render()` returns an `(height, width, 3)` uint8 image from the `side` camera, which follows the robot.

## 11. The demos

All demos share the environment flags (`--model`, `--linkage-size`, `--ratio`, `--foot-width`, `--payload-mass`, `--floor-friction`,
`--frame-skip`, `--max-episode-steps`, `--max-crank-rpm`, `--seed`).

### 11.1 Reinforcement learning (`jansen_biped.rl_demo`)

```bash
# Train. ARS needs only numpy; PPO needs stable-baselines3.
python -m jansen_biped.rl_demo train --algo ars --out runs/ars --iterations 30 --directions 8 --top 4
python -m jansen_biped.rl_demo train --algo ppo --out runs/ppo --timesteps 200000 --n-envs 4

# Evaluate headlessly, with baselines to compare against.
python -m jansen_biped.rl_demo eval --policy runs/ars/ars_policy.npz --episodes 5
python -m jansen_biped.rl_demo eval --policy constant:0.5
python -m jansen_biped.rl_demo eval --policy random

# Deploy: run a saved policy live. Close the window to stop.
python -m jansen_biped.rl_demo deploy --policy runs/ars/ars_policy.npz --manual-end
```

* **ARS** (Augmented Random Search) trains a *linear* policy `action = clip(W @ normalized_obs)` by trying random perturbations of `W`
  and stepping toward the better ones. Outputs in `--out`: `ars_policy.npz` (best by evaluation return), `ars_policy_last.npz`,
  `training_log.csv` (iteration, evaluation return, steps survived, forward distance, time). It is gradient-free, so it is easy to read.
* **PPO** trains a small MLP with Stable-Baselines3 and saves `ppo_policy.zip`. `--tensorboard` writes logs under `--out/tb`.
* **eval** prints each episode's return, how long it lasted, how far it moved, whether it fell, and the peak crank torque. `--json` saves them.
* **deploy** loads the policy and runs it in the viewer in real time (`--fast` to not slow it down). `--steps N` stops after N control steps.
* A policy is any object with `act(obs) -> array of shape (1,)`. Policy files: `.npz` for ARS, `.zip` for PPO.

**What to expect.** Not walking. Typical outcomes are a fall within seconds and returns near zero or negative. The `constant` baseline is
the thing to beat: if your trained policy does not outlast or outrun `constant:0.5` it has learned nothing useful. ARS training curves
on a problem like this are noisy; a few dozen iterations show you the plumbing, not convergence.

### 11.2 Model-based control (`jansen_biped.mpc_demo`)

```bash
python -m jansen_biped.mpc_demo --steps 300
python -m jansen_biped.mpc_demo --viewer --manual-end
python -m jansen_biped.mpc_demo --samples 48 --horizon 20 --sigma 0.3
```

Sampling MPC (MPPI-style). Each control step: copy the real state into a second environment, sample `--samples` crank-speed sequences of
length `--horizon` around the current plan, roll each out in the second environment, weight them by total reward
(`--temperature`: lower is greedier), blend, apply the first action, shift the plan. There is no training; the model is the simulator.

Cost: each control step simulates `samples x horizon x frame_skip` physics steps, so it is slower than real time on large settings. The
summary prints the mean planning time per step. Output: a JSON summary (`--json` to save it) with `ended_by`, steps, falls, distance.

**What to expect.** It plans only for the next `horizon x 10 ms` (0.12 s by default), which is too short to anticipate a tip-over, so it
is expected to fall. It is a template for how to do planning with the environment, not a solution.

## 12. Writing your own controller

A policy only needs `act(obs)`:

```python
import numpy as np
from jansen_biped.mujoco_env import JansenBipedEnv, evaluate, run_loop

class MyPolicy:
    def act(self, obs):                      # obs: float32[16]
        speed = 0.5 + 0.2 * obs[13]          # example: speed up while the left foot is down
        return np.array([speed], dtype=np.float32)

env = JansenBipedEnv()
print(evaluate(env, MyPolicy().act, episodes=3))             # headless metrics
run_loop(JansenBipedEnv(render_mode="human"), MyPolicy().act, manual_end=False)   # live
```

**Any RL library** that speaks Gymnasium works. With Stable-Baselines3 directly:

```python
from stable_baselines3 import PPO
from jansen_biped.mujoco_env import JansenBipedEnv
env = JansenBipedEnv(model_path="output_mujoco/jansen_biped.xml")
model = PPO("MlpPolicy", env, verbose=1).learn(100_000)
model.save("ppo_policy")
```

For parallel environments use `make_vec_env(lambda: JansenBipedEnv(model_path=...), n_envs=8)`; give every process the generated
`model_path` so the model is not rebuilt in each worker. Seed with `env.reset(seed=...)`; the same seed gives the same start state.

**Custom reward or observations:** subclass and override.

```python
class MyEnv(JansenBipedEnv):
    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        r += 0.1 * info["forward_velocity"]             # extra shaping, using the info dict
        return obs, r, term, trunc, info
```

## 13. Deployment and what transfers to hardware

"Deploy" in this repository means: load a saved policy and run it in the same environment, live. Nothing here talks to a real motor.
If you later move a policy to hardware:

* The policy output is a crank **speed command**; the real motor driver must track it at roughly the 100 Hz control rate used here.
* The observation needs sensors that the current design does not have: foot contact (touch sensors), chassis height and linear velocity
  (a state estimator). Train on a reduced observation if you cannot measure everything.
* An ARS policy is just `clip(W @ ((obs - mean) / (std + 1e-8)), -1, 1)`, with `W`, `mean`, `std` stored in the `.npz`, so it can be
  reimplemented in a few lines of any language.
* The model's friction, damping, motor and masses are assumptions (see README.md, section "Model limits and assumptions"). Treat simulated results as relative, not absolute.

## 14. Performance and reproducibility

* Reproducibility: pass `seed` to `reset`; the demos take `--seed`. Physics is deterministic for a fixed state and action sequence.
* Speed depends on your machine and settings; it was not measured here. `frame_skip` trades control rate for speed; many parallel
  environments and a generated `model_path` help training; MPC cost scales with `samples x horizon`.
* The model is built in memory once per process and cached; building takes a few seconds.

## 15. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `mujoco_run check` reports MISMATCH on nq/nv/neq | The XML and report are out of step; regenerate both with `mujoco_model`. |
| Loop-closure gap above tolerance | Send the output; it means MuJoCo reads the constraints differently from the offline check. |
| `Error building the model: ... larger --linkage-size` | The motor and gears do not fit the chosen linkage size; see README.md, section "Model limits and assumptions". |
| NaN or "non-finite state" | Numerical instability. Lower the timestep (`mujoco_model --timestep 0.0005`) or raise pin damping. The run ends because it cannot continue. |
| Robot falls on every episode | Expected for this design. Try `--foot-width 40` and `--payload-mass` to see how the answer changes. |
| Viewer does not open on macOS | Use `mjpython` instead of `python`. |
| `ImportError: ... stable-baselines3` | `pip install stable-baselines3`, or use `--algo ars`. |
| `gymnasium.make` cannot find the id | `import jansen_biped.mujoco_env` first (registration happens on import). |

## 16. Where things are

| File | Purpose |
|---|---|
| `jansen_biped/mujoco_model.py` | Generates the MJCF model, report and crank sweep |
| `jansen_biped/mujoco_verify.py` | Offline checker for the generated XML |
| `jansen_biped/mujoco_run.py` | `check` and `run` (with `--manual-end`) against real MuJoCo |
| `jansen_biped/mujoco_env.py` | The Gymnasium environment, `run_loop`, `evaluate` |
| `jansen_biped/rl_demo.py` | ARS and PPO training, evaluation, deployment |
| `jansen_biped/mpc_demo.py` | Sampling MPC |
| `tests/test_mujoco_env.py`, `tests/fake_mujoco.py` | Offline tests and the fake simulator |
