# Jansen biped: control toolkit (MuJoCo)

Everything you need to set up a controller for the Jansen-linkage biped in simulation, whether you are training an AI policy or
building a model-based controller. It contains the simulation model generator, the Gymnasium environment, two worked demos, and tests.
It does **not** contain any CAD, drawing, printing or assembly files.

## The robot in one paragraph

A small two-legged walker. One motor turns one shared crankshaft; both legs are Jansen linkages driven from it, with the two cranks
180 degrees apart. Everything else is passive pin joints. So **the only thing a controller decides is how fast, and in which
direction, the crank turns** (one number, from -1 to 1, every 10 ms). The walking motion comes from the mechanism, and the controller
cannot place feet or shift weight.

> **Read this before you start**
> * The two demos (RL and model-based) are **demonstrations of the setup, not working walkers**. They are expected to fall over.
>   They show how to train, save, load, plan, evaluate and deploy against this environment.
> * The code was written and unit-tested **without MuJoCo, Gymnasium or Stable-Baselines3 installed**; the tests use a fake
>   kinematic stand-in. Nobody has yet confirmed it against real MuJoCo. **Your first step is `mujoco_run check` (below).** If it reports
>   anything, tell the maintainer.
> * Model numbers (motor, masses, friction) are placeholders; see "Model limits and assumptions".

## What is in this package

```
README.md                    this file
NOTICE.md                    attribution for the upstream linkage equations
requirements.txt             numpy (needed for everything)
requirements-mujoco.txt      mujoco, gymnasium, stable-baselines3 (simulation and training)
docs/ENVIRONMENT_GUIDE.md    full reference for the environment, demos and writing your own controller
jansen_biped/
  mujoco_run.py              `check` (validate against real MuJoCo) and `run` (drive the crank, record, --manual-end)
  mujoco_env.py              the Gymnasium environment JansenBipedEnv, run_loop, evaluate
  rl_demo.py                 RL demo: train (ARS or PPO), eval, deploy
  mpc_demo.py                model-based demo: sampling MPC
  mujoco_verify.py           offline checker for a generated model (no MuJoCo needed)
  kinematics.py              the validated linkage equations (shared by everything)
  drive_layout.py            housing/gearbox layout; the model takes masses and positions from it
  layer_assignment.py        which link sits in which layer (affects positions in the model)
tests/                       offline tests and the fake simulator
```

## Setup

Python 3.10 or newer (developed on 3.12).

```bash
python -m venv .venv && source .venv/bin/activate #or conda create --name myvenv python=3.12 && conda activate myvenv
pip install -r requirements.txt
```

Run every command from the folder that contains `jansen_biped/`.
macOS: start the live viewer with `mjpython` instead of `python`. Linux headless rendering: set `MUJOCO_GL=egl` or `osmesa`.

## Your first ten minutes

```bash
# 1. watch it. The crank turns at 30 rpm. Close the window to end it (the robot may fall; it keeps going until you close it).
python -m jansen_biped.mujoco_run run --model output_mujoco/jansen_biped.xml --viewer --manual-end

# 2. get a baseline number to beat: a constant crank speed
python -m jansen_biped.rl_demo eval --policy constant:0.5 --episodes 5
```

Then in Python:

```python
from jansen_biped.mujoco_env import JansenBipedEnv
env = JansenBipedEnv()                                   # or JansenBipedEnv(model_path="output_mujoco/jansen_biped.xml")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step([0.5])
```

## Pick your route

### A. Train an AI policy

```bash
python -m jansen_biped.rl_demo train --algo ars --out runs/ars --iterations 30        # needs mujoco, not PyTorch
python -m jansen_biped.rl_demo train --algo ppo --out runs/ppo --timesteps 200000     # Stable-Baselines3
python -m jansen_biped.rl_demo eval   --policy runs/ars/ars_policy.npz --episodes 5   # headless metrics
python -m jansen_biped.rl_demo deploy --policy runs/ars/ars_policy.npz --manual-end   # live, in the viewer
```

`ars_policy.npz` is a plain linear policy; `ppo_policy.zip` is a Stable-Baselines3 model. To use another algorithm or library, the
environment is a standard Gymnasium environment (`JansenBiped-v0` after `import jansen_biped.mujoco_env`). For parallel training, give
every worker the generated `model_path` so the model is not rebuilt in each process. Change the reward by editing weights in `EnvConfig`
or by subclassing (guide, sections 7 and 12).

### B. Build a model-based controller

`python -m jansen_biped.mpc_demo --viewer --manual-end` runs a sampling-based MPC that plans with the MuJoCo model itself. The pieces you
reuse:

* `env.save_state()` / `env.load_state(s)`: exact snapshot and restore of the simulator, so a planner can try action sequences in a
  second environment and keep the best first action.
* `SamplingMPC` in `mpc_demo.py`: replace its `plan(state)` with your own planner or learned dynamics model.
* The environment's reward is the planning objective; change it in `EnvConfig` if you need a different one.

### C. Write your own controller

A policy is any object with `act(obs) -> array of shape (1,)` in [-1, 1]. Evaluate it with `evaluate(env, policy.act)` or run it live with
`run_loop(env, policy.act, ...)` (both in `mujoco_env.py`). Worked example: guide, section 12.

## The controller interface at a glance

| | |
|---|---|
| Action | 1 number in [-1, 1]: crank speed as a fraction of 60 rpm (negative = backwards) |
| Observation | 16 numbers: crank angle (sin, cos), crank speed, height error, gravity in body frame (3), body-frame linear and angular velocity (6), left and right foot contact, previous action |
| Control rate | 100 Hz (10 ms) |
| Reward | forward speed, small alive bonus, minus tilt, sideways speed, energy and action change; fall penalty |
| Episode ends | when the robot falls (tilt over 45 degrees or height under 60 %), after 1000 steps (10 s), or if the physics goes non-finite |

Exact definitions, scalings and every option are in `docs/ENVIRONMENT_GUIDE.md`.
A positive crank rotation walks the robot toward +x. Frame: x forward, y left, z up, SI units.

## Ending a run yourself

By default runs stop on their own when the robot falls or time runs out. Add `--manual-end` and a run never stops by itself: falling is
recorded but does not end it, and there is no time limit. It ends when you **close the viewer window or press Ctrl-C**, and logs and
summaries are still written. Available on `mujoco_run run`, `rl_demo deploy` and `mpc_demo`. `--reset-on-fall` instead resets the robot
after each fall and carries on.


## Model limits and assumptions

* **Fit limits.** The generator builds a housing around the motor and gears, and refuses with an explanatory message when they do not
  fit. With the default 42 x 45 mm motor, the smallest linkage size found to work (scanned in steps of 0.05) was about 1.05 for 7:1 and
  1.2 for 15:1. With stage-1 module 1.0 the largest motor diameter found to fit (1 mm steps) was 43 mm; `module= 1.25` raised that to
  54 mm (7:1) and 60 mm (15:1).
* **Placeholders to replace with real data:** motor mass and rotor inertia, no battery or electronics (`payload_mass_kg`), friction and
  damping values, and printed-part masses estimated from volume. Treat simulated results as relative, not absolute.
* **Simplifications:** the gearbox is lumped (the actuator drives the crankshaft directly; rotor and gear inertia appear as reflected
  inertia), links do not collide with each other or the floor, and the feet are rounded 3 mm-wide contacts by default. With so little
  lateral support.
* **Mechanism fact worth knowing:** bars b, d, e form a rigid triangle and so do g, h, i, so those pairs never move relative to each other.
  This is correct, not a modelling error.

## Suggested team workflow

* After any model change, run `mujoco_run check` before training.
* Record the exact command and seed with every result. Runs are reproducible for a given seed and model.
* Always compare against `rl_demo eval --policy constant:0.5` (and `random`); a policy that cannot beat it has learned nothing useful.
* `jansen_biped/mujoco_run.py run` reports peak crank torque and the motor-side equivalent, which is useful input for motor selection.

## Tests

```bash
python -m unittest discover -s tests
```

These check the generator, the offline model verifier (including loop closures over a full crank turn), the environment logic, the
ending rules, and the demos' plumbing against a fake kinematic simulator. They do **not** test real MuJoCo physics or real
Stable-Baselines3 training; a test that does is included and runs automatically when `mujoco` is installed.

## Troubleshooting

See `docs/ENVIRONMENT_GUIDE.md`, section 15. The most common first problems: `mujoco` not installed (`pip install -r
requirements-mujoco.txt`), the viewer not opening on macOS (use `mjpython`), and `mujoco_run check` reporting a mismatch (regenerate the
model and report the output).

## Licence and attribution

The linkage kinematics are a Python port of equations from an MIT-licensed OpenSCAD project; see `NOTICE.md`.
