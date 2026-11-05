"""Gymnasium environment for the Jansen biped in MuJoCo.

    from jansen_biped.mujoco_env import JansenBipedEnv
    env = JansenBipedEnv()                       # generates the default model in memory
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

The robot has ONE actuator, the shared crankshaft. The action is a single number in
[-1, 1]: the commanded crank speed as a fraction of ``max_crank_rpm`` (negative runs the
crank backwards). Everything else (the 20 passive pin joints and the 10 loop closures)
is the mechanism. See docs/ENVIRONMENT_GUIDE.md for the full description.

Works with or without ``gymnasium``: if it is not installed a minimal compatible
interface is used, so numpy-only tools (the ARS trainer, tests) still run. Install
gymnasium to use Stable-Baselines3 or ``gymnasium.make("JansenBiped-v0")``.
"""
from __future__ import annotations

import functools
import json
import math
import time
import types
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:                                   # pragma: no cover - depends on the environment
    import gymnasium as gym
    from gymnasium import spaces
    HAVE_GYMNASIUM = True
except ImportError:                    # minimal stand-ins so the module works without gymnasium
    gym = None
    HAVE_GYMNASIUM = False

    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.dtype = np.dtype(dtype)
            self.shape = tuple(shape) if shape is not None else np.shape(low)
            self.low = np.full(self.shape, low, dtype=self.dtype) if np.isscalar(low) else np.asarray(low, self.dtype)
            self.high = np.full(self.shape, high, dtype=self.dtype) if np.isscalar(high) else np.asarray(high, self.dtype)

        def sample(self, rng=None):
            rng = rng or np.random.default_rng()
            lo = np.where(np.isfinite(self.low), self.low, -1.0)
            hi = np.where(np.isfinite(self.high), self.high, 1.0)
            return rng.uniform(lo, hi).astype(self.dtype)

        def contains(self, x):
            x = np.asarray(x)
            return x.shape == self.shape and bool(np.all(x >= self.low) and np.all(x <= self.high))

    class _Env:
        metadata: dict = {}

        def __init__(self):
            self.np_random = np.random.default_rng()

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.np_random = np.random.default_rng(seed)

        def close(self):
            pass

    spaces = types.SimpleNamespace(Box=_Box)

_Base = gym.Env if HAVE_GYMNASIUM else _Env

OBS_NAMES = (
    "sin_crank", "cos_crank", "crank_speed", "height_error",
    "gravity_body_x", "gravity_body_y", "gravity_body_z",
    "lin_vel_body_x", "lin_vel_body_y", "lin_vel_body_z",
    "ang_vel_body_x", "ang_vel_body_y", "ang_vel_body_z",
    "contact_left", "contact_right", "previous_action",
)


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """Every tunable of the environment. Pass as ``JansenBipedEnv(config=...)`` or as keyword overrides."""
    # model
    model_path: Optional[str] = None          # a generated jansen_biped.xml; None -> build from `params` in memory
    params: Optional["MujocoParams"] = None   # model parameters when model_path is None (default MujocoParams())
    # time
    frame_skip: int = 10                      # physics steps per control step (control dt = frame_skip * timestep)
    max_episode_steps: Optional[int] = 1000   # truncation; None = never truncate
    # action
    max_crank_rpm: float = 60.0               # action +1 commands this crank speed
    # termination
    terminate_on_fall: bool = True            # False: falling never ends the episode (the user ends the run)
    fall_roll_deg: float = 45.0
    fall_pitch_deg: float = 45.0
    min_height_ratio: float = 0.6             # fallen if chassis height < ratio * initial height
    # reset
    randomize_crank_phase: bool = True        # start at a random point of the gait cycle (needs crank_sweep)
    orientation_noise_deg: float = 1.0
    velocity_noise: float = 0.01
    reset_height_margin_m: float = 0.003      # drop height so random phases never start inside the floor
    # observation scaling
    lin_vel_scale: float = 5.0
    ang_vel_scale: float = 0.5
    height_scale: float = 20.0
    contact_threshold_n: float = 0.05
    # reward
    forward_speed_ref: float = 0.15           # m/s that counts as "1" for the forward term
    w_forward: float = 1.0
    w_alive: float = 0.05
    w_tilt: float = 0.5
    w_lateral: float = 0.5
    w_energy: float = 0.002                   # per watt
    w_smooth: float = 0.01
    fall_penalty: float = 5.0                 # one-off, only when a fall terminates the episode
    # rendering
    render_width: int = 640
    render_height: int = 480
    render_camera: str = "side"


@functools.lru_cache(maxsize=8)
def _build_cached(params_json: str):
    from .mujoco_model import MujocoParams, build_mujoco_model
    return build_mujoco_model(MujocoParams(**json.loads(params_json)))


def quat_rotate_inv(q, v):
    """Express world-frame vector v in the frame of a body with orientation q = (w, x, y, z)."""
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R.T @ np.asarray(v, dtype=float)


def quat_from_small_angles(rx, ry, rz):
    """Unit quaternion (w, x, y, z) for a small rotation vector."""
    ang = math.sqrt(rx * rx + ry * ry + rz * rz)
    if ang < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    s = math.sin(ang / 2) / ang
    return np.array([math.cos(ang / 2), rx * s, ry * s, rz * s])


def roll_pitch(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


# ----------------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------------

class JansenBipedEnv(_Base):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None, **overrides):
        super().__init__()
        cfg = config or EnvConfig()
        if overrides:
            unknown = set(overrides) - set(asdict(cfg))
            if unknown:
                raise TypeError(f"unknown EnvConfig option(s): {sorted(unknown)}")
            cfg = replace(cfg, **overrides)
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human' or 'rgb_array'")
        if cfg.frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        self.cfg = cfg
        self.render_mode = render_mode
        self._mj = _import_mujoco()
        self._load_model()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(OBS_NAMES),), dtype=np.float32)
        self._viewer = None
        self._renderer = None
        self._steps = 0
        self._prev_action = np.zeros(1, dtype=np.float32)
        self._fell = False

    # ---- construction -----------------------------------------------------------------------
    def _load_model(self):
        mj, cfg = self._mj, self.cfg
        sweep = report = None
        if cfg.model_path:
            path = Path(cfg.model_path)
            self.model = mj.MjModel.from_xml_path(str(path))
            for name, attr in (("crank_sweep.json", "sweep"), ("jansen_biped_mujoco_report.json", "report")):
                p = path.with_name(name)
                if p.exists():
                    if attr == "sweep":
                        sweep = json.loads(p.read_text())
                    else:
                        report = json.loads(p.read_text())
        else:
            from .mujoco_model import MujocoParams
            params = replace(cfg.params or MujocoParams(), sweep_step_deg=1.0)
            built = _build_cached(json.dumps(asdict(params), sort_keys=True))
            self.model = mj.MjModel.from_xml_string(built["xml"])
            sweep, report = built["sweep"], built["report"]
        m = self.model
        self.data = mj.MjData(m)
        self.report = report
        J, S = mj.mjtObj.mjOBJ_JOINT, mj.mjtObj.mjOBJ_SENSOR
        crank = mj.mj_name2id(m, J, "crank")
        if crank < 0:
            raise ValueError("model has no 'crank' joint; was it generated by jansen_biped.mujoco_model?")
        self._qadr, self._vadr = int(m.jnt_qposadr[crank]), int(m.jnt_dofadr[crank])
        self._sensor_adr = {}
        for n in ("crank_torque", "touch_L", "touch_R"):
            i = mj.mj_name2id(m, S, n)
            self._sensor_adr[n] = int(m.sensor_adr[i]) if i >= 0 else None
        self._is_velocity = m.nu > 0 and abs(float(m.actuator_biasprm[0, 2])) > 0
        self._ctrl_max = float(m.actuator_ctrlrange[0, 1]) if m.nu else 1.0
        self._omega_max = cfg.max_crank_rpm * 2.0 * math.pi / 60.0
        self._z_ref = float(m.qpos0[2])
        self._dt = float(m.opt.timestep) * cfg.frame_skip
        self._sweep = None
        if sweep is not None:
            idx, cols = [], []
            rows = np.array(sweep["hinge_angles_rad"])
            for col, name in enumerate(sweep["joint_names"]):
                j = mj.mj_name2id(m, J, name)
                if j >= 0:
                    idx.append(int(m.jnt_qposadr[j]))
                    cols.append(col)
            self._sweep = {"offsets": np.radians(np.array(sweep["crank_offset_deg"])), "rows": rows[:, cols], "qadr": np.array(idx),
                           "crank_col": [i for i, c in enumerate(cols) if sweep["joint_names"][c] == "crank"][0]}

    # ---- state helpers -----------------------------------------------------------------------
    @property
    def dt(self) -> float:
        """Control period in seconds."""
        return self._dt

    def save_state(self) -> dict:
        """Snapshot of everything that determines the future (for model-based planning)."""
        d = self.data
        return {"qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "warm": d.qacc_warmstart.copy(), "time": float(d.time),
                "ctrl": d.ctrl.copy(), "steps": self._steps, "prev_action": self._prev_action.copy(), "fell": self._fell}

    def load_state(self, s: dict) -> None:
        d = self.data
        d.qpos[:] = s["qpos"]
        d.qvel[:] = s["qvel"]
        d.qacc_warmstart[:] = s["warm"]
        d.ctrl[:] = s["ctrl"]
        d.time = s["time"]
        self._steps, self._prev_action, self._fell = s["steps"], s["prev_action"].copy(), s["fell"]
        self._mj.mj_forward(self.model, d)

    def _crank_speed(self) -> float:
        return float(self.data.qvel[self._vadr])

    def _torque(self) -> float:
        a = self._sensor_adr["crank_torque"]
        return float(self.data.sensordata[a]) if a is not None else float(self.data.actuator_force[0]) if self.model.nu else 0.0

    def _contacts(self):
        out = []
        for n in ("touch_L", "touch_R"):
            a = self._sensor_adr[n]
            out.append(1.0 if a is not None and float(self.data.sensordata[a]) > self.cfg.contact_threshold_n else 0.0)
        return out

    def _observation(self) -> np.ndarray:
        d, c = self.data, self.cfg
        q = d.qpos[3:7]
        ang = float(d.qpos[self._qadr])
        g_body = quat_rotate_inv(q, [0.0, 0.0, -1.0])
        v_body = quat_rotate_inv(q, d.qvel[0:3])
        w_body = d.qvel[3:6]
        cl, cr = self._contacts()
        obs = np.array([math.sin(ang), math.cos(ang), self._crank_speed() / self._omega_max,
                        (float(d.qpos[2]) - self._z_ref) * c.height_scale, *g_body, *(v_body * c.lin_vel_scale),
                        *(w_body * c.ang_vel_scale), cl, cr, float(self._prev_action[0])], dtype=np.float32)
        return obs

    # ---- gym API -------------------------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = self.np_random
        mj, m, d, c = self._mj, self.model, self.data, self.cfg
        mj.mj_resetData(m, d)
        d.qpos[:] = m.qpos0
        phase_deg = 0.0
        if c.randomize_crank_phase and self._sweep is not None:
            off = float(rng.uniform(0.0, 2.0 * math.pi))
            phase_deg = math.degrees(off)
            sw = self._sweep
            for k, qadr in enumerate(sw["qadr"]):
                col = np.unwrap(sw["rows"][:, k])
                d.qpos[qadr] = float(np.interp(off, sw["offsets"], col))
            d.qpos[2] = self._z_ref + c.reset_height_margin_m
        if options and "crank_phase_deg" in options and self._sweep is not None:      # explicit phase for reproducible demos
            off = math.radians(float(options["crank_phase_deg"]) % 360.0)
            sw = self._sweep
            for k, qadr in enumerate(sw["qadr"]):
                d.qpos[qadr] = float(np.interp(off, sw["offsets"], np.unwrap(sw["rows"][:, k])))
            d.qpos[2] = self._z_ref + c.reset_height_margin_m
            phase_deg = float(options["crank_phase_deg"])
        n = math.radians(c.orientation_noise_deg)
        d.qpos[3:7] = quat_from_small_angles(*(rng.normal(0.0, n, 3) * np.array([1.0, 1.0, 0.0])))
        d.qvel[0:6] += rng.normal(0.0, c.velocity_noise, 6)
        mj.mj_forward(m, d)
        self._steps = 0
        self._prev_action = np.zeros(1, dtype=np.float32)
        self._fell = False
        self._x_start = float(d.qpos[0])
        return self._observation(), {"crank_phase_deg": phase_deg, **self._info()}

    def _info(self, terms=None) -> dict:
        d = self.data
        roll, pitch = roll_pitch(d.qpos[3:7])
        return {"x": float(d.qpos[0]), "y": float(d.qpos[1]), "z": float(d.qpos[2]), "roll": roll, "pitch": pitch,
                "forward_velocity": float(d.qvel[0]), "forward_distance": float(d.qpos[0]) - getattr(self, "_x_start", 0.0),
                "crank_angle": float(d.qpos[self._qadr]), "crank_torque": self._torque(), "fell": self._fell,
                "nonfinite": False, "reward_terms": terms or {}, "time": float(d.time)}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        if a.size != 1:
            raise ValueError(f"action must contain exactly 1 number (crank speed command), got shape {np.shape(action)}")
        mj, m, d, c = self._mj, self.model, self.data, self.cfg
        if m.nu:
            d.ctrl[0] = float(a[0]) * (self._omega_max if self._is_velocity else self._ctrl_max)
        power = 0.0
        nonfinite = False
        for _ in range(c.frame_skip):
            mj.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)):
                nonfinite = True
                break
            power += abs(self._torque() * self._crank_speed())
        power /= c.frame_skip
        self._steps += 1
        if nonfinite:
            obs = np.zeros(len(OBS_NAMES), dtype=np.float32)
            info = self._info()
            info.update(nonfinite=True)
            return obs, -c.fall_penalty, True, False, info
        roll, pitch = roll_pitch(d.qpos[3:7])
        fell = (abs(roll) > math.radians(c.fall_roll_deg) or abs(pitch) > math.radians(c.fall_pitch_deg)
                or float(d.qpos[2]) < c.min_height_ratio * self._z_ref)
        self._fell = self._fell or fell
        vx, vy = float(d.qvel[0]), float(d.qvel[1])
        terminated = bool(fell and c.terminate_on_fall)
        terms = {
            "forward": c.w_forward * float(np.clip(vx / c.forward_speed_ref, -1.0, 2.0)),
            "alive": c.w_alive,
            "tilt": -c.w_tilt * (roll * roll + pitch * pitch),
            "lateral": -c.w_lateral * abs(vy) / c.forward_speed_ref,
            "energy": -c.w_energy * power,
            "smooth": -c.w_smooth * float((a[0] - self._prev_action[0]) ** 2),
            "fall": -c.fall_penalty if terminated else 0.0,
        }
        reward = float(sum(terms.values()))
        self._prev_action = a.copy()
        truncated = bool(c.max_episode_steps is not None and self._steps >= c.max_episode_steps and not terminated)
        return self._observation(), reward, terminated, truncated, self._info(terms)

    # ---- rendering --------------------------------------------------------------------------------------
    def render(self):
        mj = self._mj
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mj.Renderer(self.model, self.cfg.render_height, self.cfg.render_width)
            self._renderer.update_scene(self.data, camera=self.cfg.render_camera)
            return self._renderer.render()
        return None

    def viewer_running(self) -> bool:
        """False once the user has closed the viewer window (always True without a viewer)."""
        return self._viewer is None or bool(self._viewer.is_running())

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        super().close()


def _import_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError("The 'mujoco' package is required: pip install mujoco") from exc
    return mujoco


if HAVE_GYMNASIUM:                                          # pragma: no cover
    try:
        gym.register(id="JansenBiped-v0", entry_point="jansen_biped.mujoco_env:JansenBipedEnv")
    except Exception:
        pass


# ----------------------------------------------------------------------------
# shared loops used by the demos
# ----------------------------------------------------------------------------

def evaluate(env, act_fn: Callable[[np.ndarray], np.ndarray], episodes: int = 5, seed: int = 0) -> list[dict]:
    """Run whole episodes headlessly and return one metrics dict per episode."""
    out = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total, steps, torque_peak = 0.0, 0, 0.0
        while True:
            obs, r, term, trunc, info = env.step(act_fn(obs))
            total += r
            steps += 1
            torque_peak = max(torque_peak, abs(info["crank_torque"]))
            if term or trunc:
                break
        out.append({"episode": ep, "return": total, "steps": steps, "seconds": info["time"],
                    "forward_distance_m": info["forward_distance"], "fell": bool(info["fell"]),
                    "peak_crank_torque_Nm": torque_peak})
    return out


def run_loop(env, act_fn: Callable[[np.ndarray], np.ndarray], *, max_steps: Optional[int] = None, manual_end: bool = False,
             reset_on_fall: bool = False, realtime: bool = True, seed: int = 0, reset_options: Optional[dict] = None,
             log: Callable[[str], None] = print) -> dict:
    """Run a controller until it is over, with the same ending rules for every demo.

    * default: stops when the episode terminates (fall) or truncates (time limit).
    * ``reset_on_fall``: after a fall, reset the robot and keep going.
    * ``manual_end``: never stops by itself. Falling does not end the run, and the only ways out are
      closing the viewer window, Ctrl-C, ``max_steps`` (if you also give one) or a non-finite state.
      Build the env with ``terminate_on_fall=False, max_episode_steps=None`` for this mode.
    """
    obs, info = env.reset(seed=seed, options=reset_options)
    steps, falls, ended = 0, 0, None
    ep_return, returns = 0.0, []
    fell_reported = False
    open_episode = True
    viewer = env.render_mode == "human"
    try:
        while max_steps is None or steps < max_steps:
            if viewer and not env.viewer_running():
                ended = "viewer closed by user"
                break
            t0 = time.perf_counter()
            obs, r, term, trunc, info = env.step(act_fn(obs))
            ep_return += r
            steps += 1
            if viewer:
                env.render()
                if realtime:
                    time.sleep(max(0.0, env.dt - (time.perf_counter() - t0)))
            if info["fell"] and not fell_reported:
                falls += 1
                fell_reported = True
                if manual_end:
                    log(f"Robot fell at {info['time']:.2f} s; continuing because of manual-end mode.")
            if term or trunc:
                returns.append(ep_return)
                open_episode = False
                if info.get("nonfinite"):
                    ended = "non-finite state (model became numerically unstable)"
                    break
                if reset_on_fall or manual_end:
                    log(f"Episode ended after {steps} steps (return {ep_return:.2f}); resetting.")
                    obs, info = env.reset()
                    ep_return, fell_reported, open_episode = 0.0, False, True
                    continue
                ended = "robot fell" if info["fell"] else "time limit"
                break
    except KeyboardInterrupt:
        ended = "Ctrl-C"
    if ended is None:
        ended = "step limit"
    if open_episode:
        returns.append(ep_return)
    return {"ended_by": ended, "steps": steps, "control_seconds": steps * env.dt, "falls": falls,
            "returns": returns, "forward_distance_m": info["forward_distance"], "final_info": info}


# ----------------------------------------------------------------------------
# command-line helpers shared by rl_demo.py and mpc_demo.py
# ----------------------------------------------------------------------------

def add_env_args(ap) -> None:
    """Environment options common to every demo command."""
    g = ap.add_argument_group("environment")
    g.add_argument("--model", default=None, help="generated jansen_biped.xml; default: build the model in memory")
    g.add_argument("--linkage-size", type=float, default=2.5)
    g.add_argument("--ratio", choices=["7", "15"], default="7")
    g.add_argument("--foot-width", type=float, default=3.0, help="lateral foot width in mm")
    g.add_argument("--payload-mass", type=float, default=0.0, help="kg of battery/electronics")
    g.add_argument("--floor-friction", type=float, default=0.8)
    g.add_argument("--frame-skip", type=int, default=10)
    g.add_argument("--max-episode-steps", type=int, default=1000)
    g.add_argument("--max-crank-rpm", type=float, default=60.0)
    g.add_argument("--seed", type=int, default=0)


def config_from_args(args, *, manual_end: bool = False, randomize_phase: bool = True) -> EnvConfig:
    from .mujoco_model import MujocoParams
    params = MujocoParams(linkage_size=args.linkage_size, ratio=args.ratio, foot_halfwidth_mm=args.foot_width / 2.0,
                          payload_mass_kg=args.payload_mass, floor_friction=args.floor_friction)
    return EnvConfig(model_path=args.model, params=params, frame_skip=args.frame_skip,
                     max_episode_steps=None if manual_end else args.max_episode_steps,
                     max_crank_rpm=args.max_crank_rpm, terminate_on_fall=not manual_end,
                     randomize_crank_phase=randomize_phase)


def make_env(args, *, render: bool = False, manual_end: bool = False, **overrides) -> "JansenBipedEnv":
    """Build the environment from parsed demo arguments; model errors become readable exits."""
    from .drive_layout import LayoutError
    cfg = config_from_args(args, manual_end=manual_end)
    if overrides:
        cfg = replace(cfg, **overrides)
    try:
        return JansenBipedEnv(cfg, render_mode="human" if render else None)
    except ImportError as exc:
        raise SystemExit(f"{exc}\nInstall the simulation requirements with: pip install -r requirements-mujoco.txt")
    except (LayoutError, ValueError) as exc:
        raise SystemExit(f"Error building the model: {exc}")
