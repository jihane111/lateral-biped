"""A fake ``mujoco`` module, only for exercising jansen_biped/mujoco_run.py without MuJoCo.

It is NOT a physics engine. It parses the generated XML with the offline verifier's
parser, and when the crank turns it replays the saved crank sweep, so the whole
mechanism follows the crank kinematically and the chassis never moves. That is
enough to run every line of mujoco_run.py. It cannot show that the real MuJoCo API
behaves the way the script assumes; only running the real thing can.
"""
from __future__ import annotations

import json
import math
import types
from pathlib import Path

import numpy as np

from jansen_biped.mujoco_verify import Model

__version__ = "fake-0"
mjtObj = types.SimpleNamespace(mjOBJ_JOINT=3, mjOBJ_GEOM=5, mjOBJ_SENSOR=21, mjOBJ_EQUALITY=17)
mjtEq = types.SimpleNamespace(mjEQ_CONNECT=0)
_SENSOR_DIM = {"jointpos": 1, "jointvel": 1, "actuatorfrc": 1, "framepos": 3, "framequat": 4, "framelinvel": 3, "touch": 1}


class _Opt:
    pass


class MjModel:
    @classmethod
    def from_xml_path(cls, path):
        path = Path(path)
        return cls._from_text(path.read_text(), path.with_name("crank_sweep.json"))

    @classmethod
    def from_xml_string(cls, text):
        return cls._from_text(text, None)

    @classmethod
    def _from_text(cls, text, sweep_path):
        self = cls()
        self._fm = Model(text)
        fm = self._fm
        self.opt = _Opt()
        self.opt.timestep = float(__import__("xml.etree.ElementTree", fromlist=["x"]).fromstring(text).find("option").get("timestep"))
        self.body_names = ["world"] + fm.order
        self.body_id = {n: i for i, n in enumerate(self.body_names)}
        self.nbody = len(self.body_names)
        self.joint_names = list(fm.joints)
        self.njnt = len(self.joint_names)
        self.jnt_qposadr, self.jnt_dofadr = [], []
        nq = nv = 0
        for jn in self.joint_names:
            self.jnt_qposadr.append(nq)
            self.jnt_dofadr.append(nv)
            nq, nv = (nq + 7, nv + 6) if fm.joints[jn]["type"] == "free" else (nq + 1, nv + 1)
        self.nq, self.nv = nq, nv
        self.qpos0 = np.zeros(nq)
        self.qpos0[0:3] = fm.bodies["chassis"]["pos"]
        self.qpos0[3] = 1.0
        self.body_mass = np.zeros(self.nbody)
        for n, b in fm.bodies.items():
            self.body_mass[self.body_id[n]] = sum(float(g.get("mass", 0)) for g in b["geoms"])
        self.ngeom = 1 + sum(len(b["geoms"]) for b in fm.bodies.values())
        self.geom_names = ["floor"] + [g["name"] for b in fm.bodies.values() for g in b["geoms"]]
        self.neq = len(fm.connects)
        self.eq_type = np.zeros(self.neq, dtype=int)
        self.eq_obj1id = np.array([self.body_id[c["body1"]] for c in fm.connects])
        self.eq_obj2id = np.array([self.body_id[c["body2"]] for c in fm.connects])
        T0 = fm.transforms({})
        self.eq_data = np.zeros((self.neq, 11))
        for i, c in enumerate(fm.connects):
            a1 = np.array([float(x) for x in c["anchor"].split()])
            R1, p1 = T0[c["body1"]]
            R2, p2 = T0[c["body2"]]
            self.eq_data[i, 0:3] = a1
            self.eq_data[i, 3:6] = R2.T @ (p1 + R1 @ a1 - p2)
        self.nu = len(fm.actuators)
        self.actuator_biasprm = np.zeros((self.nu, 10))
        self.actuator_ctrlrange = np.zeros((self.nu, 2))
        self.actuator_forcerange = np.zeros((self.nu, 2))
        self.actuator_forcelimited = np.zeros(self.nu, dtype=bool)
        for k, (tag, at) in enumerate(fm.actuators):
            self.actuator_ctrlrange[k] = [float(x) for x in at["ctrlrange"].split()]
            if tag == "velocity":
                self.actuator_biasprm[k, 2] = -float(at["kv"])
                self.actuator_forcerange[k] = [float(x) for x in at["forcerange"].split()]
                self.actuator_forcelimited[k] = True
        self.sensor_names = [at["name"] for _, at in fm.sensors]
        self.nsensor = len(self.sensor_names)
        self.sensor_adr, adr = [], 0
        for tag, at in fm.sensors:
            self.sensor_adr.append(adr)
            adr += _SENSOR_DIM[tag]
        self.nsensordata = adr
        self._sweep = json.loads(sweep_path.read_text()) if sweep_path is not None and sweep_path.exists() else None
        return self


class MjData:
    def __init__(self, m):
        self.m = m
        self.qpos = m.qpos0.copy()
        self.qvel = np.zeros(m.nv)
        self.ctrl = np.zeros(m.nu)
        self.sensordata = np.zeros(m.nsensordata)
        self.actuator_force = np.zeros(m.nu)
        self.qacc_warmstart = np.zeros(m.nv)
        self.xpos = np.zeros((m.nbody, 3))
        self.xmat = np.tile(np.eye(3).ravel(), (m.nbody, 1))
        self.time = 0.0
        self.ncon = 0
        self.contact = []
        self.ne = 0
        self.efc_pos = np.zeros(0)


def mj_name2id(m, kind, name):
    if kind == mjtObj.mjOBJ_JOINT:
        return m.joint_names.index(name) if name in m.joint_names else -1
    if kind == mjtObj.mjOBJ_GEOM:
        return m.geom_names.index(name) if name in m.geom_names else -1
    if kind == mjtObj.mjOBJ_SENSOR:
        return m.sensor_names.index(name) if name in m.sensor_names else -1
    return -1


def mj_id2name(m, kind, i):
    return m._fm.connects[i]["name"] if kind == mjtObj.mjOBJ_EQUALITY else None


def _hinge_dict(m, d):
    return {jn: float(d.qpos[m.jnt_qposadr[k]]) for k, jn in enumerate(m.joint_names) if m._fm.joints[jn]["type"] == "hinge"}


def mj_forward(m, d):
    T = m._fm.transforms(_hinge_dict(m, d))
    for n, (R, p) in T.items():
        i = m.body_id[n]
        d.xpos[i] = p
        d.xmat[i] = R.ravel()


def mj_resetData(m, d):
    d.qpos[:] = m.qpos0
    d.qvel[:] = 0
    d.time = 0.0
    mj_forward(m, d)


TILT_AT = None          # tests set this to a simulated time after which the chassis is knocked over


def mj_step(m, d):
    dt = m.opt.timestep
    if TILT_AT is not None and d.time >= TILT_AT:
        d.qpos[3:7] = [math.cos(math.radians(35)), math.sin(math.radians(35)), 0.0, 0.0]      # 70 deg roll
    crank = m.joint_names.index("crank")
    omega = float(d.ctrl[0]) if m.nu else 0.0
    q = d.qpos[m.jnt_qposadr[crank]] + omega * dt
    d.qpos[m.jnt_qposadr[crank]] = q
    d.qvel[m.jnt_dofadr[crank]] = omega
    sw = m._sweep
    if sw:                                            # kinematic replay of the whole mechanism from the crank angle
        rows = np.array(sw["hinge_angles_rad"])
        offs = np.radians(np.array(sw["crank_offset_deg"]))
        qq = (q % (2 * math.pi))
        for col, name in enumerate(sw["joint_names"]):
            if name == "crank":
                continue
            wrapped = np.unwrap(rows[:, col])
            d.qpos[m.jnt_qposadr[m.joint_names.index(name)]] = float(np.interp(qq, offs, wrapped))
    d.time += dt
    d.actuator_force[:] = 0.4 * math.sin(q)
    if "crank_torque" in m.sensor_names:
        d.sensordata[m.sensor_adr[m.sensor_names.index("crank_torque")]] = d.actuator_force[0]
    mj_forward(m, d)


class Renderer:
    def __init__(self, model, height, width):
        self.size = (height, width)

    def update_scene(self, data, camera=None):
        self.camera = camera

    def render(self):
        return np.zeros((*self.size, 3), dtype=np.uint8)

    def close(self):
        pass


class _FakeViewer:
    """A viewer window that 'closes' after `lifetime` syncs (what a user closing the window looks like)."""

    def __init__(self, lifetime):
        self.syncs = 0
        self.lifetime = lifetime
        self.closed = False

    def sync(self):
        self.syncs += 1

    def is_running(self):
        return not self.closed and self.syncs < self.lifetime

    def close(self):
        self.closed = True


class _ViewerModule(types.ModuleType):
    lifetime = 25

    def launch_passive(self, model, data):
        return _FakeViewer(self.lifetime)


viewer = _ViewerModule("mujoco.viewer")
