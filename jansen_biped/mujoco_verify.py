"""Verify a generated MJCF without MuJoCo.

The checker re-reads the XML text, builds its own body tree and does its own
forward kinematics (hinge rotations about the declared axes). It then checks:

* structure: unique names, resolvable references, identity body frames, counts;
* geometry: every bar's length, recovered *from the XML*, equals the validated
  linkage kinematics;
* closure: at every sampled crank angle (hinge angles from ``crank_sweep.json``)
  the ten ``connect`` constraints are satisfied, using MuJoCo's own semantics
  (the body-2 anchor is fixed at ``qpos0``);
* direction, floor clearance and mass sanity.

This does not replace compiling the model in MuJoCo (see ``mujoco_run check``);
it makes the generator's topology and angle derivation independently testable.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .kinematics import bar_lengths

# Attributes this generator is allowed to emit (guards against typos; not a full MJCF schema).
ALLOWED: dict[str, set[str]] = {
    "mujoco": {"model"}, "compiler": {"angle"},
    "option": {"timestep", "integrator", "gravity", "solver", "iterations", "tolerance"},
    "default": set(), "joint": {"name", "type", "axis", "damping", "armature", "frictionloss"},
    "geom": {"name", "type", "pos", "euler", "size", "fromto", "mass", "density", "rgba", "contype", "conaffinity",
             "friction", "material"},
    "asset": set(), "texture": {"name", "type", "builtin", "rgb1", "rgb2", "width", "height"},
    "material": {"name", "texture", "texrepeat", "reflectance"}, "worldbody": set(),
    "light": {"name", "pos", "dir", "directional"}, "body": {"name", "pos"}, "freejoint": {"name"},
    "camera": {"name", "mode", "pos", "xyaxes"}, "site": {"name", "pos", "size", "rgba"}, "equality": set(),
    "connect": {"name", "body1", "body2", "anchor", "solref", "solimp"}, "actuator": set(),
    "velocity": {"name", "joint", "kv", "ctrllimited", "ctrlrange", "forcelimited", "forcerange"},
    "motor": {"name", "joint", "gear", "ctrllimited", "ctrlrange"}, "sensor": set(),
    "jointpos": {"name", "joint"}, "jointvel": {"name", "joint"}, "actuatorfrc": {"name", "actuator"},
    "framepos": {"name", "objtype", "objname"}, "framequat": {"name", "objtype", "objname"},
    "framelinvel": {"name", "objtype", "objname"}, "touch": {"name", "site"},
}


@dataclass
class Check:
    name: str
    passed: bool
    value: Any = None
    limit: Any = None
    detail: str = ""

    def to_dict(self):
        return {"name": self.name, "passed": bool(self.passed), "value": self.value, "limit": self.limit,
                "detail": self.detail}


def _v(text: Optional[str], n: int = 3, default=0.0) -> np.ndarray:
    if text is None:
        return np.full(n, default, dtype=float)
    a = np.array([float(x) for x in text.split()], dtype=float)
    if a.size != n:
        raise ValueError(f"expected {n} numbers, got {text!r}")
    return a


def rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


class Model:
    """Minimal parsed view of the generated MJCF."""

    def __init__(self, xml_text: str):
        self.root = ET.fromstring(xml_text)
        self.errors: list[str] = []
        dj = self.root.find("./default/joint")
        self.default_joint = dict(dj.attrib) if dj is not None else {}
        self.bodies: dict[str, dict] = {}
        self.joints: dict[str, dict] = {}
        self.order: list[str] = []
        wb = self.root.find("worldbody")
        for el in wb.findall("body"):
            self._walk(el, None)
        self.connects = [dict(e.attrib) for e in self.root.findall("./equality/connect")]
        self.actuators = [(e.tag, dict(e.attrib)) for e in self.root.find("actuator")]
        self.sensors = [(e.tag, dict(e.attrib)) for e in self.root.find("sensor")]
        self.world_geoms = [dict(e.attrib) for e in wb.findall("geom")]

    def _walk(self, el, parent):
        name = el.get("name")
        body = {"name": name, "parent": parent, "pos": _v(el.get("pos")), "joints": [], "geoms": [],
                "sites": [], "has_orientation": any(k in el.attrib for k in ("quat", "euler", "axisangle", "xyaxes", "zaxis"))}
        if name in self.bodies:
            self.errors.append(f"duplicate body {name}")
        self.bodies[name] = body
        self.order.append(name)
        for ch in el:
            if ch.tag == "freejoint":
                self.joints[ch.get("name")] = {"name": ch.get("name"), "type": "free", "body": name, "axis": None}
                body["joints"].append(ch.get("name"))
            elif ch.tag == "joint":
                at = {**self.default_joint, **ch.attrib}
                self.joints[ch.get("name")] = {"name": ch.get("name"), "type": at.get("type", "hinge"),
                                               "axis": _v(at.get("axis", "0 0 1")), "body": name,
                                               "armature": float(at.get("armature", 0)), "attrib": at}
                body["joints"].append(ch.get("name"))
            elif ch.tag == "geom":
                body["geoms"].append(dict(ch.attrib))
            elif ch.tag == "site":
                body["sites"].append(dict(ch.attrib))
            elif ch.tag == "body":
                self._walk(ch, name)

    # ---- kinematics -------------------------------------------------------------------------------
    def transforms(self, q: dict[str, float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in self.order:                              # parents always precede children
            b = self.bodies[name]
            if b["parent"] is None:
                R, p = np.eye(3), b["pos"].copy()
            else:
                Rp, pp = out[b["parent"]]
                R, p = Rp, pp + Rp @ b["pos"]
            for jn in b["joints"]:
                j = self.joints[jn]
                if j["type"] == "hinge":
                    R = R @ rot_axis(j["axis"], q.get(jn, 0.0))
            out[name] = (R, p)
        return out

    def hinge_names(self) -> list[str]:
        return [n for n in self.joints if self.joints[n]["type"] == "hinge"]


def connect_residuals(model: Model, q: dict[str, float]) -> dict[str, float]:
    """Gap of each connect constraint, using MuJoCo's semantics (body-2 anchor frozen at qpos0)."""
    T0 = model.transforms({})
    T = model.transforms(q)
    out = {}
    for c in model.connects:
        a1 = _v(c["anchor"])
        R10, p10 = T0[c["body1"]]
        R20, p20 = T0[c["body2"]]
        a2 = R20.T @ (p10 + R10 @ a1 - p20)
        R1, p1 = T[c["body1"]]
        R2, p2 = T[c["body2"]]
        out[c["name"]] = float(np.linalg.norm((p1 + R1 @ a1) - (p2 + R2 @ a2)))
    return out


# ----------------------------------------------------------------------------------------------------
# bar lengths recovered from the XML
# ----------------------------------------------------------------------------------------------------

def xml_bar_lengths(model: Model, side: str) -> dict[str, float]:
    """Length of every bar of one leg, measured from body offsets/anchors in the XML (metres)."""
    b = model.bodies
    cons = {c["name"]: c for c in model.connects}
    L: dict[str, float] = {}
    L["crank"] = float(np.linalg.norm(b[f"j_{side}"]["pos"][[0, 2]] - 0.0))      # shaft origin -> crank pin X
    anchor = lambda node, body: float(np.linalg.norm(_v(cons[f"loop_{side}_{node}"]["anchor"])))
    L["j"] = anchor("W", f"j_{side}")
    L["e"] = anchor("V", f"e_{side}")
    L["c"] = anchor("U", f"c_{side}")
    L["f"] = anchor("T", f"f_{side}")
    L["i"] = anchor("S", f"i_{side}")
    child = lambda c: float(np.linalg.norm(b[f"{c}_{side}"]["pos"][[0, 2]]))
    L["b"] = child("e")        # origin of e (pivot W) relative to b (pivot Y)
    L["d"] = child("f")        # f pivots on d at V
    L["k"] = child("i")        # i pivots on k at U
    L["g"] = child("h")        # h pivots on g at T
    foot = [g for g in b[f"h_{side}"]["geoms"] if g.get("name", "").startswith("foot")][0]
    if foot["type"] == "capsule":
        f = _v(foot["fromto"], 6)
        L["h"] = float(np.linalg.norm(np.array([f[0], f[2]])))
    else:
        L["h"] = float(np.linalg.norm(_v(foot["pos"])[[0, 2]]))
    return L


# ----------------------------------------------------------------------------------------------------
# the full verification
# ----------------------------------------------------------------------------------------------------

def verify_model(xml_text: str, sweep: dict, hn, report: Optional[dict] = None,
                 closure_tol_m: float = 1e-7, length_tol_m: float = 1e-7) -> list[Check]:
    checks: list[Check] = []
    add = lambda *a, **k: checks.append(Check(*a, **k))
    model = Model(xml_text)

    # attribute whitelist
    bad = []
    for el in model.root.iter():
        allowed = ALLOWED.get(el.tag)
        if allowed is None:
            bad.append(f"unknown element <{el.tag}>")
        else:
            bad += [f"<{el.tag}> attribute '{a}'" for a in el.attrib if a not in allowed]
    add("only_known_elements_and_attributes", not bad, len(bad), 0, "; ".join(bad[:5]))

    # unique names
    names: dict[str, list[str]] = {}
    for el in model.root.iter():
        if el.get("name") and el.tag not in ("light", "camera", "texture", "material"):
            names.setdefault(el.tag if el.tag in ("body", "joint", "geom", "site") else "other", []).append(el.get("name"))
    dups = [n for lst in names.values() for n in set(lst) if lst.count(n) > 1]
    add("names_unique_within_type", not dups and not model.errors, len(dups), 0, ", ".join(dups[:5]))

    # references
    refs_bad = []
    for c in model.connects:
        for key in ("body1", "body2"):
            if c[key] not in model.bodies:
                refs_bad.append(f"{c['name']}.{key}={c[key]}")
    for tag, at in model.actuators:
        if at["joint"] not in model.joints:
            refs_bad.append(f"actuator {at['name']} joint {at['joint']}")
    site_names = {s["name"] for b in model.bodies.values() for s in b["sites"]}
    for tag, at in model.sensors:
        if "joint" in at and at["joint"] not in model.joints:
            refs_bad.append(f"sensor {at['name']} joint")
        if "site" in at and at["site"] not in site_names:
            refs_bad.append(f"sensor {at['name']} site")
        if "objname" in at and at["objname"] not in model.bodies:
            refs_bad.append(f"sensor {at['name']} body")
        if "actuator" in at and at["actuator"] not in {a["name"] for _, a in model.actuators}:
            refs_bad.append(f"sensor {at['name']} actuator")
    add("references_resolve", not refs_bad, len(refs_bad), 0, "; ".join(refs_bad[:5]))
    if refs_bad or dups or model.errors:
        add("verification_stopped_early", False, None, None, "fix names/references first; later checks need a consistent tree")
        return checks

    add("body_frames_are_world_aligned", not any(b["has_orientation"] for b in model.bodies.values()),
        sum(b["has_orientation"] for b in model.bodies.values()), 0, "required so qpos0 is the reference pose")

    n_hinge = len(model.hinge_names())
    add("one_free_joint_on_the_chassis", [j["body"] for j in model.joints.values() if j["type"] == "free"] == ["chassis"],
        None, None)
    if report:
        exp = report["counts"]
        add("joint_and_constraint_counts", n_hinge == exp["hinge_joints"] and len(model.connects) * 3 == exp["expected_equality_rows"],
            [n_hinge, len(model.connects) * 3], [exp["hinge_joints"], exp["expected_equality_rows"]],
            f"nq={7 + n_hinge} nv={6 + n_hinge}; net DOF = {6 + n_hinge - 2 * len(model.connects)} (expected 7)")
    add("net_degrees_of_freedom_is_seven", 6 + n_hinge - 2 * len(model.connects) == 7, 6 + n_hinge - 2 * len(model.connects), 7,
        "6 base + 1 crank; each leg's 5 planar loop closures remove 2 each")

    # every body has mass
    massless = [n for n, b in model.bodies.items() if not any(float(g.get("mass", 0)) > 0 for g in b["geoms"])]
    add("every_body_has_mass", not massless, len(massless), 0, ", ".join(massless[:5]))

    # bar lengths from the XML vs the validated kinematics
    nominal = {k: v * 1e-3 for k, v in bar_lengths(hn).items() if k != "fixed frame"}
    worst, worst_bar = 0.0, "-"
    for side in ("L", "R"):
        got = xml_bar_lengths(model, side)
        for bar, want in nominal.items():
            err = abs(got[bar] - want)
            if err > worst:
                worst, worst_bar = err, f"{bar}_{side}"
    add("bar_lengths_match_kinematics", worst <= length_tol_m, worst, length_tol_m, f"worst {worst_bar}, metres")

    # loop closure over the sweep
    names_sw = sweep["joint_names"]
    missing = [n for n in names_sw if n not in model.joints]
    add("sweep_joints_exist", not missing, len(missing), 0, ", ".join(missing[:5]))
    worst_gap, worst_at = 0.0, None
    if not missing:
        for off, row in zip(sweep["crank_offset_deg"], sweep["hinge_angles_rad"]):
            q = dict(zip(names_sw, row))
            for cname, gap in connect_residuals(model, q).items():
                if gap > worst_gap:
                    worst_gap, worst_at = gap, (cname, off)
    add("loop_closure_over_full_crank_cycle", worst_gap <= closure_tol_m, worst_gap, closure_tol_m,
        f"{len(sweep['crank_offset_deg'])} crank angles x {len(model.connects)} connects; worst {worst_at}")
    # sensitivity: the check must be able to fail
    broken = dict(zip(names_sw, sweep["hinge_angles_rad"][len(sweep["hinge_angles_rad"]) // 3]))
    first = names_sw[3] if len(names_sw) > 3 else names_sw[-1]
    broken[first] += 0.01
    sens = max(connect_residuals(model, broken).values()) if not missing else 0.0
    add("closure_check_detects_a_0.01_rad_error", sens > 100 * closure_tol_m, sens, 100 * closure_tol_m,
        "perturbing one hinge by 0.01 rad must break a loop")
    rows = np.array(sweep["hinge_angles_rad"])
    # Bodies hinged on the crankshaft turn a full -2 pi relative to it, so compare modulo 2 pi.
    wrapped = np.abs((rows[-1] - rows[0] + math.pi) % (2 * math.pi) - math.pi)
    add("sweep_is_periodic", abs(rows[-1, 0] - rows[0, 0] - 2 * math.pi) < 1e-9 and wrapped.max() < 1e-9,
        float(wrapped.max()), 1e-9, "every hinge returns to its start angle (mod 2 pi) after one crank turn; crank advances exactly 2 pi")
    total_turns = np.round((rows[-1] - rows[0]) / (2 * math.pi)).astype(int)
    shaft_children = [i for i, n in enumerate(sweep["joint_names"]) if n.split("_")[0] in ("j", "k") and i > 0]
    others = [i for i in range(1, len(total_turns)) if i not in shaft_children]
    add("only_links_on_the_crank_wind_relative_to_their_parent", bool(np.all(total_turns[shaft_children] == -1) and np.all(total_turns[others] == 0)),
        total_turns.tolist(), "crank +1, j/k -1, rest 0", "no other link makes a full turn")

    # direction of travel and floor clearance
    foot = {}
    for side in ("L", "R"):
        g = [g for g in model.bodies[f"h_{side}"]["geoms"] if g["name"].startswith("foot")][0]
        r = float(g["size"].split()[0])
        c = (_v(g["fromto"], 6)[:3] + _v(g["fromto"], 6)[3:]) / 2 if g["type"] == "capsule" else _v(g["pos"])
        foot[side] = (c, r)
    T0 = model.transforms({})
    zs = {s: (T0[f"h_{s}"][0] @ foot[s][0] + T0[f"h_{s}"][1])[2] - foot[s][1] for s in foot}   # transforms already include the chassis height
    lowest = min(zs.values())
    add("feet_start_just_above_the_floor", 0.0 <= lowest <= 0.002, lowest, [0.0, 0.002], "foot bottoms (m): " + ", ".join(f"{k}={float(v):.5f}" for k, v in zs.items()))

    foot_x, foot_z = [], []
    for off, row in zip(sweep["crank_offset_deg"], sweep["hinge_angles_rad"]):
        T = model.transforms(dict(zip(names_sw, row)))
        R, p = T["h_L"]
        pt = R @ foot["L"][0] + p
        foot_x.append(pt[0]); foot_z.append(pt[2])
    i = int(np.argmin(foot_z))
    n = len(foot_x)
    dx = foot_x[(i + 1) % n] - foot_x[(i - 1) % n]
    add("foot_moves_backward_relative_to_body_during_stance", dx < 0, dx, "< 0",
        "so a positive crank rotation walks the robot toward +X")
    shaft_ok = all(abs(row[0] - math.radians(off)) < 1e-9 for off, row in zip(sweep["crank_offset_deg"], sweep["hinge_angles_rad"]))
    add("crank_joint_angle_equals_crank_angle", shaft_ok, None, None, "positive hinge angle = positive crank angle")
    return checks


def summarise(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        v = "" if c.value is None else (f"{c.value:.3g}" if isinstance(c.value, float) else str(c.value))
        lines.append(f"[{'PASS' if c.passed else 'FAIL'}] {c.name:<52} {v:<12} {c.detail}")
    lines.append("RESULT: " + ("ALL CHECKS PASSED" if all(c.passed for c in checks) else "SOME CHECKS FAILED"))
    return "\n".join(lines)
