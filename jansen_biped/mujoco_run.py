"""Compile, check and run the generated MuJoCo model.  Needs ``pip install mujoco``.

    python -m jansen_biped.mujoco_run check --model output_mujoco/jansen_biped.xml
    python -m jansen_biped.mujoco_run run   --model output_mujoco/jansen_biped.xml --seconds 10 [--rpm 30] [--viewer]

``check`` compiles the XML with MuJoCo, compares model sizes with the generator's report,
and puts the mechanism through the saved crank sweep, reporting the largest loop-closure
gap that MuJoCo itself measures. ``run`` drives the crank and records what the robot does.

This file was written without MuJoCo available to execute it; ``check`` is the first thing to run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def _import_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit("The 'mujoco' Python package is required: pip install mujoco") from exc
    return mujoco


def _load(model_path: Path):
    mujoco = _import_mujoco()
    m = mujoco.MjModel.from_xml_path(str(model_path))
    return mujoco, m, mujoco.MjData(m)


def _sibling_json(model_path: Path, name: str):
    p = Path(model_path).with_name(name)
    return json.loads(p.read_text()) if p.exists() else None


def _id(mujoco, m, kind, name):
    i = mujoco.mj_name2id(m, kind, name)
    return None if i < 0 else i


def connect_gaps(mujoco, m, d) -> dict[str, float]:
    """Distance between the two anchor points of every connect constraint (m)."""
    gaps = {}
    for i in range(m.neq):
        if m.eq_type[i] != mujoco.mjtEq.mjEQ_CONNECT:
            continue
        b1, b2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
        p1 = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ m.eq_data[i, 0:3]
        p2 = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ m.eq_data[i, 3:6]
        gaps[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i)] = float(np.linalg.norm(p1 - p2))
    return gaps


def efc_equality_residual(d) -> float | None:
    """Largest |efc_pos| over the equality rows, if MuJoCo populated them; else None."""
    ne = int(getattr(d, "ne", 0))
    return float(np.max(np.abs(d.efc_pos[:ne]))) if ne > 0 else None


def set_pose(mujoco, m, d, names, angles):
    d.qpos[:] = m.qpos0
    for n, a in zip(names, angles):
        j = _id(mujoco, m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j is None:
            raise SystemExit(f"joint {n!r} from the crank sweep is not in the compiled model")
        d.qpos[m.jnt_qposadr[j]] = a
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)


# ----------------------------------------------------------------------------------------------
def cmd_check(args) -> int:
    mujoco, m, d = _load(Path(args.model))
    report = _sibling_json(Path(args.model), "jansen_biped_mujoco_report.json")
    sweep = _sibling_json(Path(args.model), "crank_sweep.json")
    ok = True
    print(f"MuJoCo {mujoco.__version__}: compiled {args.model}")
    print(f"  nbody={m.nbody} njnt={m.njnt} nq={m.nq} nv={m.nv} neq={m.neq} nu={m.nu} nsensor={m.nsensor} ngeom={m.ngeom}")
    total_mass = float(np.sum(m.body_mass))
    print(f"  total mass {total_mass:.4f} kg, timestep {m.opt.timestep:g} s")
    if report:
        exp = report["counts"]
        sizes = {"nq": (m.nq, exp["expected_nq"]), "nv": (m.nv, exp["expected_nv"]), "neq": (m.neq, exp["connect_constraints"])}
        for k, (got, want) in sizes.items():
            flag = "ok" if got == want else "MISMATCH"
            ok &= got == want
            print(f"  {k}: {got} (expected {want}) {flag}")
        dm = abs(total_mass - report["mass_kg"]["total"])
        print(f"  mass vs report: difference {dm:.2e} kg {'ok' if dm < 1e-6 else 'MISMATCH'}")
        ok &= dm < 1e-6

    # reference pose
    mujoco.mj_forward(m, d)
    gaps0 = connect_gaps(mujoco, m, d)
    print(f"  reference pose: largest connect gap {max(gaps0.values()):.2e} m")

    if sweep:
        worst, worst_at, efc_worst = 0.0, None, 0.0
        for off, row in zip(sweep["crank_offset_deg"], sweep["hinge_angles_rad"]):
            set_pose(mujoco, m, d, sweep["joint_names"], row)
            g = connect_gaps(mujoco, m, d)
            k = max(g, key=g.get)
            if g[k] > worst:
                worst, worst_at = g[k], (k, off)
            e = efc_equality_residual(d)
            if e is not None:
                efc_worst = max(efc_worst, e)
        tol = args.tol
        print(f"  crank sweep ({len(sweep['crank_offset_deg'])} poses): worst anchor gap {worst:.2e} m at {worst_at}")
        print(f"                     worst |efc_pos| over equality rows {efc_worst:.2e}" if efc_worst else
              "                     efc_pos not populated by mj_forward in this version (anchor gaps used)")
        crit = efc_worst if efc_worst else worst
        good = crit <= tol
        ok &= good
        print(f"  loop closure {'PASS' if good else 'FAIL'} (tolerance {tol:g} m)")
    else:
        print("  crank_sweep.json not found next to the model; sweep check skipped")

    # short dynamic smoke test: drive for 1 s and make sure nothing blows up
    mujoco.mj_resetData(m, d)
    omega = args.rpm * 2 * math.pi / 60
    steps = int(1.0 / m.opt.timestep)
    nan = False
    max_gap = 0.0
    is_vel = m.nu > 0 and abs(m.actuator_biasprm[0, 2]) > 0
    for _ in range(steps):
        d.ctrl[0] = omega if is_vel else 0.0
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            nan = True
            break
    if not nan:
        max_gap = max(connect_gaps(mujoco, m, d).values())
    print(f"  1 s dynamic smoke test: {'NaN/Inf in qpos' if nan else 'finite'}; loop gap after 1 s {max_gap:.2e} m")
    ok &= (not nan) and max_gap < 1e-3
    print("RESULT:", "ALL CHECKS PASSED" if ok else "SOMETHING FAILED (see above)")
    return 0 if ok else 1


# ----------------------------------------------------------------------------------------------
def quat_to_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def cmd_run(args) -> int:
    mujoco, m, d = _load(Path(args.model))
    report = _sibling_json(Path(args.model), "jansen_biped_mujoco_report.json")
    sid = lambda n: _id(mujoco, m, mujoco.mjtObj.mjOBJ_SENSOR, n)
    adr = lambda n: int(m.sensor_adr[sid(n)]) if sid(n) is not None else None
    a_tq, a_tl, a_tr = adr("crank_torque"), adr("touch_L"), adr("touch_R")
    crank = _id(mujoco, m, mujoco.mjtObj.mjOBJ_JOINT, "crank")
    qadr, vadr = int(m.jnt_qposadr[crank]), int(m.jnt_dofadr[crank])
    housing_geom = _id(mujoco, m, mujoco.mjtObj.mjOBJ_GEOM, "housing_collision")
    floor_geom = _id(mujoco, m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    omega = args.rpm * 2 * math.pi / 60
    is_vel = m.nu > 0 and abs(m.actuator_biasprm[0, 2]) > 0
    kp = 5.0
    manual = bool(args.manual_end)

    mujoco.mj_forward(m, d)
    z0 = float(d.qpos[2])
    x0 = float(d.qpos[0])
    rows = []
    fell_at = None
    ended_by = None
    peak_tq = sq_tq = power_sum = 0.0
    n = 0
    belly = False
    contact_L = contact_R = both = neither = 0
    max_gap = 0.0
    log_every = max(1, int(round(0.01 / m.opt.timestep)))
    step_limit = None if manual else int(args.seconds / m.opt.timestep)

    def step_once():
        if is_vel:
            d.ctrl[0] = omega
        else:
            tq = kp * (omega - d.qvel[vadr])
            d.ctrl[0] = float(np.clip(tq, m.actuator_ctrlrange[0, 0], m.actuator_ctrlrange[0, 1]))
        mujoco.mj_step(m, d)

    viewer_cm = None
    if args.viewer:
        import mujoco.viewer
        viewer_cm = mujoco.viewer.launch_passive(m, d)
    if manual:
        how = "close the viewer window" if args.viewer else "press Ctrl-C"
        print(f"Manual-end mode: the simulation never stops by itself (no time limit, no stop on falling). To end it, {how}.")
    k = 0
    try:
        while step_limit is None or k < step_limit:
            step_once()
            k += 1
            if viewer_cm is not None:
                viewer_cm.sync()
                time.sleep(m.opt.timestep)
                if not viewer_cm.is_running():
                    ended_by = "viewer closed by user"
                    break
            if not np.all(np.isfinite(d.qpos)):
                ended_by = "non-finite state (model became numerically unstable)"
                print("Simulation produced NaN/Inf; it cannot continue.")
                break
            tq = float(d.sensordata[a_tq]) if a_tq is not None else float(d.actuator_force[0])
            peak_tq = max(peak_tq, abs(tq))
            sq_tq += tq * tq
            power_sum += tq * float(d.qvel[vadr])
            n += 1
            for i in range(d.ncon):
                c = d.contact[i]
                if housing_geom is not None and housing_geom in (c.geom1, c.geom2) and floor_geom in (c.geom1, c.geom2):
                    belly = True
            if k % log_every == 0:
                roll, pitch, yaw = quat_to_rpy(d.qpos[3:7])
                tl = float(d.sensordata[a_tl]) if a_tl is not None else 0.0
                tr = float(d.sensordata[a_tr]) if a_tr is not None else 0.0
                dl, dr = tl > 0.05, tr > 0.05
                contact_L += dl
                contact_R += dr
                both += dl and dr
                neither += (not dl) and (not dr)
                gap = max(connect_gaps(mujoco, m, d).values())
                max_gap = max(max_gap, gap)
                rows.append([d.time, *d.qpos[0:3], roll, pitch, yaw, d.qpos[qadr], d.qvel[vadr], tq, tl, tr, gap])
                fallen = abs(roll) > math.radians(60) or abs(pitch) > math.radians(60) or d.qpos[2] < 0.4 * z0
                if fallen and fell_at is None:
                    fell_at = float(d.time)
                    if manual:
                        print(f"Robot fell at {fell_at:.2f} s; continuing because of --manual-end.")
                if fallen and not manual:
                    ended_by = "robot fell"
                    break
    except KeyboardInterrupt:
        ended_by = "Ctrl-C"
        print("\nInterrupted: writing the log and summary.")
    finally:
        if viewer_cm is not None:
            viewer_cm.close()
    if ended_by is None:
        ended_by = "time limit"

    t = float(d.time)
    nrec = max(len(rows), 1)
    arr = np.array(rows) if rows else np.zeros((1, 13))
    dist = float(d.qpos[0] - x0)
    G = report["gearbox"]["ratio"] if report else None
    eta = report["parameters"]["gearbox_efficiency"] if report else 0.8
    summary = {
        "ended_by": ended_by, "manual_end": manual,
        "simulated_s": t, "fell_at_s": fell_at,
        "forward_distance_m": dist, "mean_forward_speed_m_s": dist / t if t > 0 else 0.0,
        "lateral_drift_m": float(d.qpos[1]), "chassis_height_m": [float(arr[:, 3].min()), float(arr[:, 3].max())],
        "max_abs_roll_deg": float(np.degrees(np.abs(arr[:, 4]).max())), "max_abs_pitch_deg": float(np.degrees(np.abs(arr[:, 5]).max())),
        "peak_crank_torque_Nm": peak_tq, "rms_crank_torque_Nm": math.sqrt(sq_tq / n) if n else 0.0,
        "mean_crank_power_W": power_sum / n if n else 0.0,
        "peak_motor_side_torque_Nm": peak_tq / (G * eta) if G else None,
        "torque_limit_hit": bool(peak_tq >= m.actuator_forcerange[0, 1] - 1e-6) if m.nu and m.actuator_forcelimited[0] else False,
        "foot_contact_fraction": {"left": contact_L / nrec, "right": contact_R / nrec, "both": both / nrec, "neither": neither / nrec},
        "housing_touched_floor": belly,
        "max_loop_closure_gap_m": max_gap,
    }
    out = Path(args.out) if args.out else Path(args.model).with_name("run")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run_log.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "x", "y", "z", "roll", "pitch", "yaw", "crank_angle", "crank_vel", "crank_torque", "touch_L", "touch_R", "loop_gap"])
        w.writerows(rows)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    fell = f"fell at {fell_at:.2f} s; " if fell_at is not None else "did not fall; "
    print(f"\nEnded by: {ended_by}. {fell}simulated {t:.1f} s, moved {dist * 1000:.0f} mm, "
          f"peak crank torque {peak_tq:.2f} N m. Log: {out / 'run_log.csv'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="compile, compare with the report, check loop closure over the crank sweep")
    c.add_argument("--model", default="output_mujoco/jansen_biped.xml")
    c.add_argument("--rpm", type=float, default=30.0)
    c.add_argument("--tol", type=float, default=1e-6, help="loop-closure tolerance in metres")
    r = sub.add_parser("run", help="drive the crank and record the robot's motion")
    r.add_argument("--model", default="output_mujoco/jansen_biped.xml")
    r.add_argument("--seconds", type=float, default=10.0)
    r.add_argument("--rpm", type=float, default=30.0)
    r.add_argument("--out", default=None)
    r.add_argument("--viewer", action="store_true")
    r.add_argument("--manual-end", action="store_true",
                   help="never end by itself: ignore --seconds and keep going after the robot falls; "
                        "end by closing the viewer window or pressing Ctrl-C (log and summary are still written)")
    a = ap.parse_args(argv)
    return cmd_check(a) if a.cmd == "check" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
