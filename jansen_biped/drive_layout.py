"""Layout solver for the compact, between-the-legs drive module.

This module is deliberately free of CadQuery so every dimension, clearance and
mesh phase can be computed and unit-tested without a CAD kernel.
``compact_drive.py`` turns a ``DriveLayout`` into solids.

Frame (identical to the assembly builders)
-------------------------------------------
* x: in the linkage plane (fore/aft), y: up, z: along the crankshaft.
* The output/crank axis O (linkage node Z) is the origin. Left leg at +z,
  right leg at -z; the housing is centred on z = 0.
* The motor sits against the housing's back wall (-z side), shaft pointing +z.

Axial order inside the housing, back (-z) to front (+z)
--------------------------------------------------------
back wall | motor body (inside a split sleeve) | clearance | plane P1 (input
pinion and the large compound gear) | their hubs | clearance | front lid.
Plane P2 (compound pinion and output gear) sits *behind* P1, overlapping the
motor body's z range, which saves one gear width plus hub of housing length.
The output gear's hub is flipped to point away from P1 so only the bare output
shaft crosses P1's z range.

The housing is one rectangular box (tub + removable lid) that spans the crank
axis O *and* the fixed linkage pivot Y, so it replaces both fixed-frame bars.
Y is carried by a hollow post running from the back wall to the lid. The legs sit
outside the end walls, mirrored about z = 0.

Everything is in millimetres.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

# Tooth counts (input pinion, large compound gear, compound pinion, output gear).
# geartrain.py imports this table so there is one source of truth.
RATIOS: dict[str, tuple[int, int, int, int]] = {
    "7": (21, 63, 18, 42),    # (63/21) * (42/18) = 7
    "15": (16, 64, 16, 60),   # (64/16) * (60/16) = 15
}

#: Preferred module series used when the stage-2 module is chosen automatically.
STANDARD_MODULES = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


#: Lightening windows in the +y / -y housing walls (each at most this wide so they print without supports).
WINDOW_W, WINDOW_RIB = 24.0, 8.0


class LayoutError(ValueError):
    """No feasible compact layout exists for the requested parameters."""


# --------------------------------------------------------------------------
# Gear helpers (match gear_solid() in geartrain.py)
# --------------------------------------------------------------------------

def pitch_radius(teeth: int, module: float) -> float:
    return module * teeth / 2.0


def tip_radius(teeth: int, module: float) -> float:
    """Outer radius used by gear_solid(): module * (teeth/2 + 1)."""
    return module * (teeth / 2.0 + 1.0)


def root_radius(teeth: int, module: float) -> float:
    return max(module * (teeth / 2.0 - 1.20), module * 1.5)


def mesh_rotation_deg(teeth_driver: int, teeth_driven: int,
                      driver_rotation_deg: float, direction_driver_to_driven_deg: float) -> float:
    """Rotation to apply to the driven gear so its teeth interleave with the driver's.

    Both gears are generated with tooth 0 on their local +x axis. Along the line
    of centres a driver tooth must face a driven gap. With f = fractional tooth
    phase seen along that line, the invariant of a rolling mesh is
    f_driver + f_driven = 0.5 (mod 1).
    """
    pitch1 = 360.0 / teeth_driver
    pitch2 = 360.0 / teeth_driven
    a = direction_driver_to_driven_deg
    f1 = (((a - driver_rotation_deg) % pitch1) / pitch1)
    f2_target = (0.5 - f1) % 1.0
    # Direction seen from the driven gear points back at the driver.
    theta2 = (a + 180.0) - f2_target * pitch2
    return theta2 % pitch2


def tooth_phase(teeth: int, rotation_deg: float, direction_deg: float) -> float:
    """Fractional tooth phase of a gear along a given direction (0 = tooth on the line)."""
    pitch = 360.0 / teeth
    return ((direction_deg - rotation_deg) % pitch) / pitch


# --------------------------------------------------------------------------
# Specification and result containers
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DriveSpec:
    ratio: str = "7"
    motor_diameter: float = 42.0
    motor_length: float = 45.0
    motor_shaft_d: float = 5.0
    output_shaft_d: float = 8.0
    module1: float = 1.0
    module2: Optional[float] = None      # None -> smallest standard module that fits
    gear_thickness: float = 5.0
    hub_height: float = 3.0
    hub_diameter: float = 16.0
    link_thickness: float = 3.0
    bearing_od: float = 22.0
    bearing_width: float = 7.0
    motor_clearance: float = 0.8         # diametral, motor can in sleeve
    sleeve_wall: float = 3.0
    end_wall: float = 5.0                # back wall / front lid thickness
    perimeter_wall: float = 3.0
    boss_height: float = 3.0             # bearing boss standing proud of each end wall
    axial_gap: float = 1.5               # motor face -> P1, P1 hubs -> lid
    radial_margin: float = 3.0           # circles -> inside of perimeter wall
    layer_clearance: float = 1.5         # boss face -> first link layer (washer)
    gear_gap: float = 2.0                # min tip-to-obstacle distance
    shaft_gap: float = 1.5               # min large-gear-tip to output-shaft surface
    post_radius: float = 3.5             # lid-screw posts in the cavity corners
    y_post_radius: float = 5.0           # hollow post carrying the fixed pivot Y
    n_layers: int = 4                    # link layers per leg (see layer_assignment.py)
    layer_pitch: float = 3.5             # link thickness + running clearance between layers
    prefer_forward_motor: bool = True    # among near-minimal footprints, motor axis closest to Y


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    value: float
    limit: float
    comparison: str      # ">=" or "<="
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": bool(self.passed), "value": round(float(self.value), 4),
                "limit": round(float(self.limit), 4), "comparison": self.comparison, "detail": self.detail}


def _check(name: str, value: float, limit: float, cmp: str = ">=", detail: str = "") -> Check:
    ok = value >= limit if cmp == ">=" else value <= limit
    return Check(name, bool(ok), float(value), float(limit), cmp, detail)


@dataclass
class DriveLayout:
    spec: DriveSpec
    teeth: tuple[int, int, int, int]
    module1: float
    module2: float
    # shaft axes, assembly frame (x, y)
    O: tuple[float, float]
    C: tuple[float, float]
    M: tuple[float, float]
    c1: float
    c2: float
    fold_theta_c_deg: float          # direction O -> C
    fold_theta_m_deg: float          # direction C -> M
    radii: dict[str, float]
    # axial levels (z, housing centred on 0)
    z: dict[str, float]
    half_length: float               # H
    z0_leg: float                    # inner face of the first link layer, |z|
    # xy rectangles: (x0, x1, y0, y1)
    inner_rect: tuple[float, float, float, float]
    outer_rect: tuple[float, float, float, float]
    sleeve_outer_r: float
    boss_r: float
    gear_rotation_deg: dict[str, float]
    pivot_Y: tuple[float, float]
    y_post_r: float
    stack_outer_z: float             # outer face of the last link layer, |z|
    shaft_tip_z: float               # output shaft tip, |z| (retainer collar + 1 mm)
    y_pin_length_mm: float
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def side_gap_equivalent(self) -> float:
        """Distance between the two first-layer faces (2 * z0), comparable to --side-gap."""
        return 2.0 * self.z0_leg

    def to_dict(self) -> dict[str, Any]:
        r = lambda v: [round(float(x), 4) for x in v]
        return {
            "units": "mm",
            "ratio": f"{self.spec.ratio}:1",
            "teeth": list(self.teeth),
            "module_stage1": self.module1, "module_stage2": self.module2,
            "shaft_axes_xy": {"output_crank_axis_O": r(self.O), "compound_C": r(self.C), "motor_input_M": r(self.M)},
            "centre_distance_stage1": round(self.c1, 4), "centre_distance_stage2": round(self.c2, 4),
            "fold_angles_deg": {"O_to_C": self.fold_theta_c_deg, "C_to_M": self.fold_theta_m_deg},
            "radii_mm": {k: round(v, 4) for k, v in self.radii.items()},
            "axial_levels_z_mm": {k: round(v, 4) for k, v in self.z.items()},
            "housing_half_length_mm": round(self.half_length, 4),
            "housing_length_z_mm": round(2 * self.half_length, 4),
            "leg_first_layer_inner_face_z_mm": round(self.z0_leg, 4),
            "side_gap_equivalent_mm": round(self.side_gap_equivalent, 4),
            "housing_inner_rect_xyxy_mm": r(self.inner_rect),
            "housing_outer_rect_xyxy_mm": r(self.outer_rect),
            "housing_outer_size_x_y_z_mm": [round(self.outer_rect[1] - self.outer_rect[0], 3),
                                            round(self.outer_rect[3] - self.outer_rect[2], 3),
                                            round(2 * self.half_length, 3)],
            "fixed_pivot_Y_xy_mm": r(self.pivot_Y),
            "outer_stack_face_z_mm": round(self.stack_outer_z, 3),
            "output_shaft_tip_z_mm": round(self.shaft_tip_z, 3),
            "overall_width_over_legs_mm": round(2 * self.stack_outer_z, 3),
            "motor_axis_x_minus_pivot_Y_x_mm": round(self.M[0] - self.pivot_Y[0], 3),
            "pivot_Y_rod_length_mm": round(self.y_pin_length_mm, 2),
            "gear_rotation_about_own_axis_deg": {k: round(v, 4) for k, v in self.gear_rotation_deg.items()},
            "checks": [c.to_dict() for c in self.checks],
            "all_checks_passed": self.passed,
        }


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------

def _bbox(circles):
    x0 = min(cx - r for cx, cy, r in circles)
    x1 = max(cx + r for cx, cy, r in circles)
    y0 = min(cy - r for cx, cy, r in circles)
    y1 = max(cy + r for cx, cy, r in circles)
    return x0, x1, y0, y1


def _fold_candidates(spec, c1, c2, r_sleeve, r_out_tip, r_boss, r_large_tip, r_pin1_tip, r_pin2_tip, Y):
    """Feasible folds of the O-C-M shaft triangle, best first.

    The footprint is the bounding rectangle of the O, C, M envelopes *and* the Y
    post, because the box must reach pivot Y. Hard constraints here are those
    between parts sharing a z range: the motor sleeve against the output gear
    (plane P2), and the full-length Y post against everything. Ordering: folds
    within 3 % of the smallest footprint come first, and among those the motor
    axis nearest Y in x (keeps the heaviest part over the foot-stance region
    instead of behind the crank axis); the rest follow by footprint area.
    The caller runs the full check list on candidates in this order.
    """
    ry = spec.y_post_radius
    r_O, r_C, r_M = max(r_out_tip, r_boss), max(r_large_tip, r_pin2_tip, r_boss), max(r_sleeve, r_pin1_tip)
    d_min = r_sleeve + r_out_tip + spec.gear_gap
    if math.hypot(Y[0], Y[1]) < r_O + ry + 1.0:
        return []
    cands = []
    for tc in range(0, 360):
        cx, cy = c2 * math.cos(math.radians(tc)), c2 * math.sin(math.radians(tc))
        if math.hypot(cx - Y[0], cy - Y[1]) < r_C + ry + 1.0:
            continue
        for tm in range(0, 360):
            mx, my = cx + c1 * math.cos(math.radians(tm)), cy + c1 * math.sin(math.radians(tm))
            if math.hypot(mx, my) < d_min or math.hypot(mx - Y[0], my - Y[1]) < r_M + ry + 1.0:
                continue
            x0 = min(-r_O, cx - r_C, mx - r_M, Y[0] - ry)
            x1 = max(r_O, cx + r_C, mx + r_M, Y[0] + ry)
            y0 = min(-r_O, cy - r_C, my - r_M, Y[1] - ry)
            y1 = max(r_O, cy + r_C, my + r_M, Y[1] + ry)
            cands.append(((x1 - x0) * (y1 - y0), tc, tm, (cx, cy), (mx, my)))
    if not cands:
        return []
    area_min = min(c[0] for c in cands)
    def key(c):
        near = c[0] <= area_min * 1.03
        return (0 if near else 1, abs(c[4][0] - Y[0]) if (near and spec.prefer_forward_motor) else 0.0, c[0], c[1], c[2])
    return sorted(cands, key=key)


def _build(spec: DriveSpec, module2: float, hn, max_tries: int = 400) -> DriveLayout:
    """Best layout for one stage-2 module: first candidate fold that passes every check.

    If none passes, the most-preferred candidate is returned with its failed
    checks recorded. Raises LayoutError only when no fold is geometrically possible.
    """
    z1, z2, z3, z4 = RATIOS[spec.ratio]
    m1, m2 = spec.module1, module2
    c1 = pitch_radius(z1, m1) + pitch_radius(z2, m1)
    c2 = pitch_radius(z3, m2) + pitch_radius(z4, m2)
    r_sleeve = spec.motor_diameter / 2.0 + spec.motor_clearance / 2.0 + spec.sleeve_wall
    r_boss = spec.bearing_od / 2.0 + 3.0
    Y = (float(hn[0]), -float(hn[11]))
    cands = _fold_candidates(spec, c1, c2, r_sleeve, tip_radius(z4, m2), r_boss, tip_radius(z2, m1),
                             tip_radius(z1, m1), tip_radius(z3, m2), Y)
    if not cands:
        raise LayoutError(f"no fold of the gear train fits: pivot Y is {math.hypot(*Y):.1f} mm from the crank axis "
                          f"and the output gear envelope plus the Y post and motor sleeve need more room")
    first = None
    for cand in cands[:max_tries]:
        lay = _assemble(spec, module2, hn, cand)
        if lay.passed:
            return lay
        first = first or lay
    return first


def _assemble(spec: DriveSpec, module2: float, hn, cand) -> DriveLayout:
    z1, z2, z3, z4 = RATIOS[spec.ratio]
    m1, m2 = spec.module1, module2
    c1 = pitch_radius(z1, m1) + pitch_radius(z2, m1)
    c2 = pitch_radius(z3, m2) + pitch_radius(z4, m2)
    r_sleeve = spec.motor_diameter / 2.0 + spec.motor_clearance / 2.0 + spec.sleeve_wall
    r_boss = spec.bearing_od / 2.0 + 3.0
    r_in_tip, r_large_tip = tip_radius(z1, m1), tip_radius(z2, m1)
    r_pin_tip, r_out_tip = tip_radius(z3, m2), tip_radius(z4, m2)

    Y = (float(hn[0]), -float(hn[11]))
    _, tc, tm, C, M = cand
    O = (0.0, 0.0)

    # ---- axial levels (housing centred on z = 0) --------------------------------
    Tg, hub, Tw = spec.gear_thickness, spec.hub_height, spec.end_wall
    length = Tw + spec.motor_length + spec.axial_gap + Tg + hub + spec.axial_gap + Tw
    H = length / 2.0
    zl: dict[str, float] = {}
    zl["back_wall_outer"] = -H
    zl["back_wall_inner"] = -H + Tw
    zl["motor_face"] = zl["back_wall_inner"] + spec.motor_length
    zl["P1_start"] = zl["motor_face"] + spec.axial_gap
    zl["P1_end"] = zl["P1_start"] + Tg
    zl["P1_hub_end"] = zl["P1_end"] + hub
    zl["lid_inner"] = zl["P1_hub_end"] + spec.axial_gap
    zl["lid_outer"] = zl["lid_inner"] + Tw
    zl["P2_start"] = zl["P1_start"] - hub - Tg           # compound pinion; its hub meets the large gear
    zl["P2_end"] = zl["P2_start"] + Tg
    zl["output_gear_hub_start"] = zl["P2_start"] - hub   # hub flipped away from P1
    assert abs(zl["lid_outer"] - H) < 1e-9
    z0_leg = H + spec.boss_height + spec.layer_clearance
    stack_outer = z0_leg + spec.n_layers * spec.layer_pitch
    shaft_tip = z0_leg + spec.link_thickness + 0.2 + 2.0 + 1.0   # crank, gap, 2 mm collar, 1 mm tip

    # ---- housing rectangles ------------------------------------------------------
    circles = [
        (O[0], O[1], max(r_out_tip, r_boss)),
        (C[0], C[1], max(r_large_tip, r_pin_tip, r_boss)),
        (M[0], M[1], max(r_sleeve, r_in_tip)),
        (Y[0], Y[1], spec.y_post_radius),
    ]
    bx0, bx1, by0, by1 = _bbox(circles)
    mg = spec.radial_margin
    inner = (bx0 - mg, bx1 + mg, by0 - mg, by1 + mg)
    wall = spec.perimeter_wall
    outer = (inner[0] - wall, inner[1] + wall, inner[2] - wall, inner[3] + wall)

    # ---- fixed pivot Y: threaded rod through the post, both leg stacks, washer + nut each end
    y_pin_len = 2.0 * stack_outer + 2.0 * 6.0

    # ---- mesh phases ---------------------------------------------------------------
    # Stage 1: input pinion (rot 0) at M drives the large gear at C.
    a_mc = math.degrees(math.atan2(C[1] - M[1], C[0] - M[0]))
    rot_large = mesh_rotation_deg(z1, z2, 0.0, a_mc)
    # Stage 2: compound pinion at C (rot 0 by convention) drives the output gear at O.
    a_co = math.degrees(math.atan2(O[1] - C[1], O[0] - C[0]))
    rot_out = mesh_rotation_deg(z3, z4, 0.0, a_co)

    radii = {"sleeve_outer": r_sleeve, "bearing_boss": r_boss, "input_pinion_tip": r_in_tip,
             "large_gear_tip": r_large_tip, "compound_pinion_tip": r_pin_tip, "output_gear_tip": r_out_tip,
             "large_gear_root": root_radius(z2, m1), "output_gear_root": root_radius(z4, m2)}

    lay = DriveLayout(spec=spec, teeth=(z1, z2, z3, z4), module1=m1, module2=m2, O=O, C=C, M=M, c1=c1, c2=c2,
                      fold_theta_c_deg=float(tc), fold_theta_m_deg=float(tm), radii=radii, z=zl,
                      half_length=H, z0_leg=z0_leg, inner_rect=inner, outer_rect=outer,
                      sleeve_outer_r=r_sleeve, boss_r=r_boss,
                      gear_rotation_deg={"input_pinion": 0.0, "compound_large_gear": rot_large,
                                         "compound_output_pinion": 0.0, "output_gear": rot_out},
                      pivot_Y=Y, y_post_r=spec.y_post_radius, stack_outer_z=stack_outer,
                      shaft_tip_z=shaft_tip, y_pin_length_mm=y_pin_len)
    lay.checks = _run_checks(lay)
    return lay


def _run_checks(lay: DriveLayout) -> list[Check]:
    s, R = lay.spec, lay.radii
    O, C, M = lay.O, lay.C, lay.M
    dMO = math.hypot(M[0] - O[0], M[1] - O[1])
    z1, z2, z3, z4 = lay.teeth
    out: list[Check] = []
    out.append(_check("large_gear_clear_of_output_shaft", lay.c2 - R["large_gear_tip"] - s.output_shaft_d / 2.0,
                      s.shaft_gap, detail="stage-1 large gear tip to output-shaft surface (they share z range P1)"))
    out.append(_check("motor_sleeve_clear_of_output_gear", dMO - R["sleeve_outer"] - R["output_gear_tip"],
                      s.gear_gap, detail="sleeve and output gear overlap in z"))
    out.append(_check("motor_sleeve_clear_of_compound_pinion", lay.c1 - R["sleeve_outer"] - R["compound_pinion_tip"],
                      s.gear_gap, detail="sleeve and compound pinion overlap in z"))
    out.append(_check("motor_sleeve_clear_of_bearing_boss_O", dMO - R["sleeve_outer"] - lay.boss_r, 1.0))
    out.append(_check("motor_sleeve_clear_of_bearing_boss_C", lay.c1 - R["sleeve_outer"] - lay.boss_r, 1.0))
    out.append(_check("bearing_bosses_O_C_separated", lay.c2 - 2.0 * lay.boss_r, 0.5))
    # Corner posts for the lid screws must clear every circle in the cavity.
    x0, x1, y0, y1 = lay.inner_rect
    pr = s.post_radius
    posts = [(x0 + pr, y0 + pr), (x1 - pr, y0 + pr), (x0 + pr, y1 - pr), (x1 - pr, y1 - pr)]
    obstacles = [(O, max(R["output_gear_tip"], lay.boss_r)), (C, max(R["large_gear_tip"], lay.boss_r)),
                 (M, max(R["sleeve_outer"], R["input_pinion_tip"]))]
    worst = min(math.hypot(p[0] - c[0], p[1] - c[1]) - r - pr for p in posts for c, r in obstacles)
    out.append(_check("corner_posts_clear_of_gears_and_sleeve", worst, 0.5))
    # The Y post runs the full housing length, so it must clear every envelope in xy.
    Yp, ry = lay.pivot_Y, lay.y_post_r
    dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    out.append(_check("y_post_clear_of_motor_sleeve", dist(M, Yp) - max(R["sleeve_outer"], R["input_pinion_tip"]) - ry, 1.0))
    out.append(_check("y_post_clear_of_compound_gears", dist(C, Yp) - max(R["large_gear_tip"], R["compound_pinion_tip"], lay.boss_r) - ry, 1.0))
    out.append(_check("y_post_clear_of_output_gear", dist(O, Yp) - max(R["output_gear_tip"], lay.boss_r) - ry, 1.0))
    out.append(_check("y_post_inside_housing", min(Yp[0] - ry - x0, x1 - Yp[0] - ry, Yp[1] - ry - y0, y1 - Yp[1] - ry), 1.0))
    out.append(_check("y_post_clear_of_corner_posts", min(dist(p, Yp) for p in posts) - pr - ry, 0.5))
    # Axial checks
    z = lay.z
    out.append(_check("output_gear_hub_clear_of_back_wall", z["output_gear_hub_start"] - z["back_wall_inner"], 1.0))
    out.append(_check("bearing_seat_depth_covers_bearing", s.end_wall + s.boss_height - s.bearing_width, 0.5,
                      detail="pocket depth minus bearing width = retaining lip + slack"))
    out.append(_check("shaft_tip_below_swept_link_layers", (lay.z0_leg + 2 * s.layer_pitch) - lay.shaft_tip_z, 0.5,
                      detail="shaft tip must end before layer 2, where links j and k sweep over the crank axis"))
    out.append(_check("legs_clear_of_housing", lay.z0_leg - lay.half_length - s.boss_height, 1.0,
                      detail="first link layer to bearing boss face"))
    out.append(_check("stage1_centre_distance_matches_pitch_circles",
                      abs(lay.c1 - (pitch_radius(z1, lay.module1) + pitch_radius(z2, lay.module1))), 1e-9, "<="))
    out.append(_check("stage2_centre_distance_matches_pitch_circles",
                      abs(lay.c2 - (pitch_radius(z3, lay.module2) + pitch_radius(z4, lay.module2))), 1e-9, "<="))
    # Tooth interleave along each line of centres.
    a_mc = math.degrees(math.atan2(C[1] - M[1], C[0] - M[0]))
    f1 = tooth_phase(z1, 0.0, a_mc)
    f2 = tooth_phase(z2, lay.gear_rotation_deg["compound_large_gear"], a_mc + 180.0)
    out.append(_check("stage1_teeth_interleave", abs(((f1 + f2) % 1.0) - 0.5), 1e-6, "<=", detail="f_driver + f_driven = 0.5 mod 1"))
    a_co = math.degrees(math.atan2(O[1] - C[1], O[0] - C[0]))
    f3 = tooth_phase(z3, 0.0, a_co)
    f4 = tooth_phase(z4, lay.gear_rotation_deg["output_gear"], a_co + 180.0)
    out.append(_check("stage2_teeth_interleave", abs(((f3 + f4) % 1.0) - 0.5), 1e-6, "<=", detail="f_driver + f_driven = 0.5 mod 1"))
    return out


def solve_drive_layout(spec: DriveSpec, hn) -> DriveLayout:
    """Return a checked layout. ``hn`` is the (scaled) 13-number linkage tuple a..m.

    With ``spec.module2 is None`` the smallest standard stage-2 module that passes
    every check is used; an explicit value is used as given (failed checks are
    reported on the returned layout instead of raising).
    """
    if spec.ratio not in RATIOS:
        raise LayoutError(f"unknown ratio {spec.ratio!r}; choose from {sorted(RATIOS)}")
    if spec.module2 is not None:
        return _build(spec, float(spec.module2), hn)
    reasons: list[str] = []
    for m2 in STANDARD_MODULES:
        if m2 < spec.module1:
            continue
        try:
            lay = _build(spec, m2, hn)
        except LayoutError:
            reasons.append(f"module {m2:g}: no feasible fold")
            continue
        if lay.passed:
            return lay
        reasons.append(f"module {m2:g}: failed {[c.name for c in lay.checks if not c.passed]}")
    d_oy = math.hypot(float(hn[0]), float(hn[11]))
    hint = (f" Pivot Y is only {d_oy:.0f} mm from the crank axis; a larger --linkage-size is the usual fix."
            if d_oy < 100 else " Try a smaller motor, the other ratio, or an explicit --module2.")
    raise LayoutError("no standard stage-2 module gives a passing layout (" + "; ".join(reasons) + ")." + hint)


def corner_posts(L: DriveLayout):
    x0, x1, y0, y1 = L.inner_rect
    pr = L.spec.post_radius
    return [(x0 + pr, y0 + pr), (x1 - pr, y0 + pr), (x0 + pr, y1 - pr), (x1 - pr, y1 - pr)]


def window_spans(L: DriveLayout):
    """x ranges of the lightening windows, one centred on the motor axis (clamp access)."""
    x0, x1, _, _ = L.inner_rect
    lo, hi = x0 + 14.0, x1 - 14.0
    pitch = WINDOW_W + WINDOW_RIB
    spans, k = [], 0
    while True:
        added = False
        for sgn in ((1,) if k == 0 else (-1, 1)):
            cx = L.M[0] + sgn * k * pitch
            if cx - WINDOW_W / 2 >= lo and cx + WINDOW_W / 2 <= hi:
                spans.append((cx - WINDOW_W / 2, cx + WINDOW_W / 2))
                added = True
        if not added and k > 0:
            break
        k += 1
        if k > 20:
            break
    return sorted(spans)
