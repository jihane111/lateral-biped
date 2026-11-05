"""Python port of strandbeest-forge's Jansen linkage kinematics. Coordinates in mm.

Linkage size
------------
``BASE_HOLY_NUMBERS`` are the upstream values (size 1.0). Every dimension a..m
(link lengths, the Z->Y offsets and the crank radius) is multiplied by one
factor, so the linkage keeps its exact proportions and kinematics (the foot
path is the upstream path scaled by that factor). ``DEFAULT_LINKAGE_SIZE`` is
the factor used when no ``--linkage-size`` is given; pass 1.0 to reproduce the
upstream size.
"""
import math
import numpy as np

# [a,b,c,d,e,f,g,h,i,j,k,l,m] from strandbeest-forge/src/kinematics.scad (size 1.0)
BASE_HOLY_NUMBERS = (38.0, 41.5, 39.3, 40.1, 55.8, 39.4, 36.7, 65.7, 49.0, 50.0, 61.9, 7.8, 15.0)
DEFAULT_LINKAGE_SIZE = 2.5


def scaled_holy_numbers(size=DEFAULT_LINKAGE_SIZE):
    """Return the 13 linkage numbers multiplied by ``size`` (must be > 0)."""
    size = float(size)
    if not math.isfinite(size) or size <= 0:
        raise ValueError(f"linkage size must be a positive number, got {size!r}")
    return tuple(v * size for v in BASE_HOLY_NUMBERS)


# Backwards-compatible name: the numbers at the default size.
HOLY_NUMBERS = scaled_holy_numbers(DEFAULT_LINKAGE_SIZE)
NODES = ("Z", "X", "Y", "W", "V", "U", "T", "S")
BARS = (
    ("Z", "X", "crank"), ("X", "W", "j"), ("W", "Y", "b"),
    ("W", "V", "e"), ("Y", "V", "d"), ("X", "U", "k"),
    ("Y", "U", "c"), ("U", "T", "g"), ("V", "T", "f"),
    ("U", "S", "i"), ("T", "S", "h"), ("Z", "Y", "fixed frame"),
)

def _dist(p, q):
    return math.hypot(float(p[0]-q[0]), float(p[1]-q[1]))

def _add_angle(p, angle_deg, length):
    a = math.radians(angle_deg)
    return np.array([p[0] + math.cos(a)*length, p[1] - math.sin(a)*length])

def _get_angle(v1, v2):
    return math.degrees(math.atan2(v2[0]-v1[0], v2[1]-v1[1]))

def _intersect(v1, v2, l1, l2):
    separation = _dist(v1, v2)
    if separation < 1e-10:
        raise ValueError("Coincident circle centres")
    arg = (l2*l2-l1*l1-separation*separation)/(-abs(2*separation*l1))
    if arg < -1-1e-9 or arg > 1+1e-9:
        raise ValueError(f"Circles do not intersect (acos argument={arg:.6g})")
    arg = max(-1.0, min(1.0, arg))
    angle0 = math.degrees(math.acos(arg))
    return _add_angle(v1, angle0 + _get_angle(v2, v1) - 90.0, -l1)

def leg_nodes(crank_angle_deg, hn=HOLY_NUMBERS):
    a,b,c,d,e,f,g,h,i,j,k,l,m = hn
    Z = np.array([0., 0.])
    X = _add_angle(Z, crank_angle_deg, m)
    Y = np.array([a, l])
    W = _intersect(X, Y, j, b)
    V = _intersect(W, Y, e, d)
    U = _intersect(Y, X, c, k)
    T = _intersect(V, U, f, g)
    S = _intersect(T, U, h, i)
    return dict(zip(NODES, (Z, X, Y, W, V, U, T, S)))

def bar_lengths(hn=HOLY_NUMBERS):
    """Nominal length of every bar, keyed by the bar names used in BARS."""
    a, b, c, d, e, f, g, h, i, j, k, l, m = hn
    return {"crank": m, "j": j, "b": b, "e": e, "d": d, "k": k,
            "c": c, "g": g, "f": f, "i": i, "h": h, "fixed frame": math.hypot(a, l)}


def validate_full_cycle(samples=360, hn=HOLY_NUMBERS):
    for angle in np.linspace(0, 360, samples, endpoint=False):
        try:
            pts = leg_nodes(float(angle), hn)
            if not all(np.isfinite(p).all() for p in pts.values()):
                return False, f"Non-finite point at {angle:.2f}°"
        except (ValueError, OverflowError) as exc:
            return False, f"Invalid pose at {angle:.2f}°: {exc}"
    return True, "All sampled poses solved"

def pose_plot_coordinates(crank_angle_deg, phase_deg=0.0, hn=HOLY_NUMBERS):
    pts = leg_nodes(crank_angle_deg + phase_deg, hn)
    return {name: np.array([p[0], -p[1]]) for name, p in pts.items()}
