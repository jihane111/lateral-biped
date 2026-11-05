"""Assign each moving link of one leg to a z layer so no two links collide.

Why this exists
---------------
Upstream ``leg.scad`` alternates link layers only "to mimic a stacked assembly"
and avoid z-fighting in renders. With that two-layer pattern several pivots have
two or more links in the same layer (for example c, g and i at node U), which
cannot be built. Here two links must be on different layers when

* they share a pivot node, or
* their swept outlines come within (half-width + half-width + clearance) of each
  other at any sampled crank angle.

The shaft hardware is an obstacle too: the crank sits in layer 0 and the retainer
collar plus shaft tip occupy the z band of layer 1, so any link whose swept
outline comes within ``axis_radius`` of the crank axis (links j and k, which
start on the orbiting crank pin and sweep across the axis) is kept out of layers
0 and 1.

The minimum number of layers is found by exhaustive search (11 links, trivial),
with the crank fixed to layer 0 (next to the housing). Among the minimum-layer
solutions the one needing the fewest spacer washers at pivots is returned.

Pure NumPy/standard library: no CadQuery needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .kinematics import BARS, HOLY_NUMBERS, NODES, leg_nodes


@dataclass(frozen=True)
class LayerAssignment:
    layers: dict[str, int]              # bar name -> layer index (0 = innermost)
    n_layers: int
    conflicts: tuple[tuple[str, str, str], ...]   # (bar, bar, reason)
    spacers_per_node: dict[str, int]    # layers skipped between the outermost and innermost link at a node
    near_axis_links: tuple[str, ...] = ()         # links that sweep within axis_radius of the crank axis
    axis_layers: int = 2                          # layers 0..axis_layers-1 hold the crank and shaft hardware
    axis_radius: float = 7.0

    @property
    def total_spacer_layers(self) -> int:
        return sum(self.spacers_per_node.values())

    def to_dict(self):
        return {"layers_innermost_is_0": self.layers, "n_layers": self.n_layers,
                "spacer_layers_per_node": self.spacers_per_node,
                "total_spacer_layers": self.total_spacer_layers,
                "links_sweeping_the_crank_axis": list(self.near_axis_links),
                "layers_reserved_for_shaft_hardware": self.axis_layers,
                "conflict_pairs": [list(c) for c in self.conflicts]}


def _point_segment_distance(p, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _segment_distance(p1, p2, q1, q2) -> float:
    point_segment = _point_segment_distance

    def crosses(a, b, c, d):
        r, s = b - a, d - c
        den = r[0] * s[1] - r[1] * s[0]
        if abs(den) < 1e-12:
            return False
        w = c - a
        t = (w[0] * s[1] - w[1] * s[0]) / den
        u = (w[0] * r[1] - w[1] * r[0]) / den
        return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0

    if crosses(p1, p2, q1, q2):
        return 0.0
    return min(point_segment(p1, q1, q2), point_segment(p2, q1, q2),
               point_segment(q1, p1, p2), point_segment(q2, p1, p2))


def moving_bars():
    """The eleven bars that move. The fixed frame bar is the housing in the compact design."""
    return [b for b in BARS if b[2] != "fixed frame"]


def conflict_graph(hn=HOLY_NUMBERS, widths: dict[str, float] | None = None,
                   clearance: float = 0.5, step_deg: float = 1.0):
    bars = moving_bars()
    widths = widths or {}
    w = {name: widths.get(name, 6.0) for _, _, name in bars}
    pairs: dict[tuple[int, int], str] = {}
    for i in range(len(bars)):
        for j in range(i + 1, len(bars)):
            if set(bars[i][:2]) & set(bars[j][:2]):
                pairs[(i, j)] = "shared pivot"
    for ang in np.arange(0.0, 360.0, step_deg):
        pts = leg_nodes(float(ang), hn)
        for i in range(len(bars)):
            for j in range(i + 1, len(bars)):
                if (i, j) in pairs:
                    continue
                a, b, ni = bars[i]
                c, d, nj = bars[j]
                if _segment_distance(pts[a], pts[b], pts[c], pts[d]) < (w[ni] + w[nj]) / 2.0 + clearance:
                    pairs[(i, j)] = "swept outlines meet"
    return bars, pairs


def links_near_axis(hn=HOLY_NUMBERS, widths: dict[str, float] | None = None,
                    radius: float = 7.0, clearance: float = 0.5, step_deg: float = 1.0) -> dict[str, float]:
    """Edge-to-axis gap of every non-crank link at its closest approach to the crank axis."""
    widths = widths or {}
    origin = np.zeros(2)
    gap: dict[str, float] = {}
    for ang in np.arange(0.0, 360.0, step_deg):
        pts = leg_nodes(float(ang), hn)
        for a, b, name in moving_bars():
            if name == "crank":
                continue
            d = _point_segment_distance(origin, pts[a], pts[b]) - widths.get(name, 6.0) / 2.0 - radius - clearance
            gap[name] = min(gap.get(name, math.inf), d)
    return gap


def assign_layers(hn=HOLY_NUMBERS, widths: dict[str, float] | None = None,
                  clearance: float = 0.5, step_deg: float = 1.0, max_layers: int = 8,
                  axis_radius: float = 7.0, axis_layers: int = 2) -> LayerAssignment:
    bars, pairs = conflict_graph(hn, widths, clearance, step_deg)
    n = len(bars)
    axis_gap = links_near_axis(hn, widths, axis_radius, clearance, step_deg)
    near_axis = {name for name, g in axis_gap.items() if g < 0.0}
    banned = [(bars[i][2] in near_axis) for i in range(n)]
    adj = [[False] * n for _ in range(n)]
    for (i, j) in pairs:
        adj[i][j] = adj[j][i] = True

    def spacers(colouring):
        out = {}
        for node in NODES:
            ls = [colouring[i] for i, (a, b, _) in enumerate(bars) if node in (a, b)]
            out[node] = (max(ls) - min(ls) + 1 - len(set(ls))) if ls else 0
        return out

    crank_index = [i for i, b in enumerate(bars) if b[2] == "crank"][0]
    order = [crank_index] + [i for i in range(n) if i != crank_index]
    for k in range(1, max_layers + 1):
        best = None
        col = [-1] * n
        col[order[0]] = 0

        def rec(pos):
            nonlocal best
            if pos == n:
                if len(set(col)) != k:
                    return
                score = (sum(spacers(col).values()), tuple(col))
                if best is None or score < best[0]:
                    best = (score, list(col))
                return
            i = order[pos]
            for c in range(k):
                if banned[i] and c < axis_layers:
                    continue
                if all(not (adj[i][j] and col[j] == c) for j in order[:pos]):
                    col[i] = c
                    rec(pos + 1)
                    col[i] = -1

        rec(1)
        if best is not None:
            colouring = best[1]
            return LayerAssignment(
                layers={bars[i][2]: colouring[i] for i in range(n)},
                n_layers=k,
                conflicts=tuple((bars[i][2], bars[j][2], why) for (i, j), why in sorted(pairs.items())),
                spacers_per_node=spacers(colouring),
                near_axis_links=tuple(sorted(near_axis)), axis_layers=axis_layers, axis_radius=axis_radius)
    raise RuntimeError(f"no collision-free layer assignment with <= {max_layers} layers")


def same_layer_collisions(offsets_in_layers: dict[str, float]) -> list[tuple[str, str, str]]:
    """List (node, bar, bar) triples that share a pivot AND a layer in a given assignment."""
    out = []
    bars = moving_bars()
    for node in NODES:
        incident = [(name, offsets_in_layers[name]) for a, b, name in bars if node in (a, b)]
        for i in range(len(incident)):
            for j in range(i + 1, len(incident)):
                if incident[i][1] == incident[j][1]:
                    out.append((node, incident[i][0], incident[j][0]))
    return out


def clearance_around_axis(hn=HOLY_NUMBERS, widths: dict[str, float] | None = None, radius: float = 7.0,
                          layers: dict[str, int] | None = None, axis_layers: int = 2) -> float | None:
    """Smallest gap between the shaft-hardware disc and links sharing its layers.

    Only links assigned to layers 0..axis_layers-1 can touch the shaft hardware.
    Returns None when no other link lives in those layers.
    """
    gaps = links_near_axis(hn, widths, radius, clearance=0.0)
    if layers is not None:
        gaps = {k: v for k, v in gaps.items() if layers[k] < axis_layers}
    return float(min(gaps.values())) if gaps else None
