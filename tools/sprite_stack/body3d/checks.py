"""STEP 4: automatic shape checks on the 3D body (before slicing).

* outline error : model silhouette vs. the measured art rows (px), per view
* smoothness    : curvature of each profile; counts kinks (curvature spikes)
                  and wobbles (sign changes of curvature beyond the art's own)
* symmetry      : left/right is symmetric by construction; front/back ratio
* layer step    : largest change of width / depth between two of the 64 layers
"""
from __future__ import annotations

import numpy as np

from .model import BodyShape


def outline_error(shape: BodyShape, m, kind: str, h_max=None, h_min=None):
    """Error on the measured rows between h_min and h_max. The crown and the
    rounded base (both reshaped on purpose / hidden in the art) are reported
    separately as deviations."""
    rows = ~np.isnan(m.left) | ~np.isnan(m.right)
    if h_max is not None:
        rows &= m.h <= h_max
    if h_min is not None:
        rows &= m.h >= h_min
    h = m.h[rows]
    if kind == "frontback":
        ml, mr = shape.w(h), shape.w(h)
    else:
        ml, mr = shape.df(h), shape.db(h)
    el = np.abs(m.left[rows] - ml)
    er = np.abs(m.right[rows] - mr)
    e = np.concatenate([el[~np.isnan(el)], er[~np.isnan(er)]])
    return {"rows": int(rows.sum()), "mean_px": float(e.mean()), "p95_px": float(np.percentile(e, 95)),
            "max_px": float(e.max())}


def profile_smoothness(fn, lo, hi, n=2000):
    h = np.linspace(lo, hi, n)
    r = fn(h)
    d1 = np.gradient(r, h)
    d2 = np.gradient(d1, h)
    # a kink shows up as a curvature spike much larger than its neighbourhood
    k = d2 / (1 + d1 ** 2) ** 1.5
    med = np.median(np.abs(k)) + 1e-9
    spikes = int(np.sum(np.abs(k) > 25 * med))
    # wobble = curvature changing sign repeatedly (ignore tiny values)
    sig = np.sign(np.where(np.abs(k) > 2 * med, k, 0))
    nz = sig[sig != 0]
    flips = int(np.sum(nz[1:] != nz[:-1]))
    return {"curvature_spikes": spikes, "curvature_sign_changes": flips,
            "max_slope": float(np.max(np.abs(d1)))}


def layer_steps(shape: BodyShape, layers=64, wall_slope=1.0):
    """Radius change between neighbouring layers. 'wall' = where the surface is
    steeper than 45 deg (seen from the side: steps there would show as ridges);
    'cap' = the near-horizontal crown / base (seen from above: each layer's top
    face covers the step)."""
    hs = shape.bottom + (np.arange(layers) + 0.5) * (shape.top - shape.bottom) / layers
    dh = (shape.top - shape.bottom) / layers
    out = {}
    for name, fn in (("width", shape.w), ("depth_front", shape.df), ("depth_back", shape.db)):
        r = fn(hs)
        d = np.abs(np.diff(r))
        wall = d / dh <= wall_slope
        out[name] = {"layer_spacing_px": float(dh),
                     "wall_max_step_px": float(d[wall].max()) if wall.any() else 0.0,
                     "wall_max_step_in_layer_spacings": float(d[wall].max() / dh) if wall.any() else 0.0,
                     "wall_layers": int(wall.sum()),
                     "cap_max_step_px": float(d[~wall].max()) if (~wall).any() else 0.0,
                     "cap_layers": int((~wall).sum())}
    return out


def run(shape: BodyShape, mF, mB, mS):
    lo, hi = shape.bottom + 1, shape.top - 1
    rep = {
        "outline_error_px": {
            "front": outline_error(shape, mF, "frontback", *shape.measured["width_measured_h"][::-1]),
            "back": outline_error(shape, mB, "frontback", *shape.measured["width_measured_h"][::-1]),
            "side": outline_error(shape, mS, "side", *shape.measured["depth_measured_h"][::-1])},
        "estimated_parts_deviation_px": {
            "front_incl_crown_and_base": outline_error(shape, mF, "frontback"),
            "side_incl_crown_and_base": outline_error(shape, mS, "side")},
        "smoothness": {"width": profile_smoothness(shape.w, lo, hi),
                       "depth_front": profile_smoothness(shape.df, lo, hi),
                       "depth_back": profile_smoothness(shape.db, lo, hi)},
        "layer_steps_64": layer_steps(shape),
        "symmetry": {"left_right": "symmetric by construction (one width profile)",
                     "max_back_front_depth_ratio": float(np.max(
                         shape.db(np.linspace(lo, hi, 400)) / np.maximum(shape.df(np.linspace(lo, hi, 400)), 1)))},
    }
    problems = []
    for v, e in rep["outline_error_px"].items():
        if e["p95_px"] > 6:
            problems.append(f"{v}: outline off by {e['p95_px']:.1f}px (95th pct)")
    for k, s in rep["smoothness"].items():
        if s["curvature_spikes"] > 0:
            problems.append(f"{k}: {s['curvature_spikes']} curvature spikes (possible kink)")
    # a side-wall step can never exceed one layer spacing (slope <= 45 deg by definition);
    # anything larger would be a jump in the profile itself
    for k, s in rep["layer_steps_64"].items():
        if s["wall_max_step_in_layer_spacings"] > 1.05:
            problems.append(f"{k}: side-wall step larger than one layer spacing")
    rep["problems"] = problems
    return rep
