"""Estimate the body's width / depth profile per height from the view silhouettes.

Limbs, the held bread and the leaves stick OUT of the body outline, so for each
outline edge (front-left, front-right, side-front, side-back) we fit a smooth
curve that hugs the INNER side of the measured edge: points that bulge
outward are down-weighted iteratively (asymmetric robust fit). The head top,
hidden behind the leaf stems, is extrapolated as an elliptic dome.
Everything not directly measured is listed in `estimated` for the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as nd

from ..views import View, body_span, detect_neck


@dataclass
class Edges:
    """Per image row: left / right edge columns of the silhouette (NaN = empty)."""
    rows: np.ndarray
    left: np.ndarray
    right: np.ndarray
    top: int          # first silhouette row (incl. leaves)
    bottom: int       # last silhouette row
    neck: int         # row where leaves meet the head


def edges(v: View) -> Edges:
    m = nd.binary_fill_holes(body_span(v)) if v.part is not None else v.mask
    H = m.shape[0]
    L = np.full(H, np.nan)
    R = np.full(H, np.nan)
    for r in range(H):
        xs = np.flatnonzero(m[r])
        if len(xs):
            L[r], R[r] = xs[0], xs[-1] + 1
    have = np.flatnonzero(~np.isnan(L))
    neck = int(round(detect_neck(v) * v.h))
    return Edges(np.arange(H), L, R, int(have[0]), int(have[-1]), neck)


def inner_fit(y, e, outward_sign, deg=6, iters=40, tol=2.0):
    """Polynomial fit of edge e(y) that ignores outward bulges.

    outward_sign = -1 for a left edge (outward = smaller x), +1 for a right edge.
    """
    y = np.asarray(y, float)
    e = np.asarray(e, float)
    t = (y - y.min()) / max(np.ptp(y), 1)
    w = np.ones_like(e)
    for _ in range(iters):
        c = np.polyfit(t, e, deg, w=w)
        res = (e - np.polyval(c, t)) * outward_sign   # >0: bulges outward
        w = np.where(res > tol, 0.02, 1.0)
    return lambda yy: np.polyval(c, (np.asarray(yy, float) - y.min()) / max(np.ptp(y), 1))


@dataclass
class Profile:
    """Body extents per normalised height h in [0, 1] (0 = ground, 1 = head top).
    All lengths are in 'view pixels' of the front view scale."""
    height: float
    left: callable      # h -> distance from axis to the left edge  (character's right, -X)
    right: callable     # h -> distance to the right edge (+X)
    front: callable     # h -> distance to the front (-Z)
    back: callable      # h -> distance to the back (+Z)
    measured_top: float  # h above which the outline is extrapolated (hidden by leaves)
    estimated: list = field(default_factory=list)


def _dome(h_rows, r_rows, h_neck):
    """Fit r^2 = a + b h + c h^2 on the upper body and return h where r -> 0."""
    sel = h_rows > h_neck - 0.35
    A = np.stack([np.ones(sel.sum()), h_rows[sel], h_rows[sel] ** 2], 1)
    coef, *_ = np.linalg.lstsq(A, r_rows[sel] ** 2, rcond=None)
    roots = np.roots(coef[::-1])
    roots = roots[np.isreal(roots)].real
    roots = roots[roots > h_neck]
    return (float(roots.min()) if len(roots) else h_neck + 0.1), coef


def estimate(views: dict[str, View], margin: float = 0.07, deg: int = 4) -> Profile:
    """Left/right extents come from the BACK view (mirrored): it shows the body
    almost free of arms, feet and bread. Front/back extents come from the SIDE
    view. The front view is kept for checking only (arms + bread cover its
    outline). Measurement stops `margin` of the height below the leaf base,
    where leaf stems still widen the silhouette; above that the head is an
    elliptic dome fitted to the upper body."""
    f, s, b = views["front"], views["side"], views["back"]
    eF, eS, eB = edges(f), edges(s), edges(b)
    kS = (eF.bottom - eF.top) / max(eS.bottom - eS.top, 1)
    kB = (eF.bottom - eF.top) / max(eB.bottom - eB.top, 1)

    def fit(e: Edges, k: float):
        top_row = e.neck + int(margin * (e.bottom - e.top))
        rows = np.arange(top_row, e.bottom + 1)
        rows = rows[~np.isnan(e.left[rows])]
        hpx = (e.bottom - rows) * k
        return (hpx, inner_fit(hpx, e.left[rows] * k, -1, deg=deg),
                inner_fit(hpx, e.right[rows] * k, +1, deg=deg),
                (e.bottom - top_row) * k, (e.bottom - e.neck) * k)

    hB, lB, rB, mB, nB = fit(eB, kB)
    hS, fS, bS, mS, nS = fit(eS, kS)
    h_meas = min(mB, mS)
    h_neck = min(nB, nS)
    hh = np.linspace(0, h_meas, 240)
    mid = h_meas * 0.5
    cB = (lB(mid) + rB(mid)) / 2
    cS = (fS(mid) + bS(mid)) / 2
    curves = {
        "left": rB(hh) - cB,     # mirrored back view: its right edge is the character's right (-X)
        "right": cB - lB(hh),
        "front": cS - fS(hh),
        "back": bS(hh) - cS,
    }
    # smooth the measured curves (removes polynomial wiggle -> no wavy shading)
    from scipy.ndimage import gaussian_filter1d
    curves = {k: gaussian_filter1d(r, 6, mode="nearest") for k, r in curves.items()}
    # head dome: the leaves sit on the head top, so the head ends just above the leaf base.
    # Above the last measured height the outline continues as a quarter ellipse
    # (zero slope where it starts = rounded, broad head as in the art).
    h_top = h_neck + 0.05 * (eB.bottom - eB.top) * kB
    est = [
        f"head top: measured up to {h_meas:.0f}px; above it an elliptic dome closing at {h_top:.0f}px "
        f"(leaf base at {h_neck:.0f}px hides the real head top in every view)",
        "left/right extents: back view (mirrored); arms/feet/bread treated as outward bulges",
        "front/back extents: side view inner envelope; arm, bread and foot ignored",
    ]
    D = h_top - h_meas

    def make(key):
        r = curves[key]
        r_m = float(r[-1])
        slope = float(np.gradient(r, hh)[-8:].mean())      # dr/dh at the end of the measured part
        # ellipse r = R sqrt(1 - ((h-hc)/a)^2) closing at h_top, matching r_m and slope at h_meas
        def slope_of(a):
            u = a - D
            return -r_m * u / max(a * a - u * u, 1e-9)
        lo, hi = D, D * 200
        if slope >= 0 or slope_of(hi) > slope:
            a = D if slope >= 0 else hi
        else:
            for _ in range(80):
                mid = (lo + hi) / 2
                if slope_of(mid) < slope:
                    hi = mid
                else:
                    lo = mid
            a = (lo + hi) / 2
        u = a - D
        R = r_m / np.sqrt(max(1 - (u / a) ** 2, 1e-9))
        hc = h_top - a

        def fn(h):
            hp = np.asarray(h, float) * h_top
            meas = np.interp(hp, hh, r)
            cap = R * np.sqrt(np.clip(1 - ((hp - hc) / a) ** 2, 0, 1))
            return np.clip(np.where(hp <= h_meas, meas, cap), 0, None)
        return fn

    prof = Profile(h_top, make("left"), make("right"), make("front"), make("back"), h_meas / h_top, est)
    prof.axis = {"back_view_axis_px": float(cB), "side_view_axis_px": float(cS), "kB": kB, "kS": kS}
    return prof
