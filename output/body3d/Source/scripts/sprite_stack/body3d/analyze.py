"""STEP 1: measure the body in each view.

All views share one pixel scale (the sheet draws them at the same size) and
are aligned on the ground (lowest silhouette row = feet bottom).

Per view and per row we take the silhouette edges, then mark rows where
something other than the body forms the edge:
  * arms   : a second outline runs just inside the silhouette edge
  * leaves : everything above the narrowest row of the upper part (leaf base)
  * feet   : below the body's bottom (measured at the centre column, where the
             art shows the body's lower outline between the feet)
  * tail   : side view, back edge below the tail top
The body outline across masked rows is filled by a smooth spline through the
measured rows only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as nd
from scipy.interpolate import PchipInterpolator, UnivariateSpline

from ..views import View


def line_mask(v: View, lum_max: float = 150) -> np.ndarray:
    lum = v.rgb @ np.array([0.299, 0.587, 0.114])
    return (lum < lum_max) & v.mask


@dataclass
class ViewMeasure:
    name: str
    ground: int                      # row of the ground (feet bottom)
    axis: float                      # column of the body axis
    h: np.ndarray                    # height above ground per row (px)
    left: np.ndarray                 # distance axis -> left edge (px), NaN = not body
    right: np.ndarray                # distance axis -> right edge
    raw_left: np.ndarray
    raw_right: np.ndarray
    notes: dict = field(default_factory=dict)


def _edges(v: View):
    H = v.h
    L = np.full(H, np.nan)
    R = np.full(H, np.nan)
    for r in range(H):
        xs = np.flatnonzero(v.mask[r])
        if len(xs):
            L[r], R[r] = xs[0], xs[-1]
    return L, R


def _inner_line(v: View, line: np.ndarray, r: int, start: int, step: int, lim: int):
    """Distance from the silhouette edge to the next outline inward (None if none)."""
    i = start
    while 0 <= i < v.w and line[r, i]:
        i += step
    j = i
    while 0 <= j < v.w and abs(j - start) < lim and v.mask[r, j] and not line[r, j]:
        j += step
    if 0 <= j < v.w and line[r, j] and abs(j - start) < lim and abs(j - i) > 3:
        return abs(j - start)
    return None


def _runs(flags: np.ndarray, gap: int):
    """Close gaps <= gap in a boolean row mask and return (start, end) runs."""
    f = nd.binary_closing(flags, structure=np.ones(gap * 2 + 1)) | flags
    lab, n = nd.label(f)
    return [tuple(np.flatnonzero(lab == k)[[0, -1]]) for k in range(1, n + 1)]


def measure(v: View, kind: str) -> ViewMeasure:
    """kind: 'frontback' (left/right = width) or 'side' (left = front, right = back)."""
    line = line_mask(v)
    L, R = _edges(v)
    rows = np.flatnonzero(~np.isnan(L))
    ground = int(rows[-1])
    top = int(rows[0])
    Hs = ground - top
    h = (ground - np.arange(v.h)).astype(float)
    W = np.nanmax(R - L)
    notes = {}

    # leaf base = narrowest row in the upper 15-40% of the silhouette
    w = nd.uniform_filter1d(np.nan_to_num(R - L, nan=1e9), 5)
    a, b = top + int(0.15 * Hs), top + int(0.40 * Hs)
    neck = a + int(np.argmin(w[a:b]))
    notes["leaf_base_h"] = float(h[neck])

    # body bottom: lowest body pixel at the centre column (between the feet)
    head = (h > h[neck] - 0.25 * Hs) & (h < h[neck] - 0.08 * Hs) & ~np.isnan(L)
    axis = float(np.median((L[head] + R[head]) / 2))
    col = v.mask[:, int(round(axis))]
    # walk up from the ground at the centre column until the body starts
    r = ground
    while r > top and not col[r]:
        r -= 1
    # the art draws the body's bottom outline as a line; if the column is solid
    # down to the ground (feet touching), use the lowest line pixel near the bottom
    if r == ground:
        lc = line[:, int(round(axis))]
        cand = np.flatnonzero(lc[ground - int(0.15 * Hs):ground]) + ground - int(0.15 * Hs)
        r = int(cand[0]) if len(cand) else ground
    body_bottom = r
    notes["body_bottom_h"] = float(h[body_bottom])

    good_l = ~np.isnan(L) & (np.arange(v.h) > neck + 2) & (np.arange(v.h) <= body_bottom)
    good_r = good_l.copy()
    lim = int(0.25 * W)
    if kind == "frontback":
        arm_l = np.zeros(v.h, bool)
        arm_r = np.zeros(v.h, bool)
        for rr in np.flatnonzero(good_l):
            dl = _inner_line(v, line, rr, int(L[rr]), +1, lim)
            dr = _inner_line(v, line, rr, int(R[rr]), -1, lim)
            arm_l[rr] = dl is not None and dl > 0.06 * W
            arm_r[rr] = dr is not None and dr > 0.06 * W
        arms = arm_l | arm_r
        runs = [r_ for r_ in _runs(arms, 8) if r_[1] - r_[0] > 0.04 * Hs]
        if runs:
            s0, s1 = max(runs, key=lambda t: t[1] - t[0])
            # the arm joins the outline above where its inner line becomes visible:
            # extend the masked range upward to where the edge starts to bulge
            ext = int(0.12 * Hs)
            m0, m1 = s0 - ext, min(s1 + int(0.03 * Hs), body_bottom)
            good_l[m0:m1 + 1] = False
            good_r[m0:m1 + 1] = False
            notes["arms_h"] = [float(h[m1]), float(h[m0])]
        # feet: the lowest ~12% of the body height bulge outward at the sides
        fb = body_bottom - int(0.10 * (body_bottom - neck))
        good_l[fb:] = False
        good_r[fb:] = False
        notes["feet_from_h"] = float(h[fb])
    else:
        # side: tail sticks out at the back (right) near the bottom: the back edge
        # jumps outward; mask from the jump down
        dR = np.diff(np.nan_to_num(R, nan=0))
        low = np.arange(v.h - 1) > neck + 0.5 * (body_bottom - neck)
        jumps = np.flatnonzero(low & (dR > 0.04 * W))
        if len(jumps):
            t0 = int(jumps[0]) - int(0.03 * Hs)
            good_r[t0:] = False
            notes["tail_from_h"] = float(h[t0])
        fb = body_bottom - int(0.10 * (body_bottom - neck))
        good_l[fb:] = False
        good_r[fb:] = False
        notes["feet_from_h"] = float(h[fb])

    left = np.where(good_l, axis - L, np.nan)
    right = np.where(good_r, R - axis, np.nan)
    return ViewMeasure(v.name, ground, axis, h, left, right, axis - L, R - axis, notes)


def top_view_exponent(v: View, w: float, db: float, sector_deg: float = 60.0):
    """Superellipse exponent that best matches the top view's outline in the
    sector around the BACK direction (image up). Sides carry the arms, the front
    carries the feet. w / db: the model's half width and back half depth (same
    pixel scale as the sheet). Returns (p, rms error px, centre)."""
    m = nd.binary_fill_holes(v.mask)
    ys, xs = np.nonzero(m)
    cx = (xs.min() + xs.max()) / 2
    col = np.flatnonzero(m[:, int(round(cx))])
    cy = col[0] + db                     # centre = back edge + back depth

    def ray(phi):                        # phi from 'up' towards the right
        dx, dy = np.sin(phi), -np.cos(phi)
        r = 0.0
        while True:
            x, y = int(round(cx + dx * r)), int(round(cy + dy * r))
            if not (0 <= y < m.shape[0] and 0 <= x < m.shape[1]) or not m[y, x]:
                return r
            r += 0.5

    phis = np.radians(np.linspace(-sector_deg, sector_deg, 25))
    meas = np.array([ray(f) for f in phis])
    best = (9e9, 2.0)
    for p in np.linspace(1.6, 4.0, 97):
        model = 1 / ((np.abs(np.sin(phis)) / w) ** p + (np.abs(np.cos(phis)) / db) ** p) ** (1 / p)
        err = float(np.sqrt(np.mean((model - meas) ** 2)))
        if err < best[0]:
            best = (err, float(p))
    return best[1], best[0], (float(cx), float(cy))
