"""STEP 2: smooth 3D body.

Profiles (all in sheet pixels, height h above the ground):
  w(h)   half width      (front + mirrored back views, both edges, averaged)
  df(h)  half depth, front (side view, face side)
  db(h)  half depth, back  (side view, back side; tail rows bridged)
Each is a smoothing cubic spline through the MEASURED rows only, so arms /
feet / tail / leaves never shape it. Head top and body bottom are closed with
slope-matched ellipse caps, so the outline is C1 everywhere (no ridges).

Cross-section at height h: superellipse |x/w|^p + |z/d|^p = 1 with
d = df for z < 0 (front) and db for z > 0 (back); p from the top view.
Coordinates: x = character's left (+), y = up, z = back (+). Origin: body
axis on the ground.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import UnivariateSpline

from .analyze import ViewMeasure


def _binned(h, r, step=3.0):
    ok = ~np.isnan(r)
    h, r = h[ok], r[ok]
    edges = np.arange(h.min(), h.max() + step, step)
    idx = np.digitize(h, edges)
    hb, rb = [], []
    for k in np.unique(idx):
        sel = idx == k
        hb.append(h[sel].mean())
        rb.append(np.median(r[sel]))
    order = np.argsort(hb)
    return np.array(hb)[order], np.array(rb)[order]


def _spline(h, r, smooth_px=1.5):
    hb, rb = _binned(h, r)
    return UnivariateSpline(hb, rb, k=3, s=len(hb) * smooth_px ** 2), hb.min(), hb.max()


def _cap(r_m, s, h_m, end, aspect=None):
    """Ellipse r(h) through (h_m, r_m) with slope s, reaching r = 0 at h = end.
    Works upward (end > h_m) and downward (end < h_m)."""
    D = abs(end - h_m)
    sgn = 1.0 if end > h_m else -1.0
    s_out = s * sgn                     # slope measured moving towards the cap end
    s_out = min(s_out, -0.02)

    def slope_of(a):
        u = a - D
        return -r_m * u / max(a * a - u * u, 1e-9)

    lo, hi = D, D * 400
    for _ in range(100):
        mid = (lo + hi) / 2
        if slope_of(mid) < s_out:
            hi = mid
        else:
            lo = mid
    a = (lo + hi) / 2
    u = a - D
    R = r_m / math.sqrt(max(1 - (u / a) ** 2, 1e-9))
    hc = end - sgn * a
    return lambda hh: R * np.sqrt(np.clip(1 - ((np.asarray(hh) - hc) / a) ** 2, 0, 1))


@dataclass
class Profile1D:
    spline: object
    lo: float           # measured range used (cap below lo / above hi)
    hi: float
    bottom: float
    top: float
    cap_lo: object = None
    cap_hi: object = None

    def __call__(self, h):
        h = np.asarray(h, float)
        mid = np.clip(self.spline(np.clip(h, self.lo, self.hi)), 0, None)
        out = np.where(h < self.lo, self.cap_lo(h), np.where(h > self.hi, self.cap_hi(h), mid))
        return np.where((h < self.bottom) | (h > self.top), 0.0, np.clip(out, 0, None))

    def close(self):
        s = self.spline
        self.cap_lo = _cap(float(s(self.lo)), float(s.derivative()(self.lo)), self.lo, self.bottom)
        self.cap_hi = _cap(float(s(self.hi)), float(s.derivative()(self.hi)), self.hi, self.top)
        return self


@dataclass
class BodyShape:
    w: Profile1D
    df: Profile1D
    db: Profile1D
    p: float
    bottom: float
    top: float
    estimated: list = field(default_factory=list)
    measured: dict = field(default_factory=dict)

    # -------- implicit surface --------
    def F(self, x, y, z):
        w = np.maximum(self.w(y), 1e-6)
        d = np.where(z < 0, self.df(y), self.db(y))
        d = np.maximum(d, 1e-6)
        inside_h = (y >= self.bottom) & (y <= self.top)
        f = (np.abs(x) / w) ** self.p + (np.abs(z) / d) ** self.p - 1
        return np.where(inside_h, f, 1e3)

    def grad(self, x, y, z, e=0.35):
        gx = (self.F(x + e, y, z) - self.F(x - e, y, z)) / (2 * e)
        gy = (self.F(x, y + e, z) - self.F(x, y - e, z)) / (2 * e)
        gz = (self.F(x, y, z + e) - self.F(x, y, z - e)) / (2 * e)
        n = np.stack([gx, gy, gz], -1)
        return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)

    # -------- mesh --------
    def mesh(self, n_rings=220, n_seg=160):
        """Triangle mesh: rings of the superellipse at heights spaced by arc
        length of the outline (dense at the caps), closed with a vertex at each pole."""
        hs = np.linspace(self.bottom, self.top, 4000)
        r = np.maximum(self.w(hs), self.df(hs))
        arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(hs), np.diff(r)))])
        hr = np.interp(np.linspace(0, arc[-1], n_rings), arc, hs)[1:-1]
        t = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
        c, s = np.cos(t), np.sin(t)
        ex = 2.0 / self.p
        ux = np.sign(c) * np.abs(c) ** ex
        uz = np.sign(s) * np.abs(s) ** ex
        verts = [(0.0, self.bottom, 0.0)]
        for h in hr:
            w = float(self.w(h))
            df, db = float(self.df(h)), float(self.db(h))
            d = np.where(uz < 0, df, db)
            for k in range(n_seg):
                verts.append((w * ux[k], h, d[k] * uz[k]))
        verts.append((0.0, self.top, 0.0))
        V = np.array(verts)
        faces = []
        nr = len(hr)
        ring = lambda i, k: 1 + i * n_seg + (k % n_seg)  # noqa: E731
        for k in range(n_seg):
            faces.append((0, ring(0, k + 1), ring(0, k)))
        for i in range(nr - 1):
            for k in range(n_seg):
                a, b = ring(i, k), ring(i, k + 1)
                c_, d_ = ring(i + 1, k), ring(i + 1, k + 1)
                faces.append((a, b, d_))
                faces.append((a, d_, c_))
        top = len(V) - 1
        for k in range(n_seg):
            faces.append((top, ring(nr - 1, k), ring(nr - 1, k + 1)))
        return V, np.array(faces)

    def write_obj(self, path, scale=1.0):
        V, Fc = self.mesh()
        # vertex normals from the implicit gradient
        N = self.grad(V[:, 0], V[:, 1], V[:, 2])
        N[0] = (0, -1, 0)
        N[-1] = (0, 1, 0)
        with open(path, "w") as fh:
            fh.write("# mint body, reconstructed from the 4-view sheet\n")
            fh.write("# units: sheet pixels * scale; y up; origin = body axis on the ground; front = -Z\n")
            fh.write("o Body\n")
            for v in V * scale:
                fh.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for n in N:
                fh.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
            for f in Fc + 1:
                fh.write(f"f {f[0]}//{f[0]} {f[1]}//{f[1]} {f[2]}//{f[2]}\n")
        return len(V), len(Fc)

    def to_json(self, path):
        hs = np.linspace(self.bottom, self.top, 257)
        data = {
            "units": "sheet px, height above ground",
            "bottom": self.bottom, "top": self.top, "superellipse_p": self.p,
            "samples": {"h": hs.round(3).tolist(), "half_width": self.w(hs).round(3).tolist(),
                        "half_depth_front": self.df(hs).round(3).tolist(),
                        "half_depth_back": self.db(hs).round(3).tolist()},
            "measured": self.measured, "estimated": self.estimated,
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=1, ensure_ascii=False)


class _RoundTop:
    """Above `h0` blend the width towards the side-view depth, reaching `amount`
    at the head top: the crown becomes a round dome instead of a ridge
    (width >> depth there). The side silhouette is left as measured."""

    def __init__(self, f, w, df, db, h0, top, amount):
        self.f, self.w, self.df, self.db = f, w, df, db
        self.h0, self.top, self.amount = h0, top, amount

    def __call__(self, h):
        h = np.asarray(h, float)
        t = np.clip((h - self.h0) / max(self.top - self.h0, 1e-6), 0, 1)
        t = t * t * (3 - 2 * t) * self.amount
        m = 0.5 * (self.df(h) + self.db(h))          # the (narrower) depth from the side view
        return self.f(h) * (1 - t) + np.minimum(m, self.f(h)) * t


def build(front: ViewMeasure, back: ViewMeasure, side: ViewMeasure, p: float,
          head_aspect: float = 1.0, smooth_px: float = 1.5, side_top_margin: float = 0.10,
          base_round: float = 0.08, round_top: float = 0.0, width_top_margin: float = 0.10) -> BodyShape:
    Hs = side.notes["leaf_base_h"] - side.notes["body_bottom_h"]
    # width: four edges (front L/R, back L/R) -> one symmetric profile
    hW = np.concatenate([front.h, front.h, back.h, back.h])
    rW = np.concatenate([front.left, front.right, back.left, back.right])
    sw, lo_w, hi_w = _spline(hW, rW, smooth_px)
    # depth: side view. The leaf overlaps the back of the head near the top, so the
    # depth is measured only up to side_top_margin below the leaf base.
    hcut = side.notes["leaf_base_h"] - side_top_margin * Hs
    dl = np.where(side.h <= hcut, side.left, np.nan)
    dr = np.where(side.h <= hcut, side.right, np.nan)
    sf, lo_f, hi_f = _spline(side.h, dl, smooth_px)
    # back edge hidden by the tail near the bottom: add the front edge there, scaled by
    # the back/front ratio just above the tail, and fit ONE spline (no seam)
    okb = ~np.isnan(dr)
    lo_b = float(side.h[okb].min())
    band = okb & (side.h < lo_b + 30)
    ratio = float(np.nanmedian(dr[band] / np.where(np.isnan(dl[band]), np.nan, dl[band])))
    fill = np.isnan(dr) & ~np.isnan(dl) & (side.h < lo_b)
    dr2 = np.where(fill, dl * ratio, dr)
    sb, lo_b2, hi_b = _spline(side.h, dr2, smooth_px)

    bottom = float(np.median([front.notes["body_bottom_h"], back.notes["body_bottom_h"]]))
    # bottom cap starts a little above the feet so the base edge is softly rounded
    lo = max(lo_w, lo_f, lo_b2) + base_round * Hs
    # the head is closed by a dome from a little below the leaf base in BOTH profiles:
    # a rounder crown than following the wide leaf-base outline to the very top
    Hf = front.notes["leaf_base_h"] - front.notes["body_bottom_h"]
    hi_w_ = min(hi_w, front.notes["leaf_base_h"] - width_top_margin * Hf)
    hi_d = min(hi_f, hi_b)
    # head top from the width profile: slope-matched ellipse with the given aspect
    r_m, sl = float(sw(hi_w_)), float(sw.derivative()(hi_w_))
    t = -min(sl, -0.02) * head_aspect
    q = t / math.sqrt(1 + t * t)
    R = r_m / math.sqrt(1 - q * q)
    top = hi_w_ - q * head_aspect * R + head_aspect * R

    w = Profile1D(sw, lo, hi_w_, bottom, top).close()
    df = Profile1D(sf, lo, hi_d, bottom, top).close()
    db = Profile1D(sb, lo, hi_d, bottom, top).close()

    if round_top > 0:
        h0 = hi_d                         # start where the depth stops being measured
        w0, df0, db0 = w, df, db
        w = _RoundTop(w0, w0, df0, db0, h0, top, round_top)   # only the width narrows; side view kept
    shape = BodyShape(w, df, db, p, bottom, top)
    shape.measured = {
        "front": front.notes, "back": back.notes, "side": side.notes,
        "width_rows_measured": int(np.sum(~np.isnan(rW))),
        "depth_rows_measured": [int(np.sum(~np.isnan(dl))), int(np.sum(okb))],
        "width_measured_h": [lo, hi_w_], "depth_measured_h": [lo, hi_d],
        "back_front_depth_ratio_at_tail": ratio,
    }
    shape.estimated = [
        f"head top: width measured up to h={hi_w_:.0f}px, depth up to h={hi_d:.0f}px "
        f"(the leaf overlaps the back of the head in the side view); slope-matched ellipse caps "
        f"close at h={top:.0f}px",
        f"body bottom: rounded from h={lo:.0f}px down to h={bottom:.0f}px (feet cover it; the end "
        "height is the body's lower outline seen between the feet)",
        "sides behind the arms: spline bridged between the measured rows above and below the arms",
        f"back behind the tail: front edge x {ratio:.2f} (back/front ratio just above the tail)",
        f"head top roundness: above h={hi_d:.0f}px the width narrows towards the side-view depth "
        f"({round_top * 100:.0f}% at the top) so the crown is a round dome; dome aspect {head_aspect}",
        "left/right symmetry: front and back edges on both sides averaged into one width",
    ]
    return shape
