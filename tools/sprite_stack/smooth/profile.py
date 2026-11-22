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


# ---------------------------------------------------------------------------
# v2: follow the art's real outline; replace only rows covered by bread / feet
# ---------------------------------------------------------------------------

def _raw_edges(v: View):
    """Raw left/right silhouette edges per row and whether each edge pixel is part (bread)."""
    m = v.mask
    H = m.shape[0]
    L = np.full(H, np.nan)
    R = np.full(H, np.nan)
    pl = np.zeros(H, bool)
    pr = np.zeros(H, bool)
    for r in range(H):
        xs = np.flatnonzero(m[r])
        if len(xs):
            L[r], R[r] = xs[0], xs[-1] + 1
            if v.part is not None:
                pl[r] = v.part[r, xs[0]:xs[0] + 4].any()
                pr[r] = v.part[r, max(xs[-1] - 3, 0):xs[-1] + 1].any()
    return L, R, pl, pr


def _half_extent(v: View, h_head=(0.60, 0.72)):
    """Half extent of the body per height from the cleaner side of the view.

    Returns (h array 0..1 of the view's full height incl. leaves, r in view px,
    neck height, which side was used, contaminated-row mask)."""
    from scipy.ndimage import gaussian_filter1d
    L, R, pl, pr = _raw_edges(v)
    rows = np.flatnonzero(~np.isnan(L))
    top, bottom = rows[0], rows[-1]
    H = bottom - top
    neck = int(round(detect_neck(v) * v.h))
    h = (bottom - np.arange(v.h)) / H
    head = (h >= h_head[0]) & (h <= h_head[1]) & ~np.isnan(L)
    axis = float(np.median((L[head] + R[head]) / 2))
    use_right = pr[neck:bottom].sum() <= pl[neck:bottom].sum()
    edge, bad = (R, pr) if use_right else (L, pl)
    r = np.abs(edge - axis)
    return h, r, (bottom - neck) / H, ("right" if use_right else "left"), bad, axis, H


def estimate_v2(views: dict[str, View], feet_h: float = 0.24, flare: float = 0.04,
                base_round: float = 0.06, dome_aspect: float = 0.75, smooth: float = 0.035,
                front_tuck: float = 0.10) -> Profile:
    """Body profile that keeps the art's outline.

    * width  : back view, the edge without bread (left/right assumed symmetric)
    * depth  : side view, the back edge (front assumed = back: the front edge is
               covered by face line, arm and bread)
    * head   : measured up to just under the leaf base, then a slope-matched
               ellipse dome (aspect `dome_aspect`, 1 = round) closes the top
    * bottom : below `feet_h` the feet cover the outline: keep the widest body
               radius, flare it slightly outward and round the ground edge
    """
    from scipy.ndimage import gaussian_filter1d
    out = {}
    info = {}
    for key, v in (("width", views["back"]), ("depth", views["side"])):
        h, r, h_neck, side, bad, axis, H = _half_extent(v)
        sel = (h >= 0) & (h <= h_neck - 0.012) & ~np.isnan(r)
        hs, rs, bs = h[sel][::-1], r[sel][::-1], bad[sel][::-1]
        # rows where the used edge is bread: interpolate across them
        if bs.any() and (~bs).sum() > 2:
            rs = rs.copy()
            rs[bs] = np.interp(hs[bs], hs[~bs], rs[~bs])
        rs = gaussian_filter1d(rs, max(1.0, 0.01 * H), mode="nearest")
        grid = np.linspace(0, hs.max(), 400)
        rg = np.interp(grid, hs, rs)
        # lower body is full: below its widest point the outline never pulls back in
        # (the notches there are where arm / feet line art meets the body)
        lower = grid <= 0.5
        i_w = int(np.argmax(np.where(lower, rg, -1)))
        rg[:i_w] = np.maximum.accumulate(rg[:i_w + 1][::-1])[::-1][:i_w]
        # bottom: feet zone
        zone = grid < feet_h
        plateau = rg[(grid >= feet_h) & (grid <= feet_h + 0.18)].max()
        t = np.clip(1 - grid / feet_h, 0, 1)
        rg = np.where(zone, plateau * (1 + flare * t * t), rg)
        # strong smoothing: pixel-art outline steps would show as ridges on the stack
        rg = gaussian_filter1d(rg, smooth * 400, mode="nearest")
        # rounded ground edge (quarter-circle fillet of radius base_round*H)
        rr = base_round
        f = np.clip(grid / rr, 0, 1)
        rg = rg - rr * H * (1 - np.sqrt(1 - (1 - f) ** 2))
        out[key] = (grid, rg)
        info[key] = dict(side=side, h_neck=h_neck, H=H, h_meas=float(hs.max()))

    # common scale: both views in px of their own height; normalise to the back view
    kd = info["width"]["H"] / info["depth"]["H"]
    gW, rW = out["width"]
    gD, rD = out["depth"]
    rD = rD * kd
    h_meas = min(gW.max(), gD.max())
    Hpx = info["width"]["H"]

    def dome(grid, rg):
        g = grid * Hpx
        r_m = float(np.interp(h_meas * Hpx, g, rg))
        sl = np.gradient(rg, g)
        s = float(np.interp(h_meas * Hpx, g, sl))
        s = min(s, -0.05)
        t_ = -s * dome_aspect
        q = t_ / np.sqrt(1 + t_ * t_)
        R = r_m / np.sqrt(1 - q * q)
        a = dome_aspect * R
        hc = h_meas * Hpx - q * a
        return R, a, hc, hc + a

    dW, dD = dome(gW, rW), dome(gD, rD)
    h_top_px = 0.5 * (dW[3] + dD[3])

    def make(grid, rg):
        g = grid * Hpx
        r_m = float(np.interp(h_meas * Hpx, g, rg))
        s = min(float(np.interp(h_meas * Hpx, g, np.gradient(rg, g))), -0.05)
        D = h_top_px - h_meas * Hpx
        # ellipse through (h_meas, r_m) with slope s, closing at h_top (solve the vertical semi-axis)
        def slope_of(a):
            u = a - D
            return -r_m * u / max(a * a - u * u, 1e-9)
        lo, hi = D, D * 400
        for _ in range(100):
            mid = (lo + hi) / 2
            if slope_of(mid) < s:
                hi = mid
            else:
                lo = mid
        a = (lo + hi) / 2
        u = a - D
        R = r_m / np.sqrt(max(1 - (u / a) ** 2, 1e-9))
        hc = h_top_px - a

        def fn(hn):
            hp = np.asarray(hn, float) * h_top_px
            meas = np.interp(hp, g, rg)
            cap = R * np.sqrt(np.clip(1 - ((hp - hc) / a) ** 2, 0, 1))
            return np.clip(np.where(hp <= h_meas * Hpx, meas, cap), 0, None)
        return fn

    fw, fd = make(gW, rW), make(gD, rD)
    # front of the lower body tucks in a little so the (separate) feet can sit in front
    feet_n = feet_h * Hpx / h_top_px

    def ffront(hn, fd=fd):
        hn = np.asarray(hn, float)
        t = np.clip(1 - hn / feet_n, 0, 1)
        t = t * t * (3 - 2 * t)
        return fd(hn) * (1 - front_tuck * t)
    est = [
        f"head top: outline measured up to {h_meas:.2f} of the view height (just under the leaf base); "
        f"above it a round dome (aspect {dome_aspect}) closes at {h_top_px / Hpx:.2f}, hidden by the leaf stems",
        f"left/right: back view {info['width']['side']} edge (no bread); body assumed left-right symmetric",
        f"front/back depth: side view {info['depth']['side']} edge; front assumed equal to back "
        "(face line, arm and bread cover the front edge)",
        f"below {feet_h:.2f} of the height the feet cover the outline: widest body radius kept, flared "
        f"{flare * 100:.0f}% outward, ground edge rounded ({base_round:.2f} of the height); feet are separate parts",
    ]
    est.append(f"front of the body below the feet line tucked in up to {front_tuck * 100:.0f}% "
               "to make room for the feet")
    prof = Profile(h_top_px, fw, fw, ffront, fd, h_meas * Hpx / h_top_px, est)
    prof.shaped = True
    prof.raw = {"width": (gW, rW, Hpx), "depth": (gD, rD, Hpx)}
    return prof


def top_exponent(top: View) -> float:
    """Cross-section roundness from the top view: superellipse exponent that
    passes through the measured radius on the two BACK diagonals (the front
    diagonals carry the feet, the sides carry arm and bread)."""
    m = top.mask & ~(top.part if top.part is not None else np.zeros_like(top.mask))
    m = nd.binary_fill_holes(m)
    ys, xs = np.nonzero(m)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2

    def ray(dx, dy):
        r = 0.0
        while True:
            x, y = int(round(cx + dx * r)), int(round(cy + dy * r))
            if not (0 <= y < m.shape[0] and 0 <= x < m.shape[1]) or not m[y, x]:
                return r
            r += 0.5

    rz = ray(0, -1)                         # back (image up)
    rx = min(ray(-1, 0), ray(1, 0))         # inner of left/right (arm / bread bulge outward)
    ps = []
    for sx in (-1, 1):
        d = ray(sx / np.sqrt(2), -1 / np.sqrt(2))
        # point (d/sqrt2, d/sqrt2) on |x/rx|^p + |z/rz|^p = 1
        a, b = d / np.sqrt(2) / rx, d / np.sqrt(2) / rz
        lo, hi = 1.2, 8.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if a ** mid + b ** mid > 1:
                lo = mid
            else:
                hi = mid
        ps.append((lo + hi) / 2)
    return float(np.clip(np.median(ps), 1.8, 4.0))
