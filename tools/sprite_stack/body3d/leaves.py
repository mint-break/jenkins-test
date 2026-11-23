"""Leaves: two thin, slightly curved leaf blades on the crown, sliced on the
body's layer grid.

Geometry (sheet px, x = character's left, y = up, z = back, origin = body
axis on the ground), measured from the front and side views:
  * base   : stem at the crown centre, a little inside the head
  * tip    : +-tip_x sideways, up, and BACK (the side view shows the leaves
             leaning back)
  * width  : the front view's width profile along the midrib (ovate leaf)
  * roll   : rotation of the blade about its midrib. roll 80 deg reproduces the
             front AND side widths (but the blade is almost edge-on from
             above); smaller roll lays the blade down so it reads as a leaf
             from above (the top view is an estimated drawing).
The blade is a surface P(u, s) = midrib(u) + w * s * halfwidth(u) + cupping,
with a small bend along the midrib, a serrated edge, a midrib + side veins in
the albedo, and a short stem into the head.

Slicing: a leaf is a tilted thin plate, so its exact cross-section at one
height would be a hairline. Each layer instead takes every part of the leaf
inside its slab [h, h + spacing) and projects it from above (top-most point
wins), which is exactly what that layer covers when the stack is drawn.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage as nd


@dataclass
class LeafParams:
    base: tuple = (1.5, 355.0, 3.0)        # right leaf (mirrored for the left): both blades start at
                                           # the top of ONE shared stem, ~5 px above the crown (349.8)
    tip: tuple = (150.0, 436.0, 108.0)
    width: float = 113.0                   # true max width (sheet px), solved from the front + side views
    roll_deg: float = 75.0
    bend: float = 0.05                     # midrib bend (fraction of length) towards the upper face
    cup: float = 0.10                      # blade edges rise (fraction of half width)
    thickness: float = 8.0
    roll_tip_deg: float | None = None   # twist: roll at the tip (None = no twist)
    fold_deg: float = 15.0       # V fold of the two halves about the midrib (0 = flat)
    fan_deg: float = 8.0         # right leaf turned about the vertical (deg, + = tip forward)
    fan_deg_left: float = 3.0    # left leaf (mirror) turned the other way by this much
    serration: float = 0.035
    serration_count: int = 11
    # width profile along the midrib (u = 0 base .. 1 tip), from the front view
    profile_u: tuple = (0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0)
    profile_w: tuple = (0.0, 8, 44, 68, 85, 100, 99, 95, 80, 60, 30, 0.0)
    stem_len: float = 16.0               # shared vertical stem; lower end inside the head
    stem_radius: float = 4.5
    colours: dict = field(default_factory=lambda: {
        "base": (127, 224, 169), "light": (147, 232, 182), "vein": (80, 180, 132),
        "edge": (36, 104, 80), "stem": (98, 196, 146)})


def _frame(p: LeafParams):
    B = np.array(p.base, float)
    T = np.array(p.tip, float)
    d = T - B
    L = float(np.linalg.norm(d))
    d /= L
    e1 = np.cross(d, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    phi = math.radians(p.roll_deg)
    w = math.cos(phi) * e1 + math.sin(phi) * e2
    n = np.cross(d, w)
    if n[1] < 0:                            # 'upper' face points up
        n = -n
        w = -w
    return B, d, w, n, L


def halfwidth(p: LeafParams, u):
    prof = np.interp(u, p.profile_u, p.profile_w) / max(p.profile_w)
    serr = 1 + p.serration * np.abs(np.sin(math.pi * p.serration_count * u)) - p.serration * 0.5
    return 0.5 * p.width * prof * np.where((u > 0.12) & (u < 0.97), serr, 1.0)


def sample(p: LeafParams, side: int = +1, nu=900, ns=360):
    """Points on both faces of one leaf (+1 = right / +X, -1 = left).
    Returns dict of arrays: pos (N,3), nrm (N,3), col (N,3), uv (N,2)."""
    B, d, w, n, L = _frame(p)
    u = np.linspace(0, 1, nu)
    s = np.linspace(-1, 1, ns)
    U, Sg = np.meshgrid(u, s, indexing="ij")
    hw = halfwidth(p, U)
    # twist: the blade's roll goes from roll_deg at the base to roll_tip_deg at the tip
    # (upright at the stem as the front view shows, lying back towards the tip as the
    # top view shows)
    rt = p.roll_deg if p.roll_tip_deg is None else p.roll_tip_deg
    e1 = np.cross(d, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    phi = np.radians(p.roll_deg + (rt - p.roll_deg) * np.clip(U, 0, 1) ** 1.2)
    w_u = np.cos(phi)[..., None] * e1 + np.sin(phi)[..., None] * e2
    n_u = np.cross(d, w_u)
    sgn = np.sign(n_u[..., 1:2] + 1e-9)
    w_u, n_u = w_u * sgn, n_u * sgn
    mid = B + np.outer(U.ravel() * L, d).reshape(U.shape + (3,)) \
        + (p.bend * L * U ** 2)[..., None] * n_u
    # V fold along the midrib (rounded at the vein): a folded blade is never edge-on
    # from every side at once, so it never collapses to a line while it turns
    e = 0.12
    fold = (np.sqrt(Sg ** 2 + e * e) - e) / (np.sqrt(1 + e * e) - e)
    P = mid + (Sg * hw * math.cos(math.radians(p.fold_deg)))[..., None] * w_u \
        + (hw * fold * math.sin(math.radians(p.fold_deg)) + p.cup * hw * Sg ** 2)[..., None] * n_u
    # surface normal from the parametric derivatives
    dPu = np.gradient(P, axis=0)
    dPs = np.gradient(P, axis=1)
    N = np.cross(dPu, dPs)
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
    N *= np.sign(np.sum(N * n_u, -1, keepdims=True) + 1e-12)      # align with the upper face
    valid = hw > 0.3
    col = _albedo(p, U, Sg)
    out_pos, out_nrm, out_col = [], [], []
    # lens-shaped section: full thickness on the midrib, thin at the edge (a plush leaf)
    th = p.thickness * np.sqrt(np.clip(1 - Sg ** 2, 0, 1)) * np.clip(hw / (0.25 * p.width), 0.3, 1)
    th = np.maximum(th, 2.0)
    for face in (+1, -1):
        Pf = P + face * 0.5 * th[..., None] * N
        out_pos.append(Pf[valid])
        # both faces store the upper-face normal: a pixel can mix points of both faces,
        # and opposite normals would average to nothing. The shader lights thin
        # surfaces from the side facing the viewer (two-sided).
        out_nrm.append(N[valid])
        out_col.append(col[valid])
    pos = np.concatenate(out_pos)
    nrm = np.concatenate(out_nrm)
    colr = np.concatenate(out_col)
    if side < 0:
        pos = pos * np.array([-1, 1, 1])
        nrm = nrm * np.array([-1, 1, 1])
    # fan: turn each leaf about the vertical through the shared stem, the two by
    # different amounts, so in the side view the two blades separate and overlap
    # (as in the art) instead of lying exactly on top of each other
    a = math.radians(p.fan_deg if side > 0 else -p.fan_deg_left)
    ca, sa = math.cos(a), math.sin(a)
    rot = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    piv = np.array([0.0, p.base[1], p.base[2]])
    pos = (pos - piv) @ rot.T + piv
    nrm = nrm @ rot.T
    return {"pos": pos, "nrm": nrm, "col": colr}


def _albedo(p: LeafParams, U, S):
    c = p.colours
    base = np.array(c["base"], float)
    light = np.array(c["light"], float)
    vein = np.array(c["vein"], float)
    edge = np.array(c["edge"], float)
    col = base + (light - base) * (0.5 + 0.5 * np.cos(math.pi * np.clip(np.abs(S), 0, 1)))[..., None] * 0.6
    # midrib
    mr = np.exp(-(S / 0.06) ** 2) * (U < 0.96)
    # side veins: start on the midrib at u_i and run towards the tip as they go outwards
    sv = np.zeros_like(U)
    for ui in (0.18, 0.32, 0.46, 0.60, 0.74):
        sv = np.maximum(sv, np.exp(-((U - ui - 0.30 * np.abs(S)) / 0.022) ** 2) * (np.abs(S) < 0.85))
    v = np.clip(np.maximum(0.8 * mr, 0.45 * sv), 0, 1)[..., None]
    col = col + (vein - col) * v
    # edge line (the art draws a dark outline round the blade)
    e = np.clip((np.abs(S) - 0.93) / 0.07, 0, 1)[..., None]
    col = col + (edge - col) * e
    return col


def stem(p: LeafParams):
    """One shared stem on the body axis, from inside the head up to where the two
    blades start (no fork: both leaves grow from its rounded top)."""
    top = np.array([0.0, p.base[1], p.base[2]])
    t = np.linspace(-1, 0, 50)
    a = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    T_, A_ = np.meshgrid(t, a, indexing="ij")
    rad = np.stack([np.cos(A_), np.zeros_like(A_), np.sin(A_)], -1)
    pos = top + (T_ * p.stem_len)[..., None] * np.array([0, 1.0, 0]) + p.stem_radius * rad
    nrm = rad.reshape(-1, 3)
    pos = pos.reshape(-1, 3)
    # rounded cap
    th = np.linspace(0, math.pi / 2, 12)
    TH, A2 = np.meshgrid(th, a, indexing="ij")
    cap_n = np.stack([np.cos(TH) * np.cos(A2), np.sin(TH), np.cos(TH) * np.sin(A2)], -1).reshape(-1, 3)
    pos = np.concatenate([pos, top + p.stem_radius * cap_n])
    nrm = np.concatenate([nrm, cap_n])
    col = np.tile(np.array(p.colours["stem"], float), (len(pos), 1))
    return {"pos": pos, "nrm": nrm, "col": col}


def all_points(p: LeafParams):
    a = sample(p, +1)
    b = sample(p, -1)
    c = stem(p)
    return {k: np.concatenate([a[k], b[k], c[k]]) for k in a}


def slice_layers(pts, heights, spacing_sheet, scale=1.2, size=512, ss=2):
    """heights: slab bottoms (sheet px). Returns RGBA albedo and normal layers
    (layer-space normals: x = image right, y = image up (back), z = up)."""
    n = size * ss
    c = (n - 1) / 2
    px = np.round(pts["pos"][:, 0] * scale * ss + c).astype(int)
    py = np.round(c - pts["pos"][:, 2] * scale * ss).astype(int)   # image up = back (+z)
    ok = (px >= 0) & (px < n) & (py >= 0) & (py < n)
    alb, nor = [], []
    y = pts["pos"][:, 1]
    for h0 in heights:
        sel = ok & (y >= h0) & (y < h0 + spacing_sheet)
        A = np.zeros((size, size, 4), np.uint8)
        Nm = np.zeros((size, size, 4), np.uint8)
        if sel.any():
            idx = np.flatnonzero(sel)
            order = idx[np.argsort(y[idx])]           # low first, the top-most point wins
            flat = py[order] * n + px[order]
            cov = np.zeros(n * n, bool)
            cov[flat] = True
            colbuf = np.zeros((n * n, 3))
            nbuf = np.zeros((n * n, 3))
            colbuf[flat] = pts["col"][order]
            nm = pts["nrm"][order]
            nbuf[flat] = np.stack([nm[:, 0], nm[:, 2], nm[:, 1]], -1)   # layer space
            cov = cov.reshape(n, n)
            # close tiny holes between samples, then fill their colour from neighbours
            closed = nd.binary_closing(cov, iterations=2) | cov
            fill = closed & ~cov
            if fill.any():
                _, (iy, ix) = nd.distance_transform_edt(~cov, return_indices=True)
                cb = colbuf.reshape(n, n, 3)
                nb = nbuf.reshape(n, n, 3)
                cb[fill] = cb[iy[fill], ix[fill]]
                nb[fill] = nb[iy[fill], ix[fill]]
            cov = closed
            cb = colbuf.reshape(n, n, 3) * cov[..., None]
            nb = nbuf.reshape(n, n, 3) * cov[..., None]
            # downsample (area) -> anti-aliased coverage, premultiplied colour
            a = cov.reshape(size, ss, size, ss).mean((1, 3))
            csum = cb.reshape(size, ss, size, ss, 3).sum((1, 3))
            nsum = nb.reshape(size, ss, size, ss, 3).sum((1, 3))
            cnt = cov.reshape(size, ss, size, ss).sum((1, 3))[..., None]
            m = cnt[..., 0] > 0
            colr = np.where(m[..., None], csum / np.maximum(cnt, 1), 0)
            nv = np.where(m[..., None], nsum / np.maximum(cnt, 1), 0)
            nv /= np.linalg.norm(nv, axis=-1, keepdims=True) + 1e-12
            A[..., :3] = np.clip(colr, 0, 255).astype(np.uint8)
            A[..., 3] = np.round(a * 255).astype(np.uint8)
            Nm[..., :3] = np.where(m[..., None], np.clip(np.round((nv * 0.5 + 0.5) * 255), 0, 255), 0).astype(np.uint8)
            Nm[..., 3] = A[..., 3]
            A[A[..., 3] == 0, :3] = 0
            covd = Nm[..., 3] > 0
            if covd.any():
                d_, (iy, ix) = nd.distance_transform_edt(~covd, return_indices=True)
                pad = (~covd) & (d_ <= 4)
                Nm[pad, :3] = Nm[iy[pad], ix[pad], :3]
        alb.append(A)
        nor.append(Nm)
    return alb, nor


def ortho(pts, view: str, scale=1.0, size=(420, 470), origin=(210, 460)):
    """Orthographic point render for checking: returns (mask, colour, normal) images.
    view: front (from -Z), side (from -X, front on the left), top (from +Y, front down)."""
    W, H = size
    P = pts["pos"]
    if view == "front":
        u, v, depth = P[:, 0], P[:, 1], -P[:, 2]
    elif view == "back":
        u, v, depth = -P[:, 0], P[:, 1], P[:, 2]
    elif view == "side":
        u, v, depth = P[:, 2], P[:, 1], -P[:, 0]
    else:
        u, v, depth = P[:, 0], P[:, 2], P[:, 1]
    x = np.round(origin[0] + u * scale).astype(int)
    y = np.round(origin[1] - v * scale).astype(int)
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    order = np.flatnonzero(ok)[np.argsort(depth[ok])]
    img = np.zeros((H, W, 3))
    msk = np.zeros((H, W), bool)
    img[y[order], x[order]] = pts["col"][order]
    msk[y[order], x[order]] = True
    msk = nd.binary_closing(msk, iterations=1) | msk
    return msk, img
