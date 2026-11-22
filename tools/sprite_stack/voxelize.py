"""Visual-hull voxelisation + surface colouring. Arrays are indexed [x, y, z]."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as nd

from .views import View, detect_neck, protrusion_mask, resize_mask, resize_view

OFFS26 = np.array([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                   if (dx, dy, dz) != (0, 0, 0)])


@dataclass
class VoxelModel:
    occ: np.ndarray                 # bool [X,Y,Z]
    rgb: np.ndarray                 # uint8 [X,Y,Z,3]
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.occ.shape


def shift(a: np.ndarray, o, fill=False) -> np.ndarray:
    """out[p] = a[p + o] (out of range -> fill)."""
    out = np.full_like(a, fill)
    src, dst = [], []
    for d, n in zip(o, a.shape):
        src.append(slice(max(d, 0), n + min(d, 0)))
        dst.append(slice(max(-d, 0), n - max(d, 0)))
    out[tuple(dst)] = a[tuple(src)]
    return out


def normals26(occ: np.ndarray) -> np.ndarray:
    """Normal = sum of unit offsets towards empty 26-neighbours (0 for interior)."""
    n = np.zeros(occ.shape + (3,), np.float32)
    for o in OFFS26:
        empty = ~shift(occ, o, fill=False)
        n += empty[..., None] * (o / np.linalg.norm(o)).astype(np.float32)
    n *= occ[..., None]
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    return np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-6)


def surface(occ: np.ndarray) -> np.ndarray:
    s = np.zeros_like(occ)
    for o in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        s |= ~shift(occ, o, fill=False)
    return s & occ


def _best_shift(ref: np.ndarray, mov: np.ndarray, width: int) -> np.ndarray:
    """Place `mov` (h x w') into an h x width canvas at the x-offset maximising IoU with ref."""
    h, w = mov.shape[:2]
    best, best_dx = -1.0, 0
    for dx in range(-w, width + 1):
        c = np.zeros((h, width), bool)
        a, b = max(dx, 0), min(dx + w, width)
        if b <= a:
            continue
        c[:, a:b] = mov[:, a - dx:b - dx] if mov.ndim == 2 else mov[:, a - dx:b - dx, 0]
        iou = (c & ref).sum() / max((c | ref).sum(), 1)
        if iou > best:
            best, best_dx = iou, dx
    return best_dx, best


def _place(mov: np.ndarray, dx: int, width: int) -> np.ndarray:
    h, w = mov.shape[:2]
    out = np.zeros((h, width) + mov.shape[2:], mov.dtype)
    if mov.dtype != bool:  # extend colours sideways so nothing samples black
        out[:] = mov[:, [0]] if dx > 0 else mov[:, [-1]]
    a, b = max(dx, 0), min(dx + w, width)
    out[:, a:b] = mov[:, a - dx:b - dx]
    return out


def build(views: dict[str, View], layers: int = 32, appendage: str = "auto",
          plate_thickness: int = 2, round_exp: float = 0.0, log=print) -> VoxelModel:
    f, s, b, t = views["front"], views["side"], views["back"], views["top"]
    Y = layers
    X = max(1, round(f.w * Y / f.h))
    Z = max(1, round(s.w * Y / s.h))
    Xb = max(1, round(b.w * Y / b.h))
    log(f"voxel grid X={X} Y={Y} Z={Z}  (front {f.w}x{f.h}px, side {s.w}x{s.h}px, "
        f"back {b.w}x{b.h}px, top {t.w}x{t.h}px)")
    log(f"  top view forced to {X}x{Z}: x-scale {X / t.w * f.h / Y:.3f}, "
        f"z-scale {Z / t.h * s.h / Y:.3f} relative to front/side")

    Fm, Fc = resize_view(f, X, Y)
    Sm, Sc = resize_view(s, Z, Y)
    Bm, Bc = resize_view(b, Xb, Y, mirror=True)
    Tm, Tc = resize_view(t, X, Z)
    dx, iou = _best_shift(Fm, Bm, X)
    log(f"  back view aligned to front: dx={dx}, IoU={iou:.3f}")
    Bm, Bc = _place(Bm, dx, X), _place(Bc, dx, X)

    # [row from top] -> [y from bottom]; top-view rows: image bottom = front = z 0
    F, FC = Fm[::-1], Fc[::-1]
    B, BC = Bm[::-1], Bc[::-1]
    S, SC = Sm[::-1], Sc[::-1]
    T, TC = Tm[::-1], Tc[::-1]

    def hull(f2, b2, s2, t2):
        return (f2 & b2).T[:, :, None] & s2[None, :, :] & t2.T[:, None, :]

    occ = hull(F, B, S, T)
    log(f"  visual hull: {int(occ.sum())} voxels")

    has_part = all(v.part is not None and v.part.any() for v in (f, s, b, t))
    part_px = None
    if has_part:
        occ = _part_split_hull(f, s, b, t, X, Y, Z, Xb, dx, occ, hull, log)
        part_px = {"front": resize_mask(f.part, X, Y)[::-1],
                   "back": _place(resize_mask(b.part, Xb, Y, True), dx, X)[::-1],
                   "side": resize_mask(s.part, Z, Y)[::-1],
                   "top": resize_mask(t.part, X, Z)[::-1]}

    neck_y = None
    if appendage != "off":
        frac = detect_neck(f) if appendage == "auto" else float(appendage)
        neck_y = Y - int(round(frac * Y))  # first y that belongs to the appendage
        body_top = occ[:, :neck_y].copy()
        if round_exp > 0:
            body_top = _round_layers(body_top, round_exp)
        occ[:] = False
        occ[:, :neck_y] = body_top
        # Rebuild the appendage as a thin plate: x/y from FRONT silhouette, z follows
        # the centre line of the SIDE silhouette at each height.
        z0_prev = None
        for y in range(neck_y, Y):
            zs = np.flatnonzero(S[y])
            if len(zs) == 0 and z0_prev is None:
                continue
            z0 = int(round(zs.mean() - (plate_thickness - 1) / 2)) if len(zs) else z0_prev
            # span from the previous row's plate so a sloped plate has no see-through steps
            lo = z0 if z0_prev is None else min(z0, z0_prev)
            hi = z0 + plate_thickness if z0_prev is None else max(z0, z0_prev) + plate_thickness
            z0_prev = z0
            xs = np.flatnonzero(F[y] | B[y])
            occ[xs[:, None], y, np.clip(np.arange(lo, hi), 0, Z - 1)[None]] = True
        log(f"  appendage above y={neck_y} (neck at {frac:.3f} of height) rebuilt as "
            f"{plate_thickness}-voxel plate")
    elif round_exp > 0:
        occ = _round_layers(occ, round_exp)

    no_top = np.zeros(occ.shape, bool)
    if neck_y is not None:
        no_top[:, neck_y:] = True  # thin plate: its edges must not take the top-view colour
    rgb = colour(occ, {"front": FC, "back": BC, "side": SC, "top": TC}, log, no_top=no_top,
                 part_px=part_px)
    return VoxelModel(occ, rgb, {"X": X, "Y": Y, "Z": Z, "neck_y": neck_y, "back_dx": dx,
                                 "back_iou": round(float(iou), 4)})


def _part_split_hull(f, s, b, t, X, Y, Z, Xb, dx, full, hull, log):
    """Remove visual-hull ghosts created by a separable part (e.g. a held object).

    1. Voxels projecting onto a part pixel that sticks out of the body outline in
       some view must belong to the part ("must_part").
    2. A must_part voxel survives each view only if it projects onto part pixels
       there, or the rest of the model (body) hides it from that view.
    The body itself is never carved, so no see-through tunnels appear.
    """
    def maps(fn):
        return (fn(f, X, Y)[::-1], _place(fn(b, Xb, Y, True), dx, X)[::-1],
                fn(s, Z, Y)[::-1], fn(t, X, Z)[::-1])

    PF, PB, PS, PT = maps(lambda v, w, h, m=False: resize_mask(v.part, w, h, m))
    QF, QB, QS, QT = maps(lambda v, w, h, m=False: resize_mask(protrusion_mask(v), w, h, m))
    x, y, z = np.indices(full.shape)
    must = full & (QF[y, x] | QB[y, x] | QS[y, z] | QT[z, x])
    body = full & ~must
    cz = np.cumsum(body, 2)
    cx = np.cumsum(body, 0)
    cy = np.cumsum(body[:, ::-1], 1)[:, ::-1]
    keep = must.copy()
    keep &= PF[y, x] | (cz > 0)                       # front: body at smaller z
    keep &= PB[y, x] | ((cz[:, :, -1:] - cz) > 0)     # back: body at larger z
    keep &= PS[y, z] | (cx > 0)                       # side (viewer at -X)
    keep &= PT[z, x] | (cy > 0)                       # top: body above
    log(f"  part split: {int(must.sum())} voxels must be part, kept {int(keep.sum())}, "
        f"removed {int((must & ~keep).sum())} ghost voxels")
    return body | keep


def _round_layers(occ: np.ndarray, p: float) -> np.ndarray:
    """Carve every horizontal layer to the superellipse inscribed in its bbox."""
    out = occ.copy()
    X, _, Z = occ.shape
    gx, gz = np.meshgrid(np.arange(X) + 0.5, np.arange(Z) + 0.5, indexing="ij")
    for y in range(occ.shape[1]):
        xs, zs = np.nonzero(occ[:, y])
        if len(xs) == 0:
            continue
        cx, cz = (xs.min() + xs.max() + 1) / 2, (zs.min() + zs.max() + 1) / 2
        rx, rz = (xs.max() + 1 - xs.min()) / 2, (zs.max() + 1 - zs.min()) / 2
        inside = np.abs((gx - cx) / rx) ** p + np.abs((gz - cz) / rz) ** p <= 1.0
        out[:, y] &= inside
    return out


def colour(occ: np.ndarray, img: dict[str, np.ndarray], log=print, no_top=None,
           part_px=None) -> np.ndarray:
    """Project each surface voxel's colour from the best visible view.

    With `part_px` (per-view part pixel masks) a voxel counts as part when it
    projects onto part pixels in >= 2 views; a view may only paint part pixels
    onto part voxels and body pixels onto body voxels (stops e.g. the side view
    painting the held object onto the far side of the body).
    """
    X, Y, Z = occ.shape
    n = normals26(occ)
    surf = surface(occ)
    x, y, z = np.indices(occ.shape)

    def first(axis, rev=False):
        a = np.flip(occ, axis) if rev else occ
        idx = np.argmax(a, axis=axis)
        return (occ.shape[axis] - 1 - idx) if rev else idx

    tol = 1
    fz, bz = first(2), first(2, True)
    lx, rx = first(0), first(0, True)
    ty = first(1, True)
    vis = {
        "front": (z <= fz[:, :, None] + tol),
        "back": (z >= bz[:, :, None] - tol),
        "sideL": (x <= lx[None] + tol),
        "sideR": (x >= rx[None] - tol),
        "top": (y >= ty[:, None, :] - tol) & (True if no_top is None else ~no_top),
    }
    dirs = {"front": (0, 0, -1), "back": (0, 0, 1), "sideL": (-1, 0, 0), "sideR": (1, 0, 0),
            "top": (0, 1, 0)}
    weight = {"front": 1.0, "back": 1.0, "sideL": 1.0, "sideR": 0.85, "top": 1.0}
    samples = {
        "front": img["front"][y, x],
        "back": img["back"][y, x],
        "sideL": img["side"][y, z],
        "sideR": img["side"][y, z],   # no right-side view: reuse the left one
        "top": img["top"][z, x],
    }
    label_ok = {k: True for k in dirs}
    if part_px is not None:
        hits = {"front": part_px["front"][y, x], "back": part_px["back"][y, x],
                "sideL": part_px["side"][y, z], "sideR": part_px["side"][y, z],
                "top": part_px["top"][z, x]}
        is_part = (hits["front"].astype(int) + hits["back"] + hits["sideL"] + hits["top"]) >= 2
        label_ok = {k: hits[k] == is_part for k in dirs}
    best = np.full(occ.shape, 0.05, np.float32)
    out = np.zeros(occ.shape + (3,), np.float32)
    got = np.zeros(occ.shape, bool)
    for k, d in dirs.items():
        sc = (n @ np.array(d, np.float32)) * weight[k]
        sc = np.where(vis[k] & surf & label_ok[k], sc, -1)
        upd = sc > best
        best[upd] = sc[upd]
        out[upd] = samples[k][upd]
        got |= upd
    missing = occ & ~got
    log(f"  coloured {int(got.sum())} surface voxels, {int((surf & ~got).sum())} surface "
        f"voxels unseen by every view -> filled from neighbours")
    # fill unseen surface voxels and the interior with the nearest coloured voxel
    if missing.any() and got.any():
        _, idx = nd.distance_transform_edt(~got, return_indices=True)
        out[missing] = out[idx[0], idx[1], idx[2]][missing]
    return np.clip(out, 0, 255).astype(np.uint8)


def quantize(model: VoxelModel, colors: int) -> None:
    from PIL import Image
    vals = model.rgb[model.occ]
    pal_img = Image.fromarray(vals[None]).quantize(colors, method=Image.Quantize.MEDIANCUT)
    model.rgb[model.occ] = np.asarray(pal_img.convert("RGB"))[0]
