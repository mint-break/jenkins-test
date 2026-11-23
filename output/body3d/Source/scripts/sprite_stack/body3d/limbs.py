"""Arms and feet: smooth solids measured from the A-pose sheet, sliced on the
body's layer grid (feet reach below the body, so their indices go negative).

Arms (A-pose, hanging at the sides; right arm = +X, left mirrored):
  front view : the arm is the outline outside the body plus the inner line
               -> centre x(h) and half width rx(h) for h = 104..168
  side view  : the open U line drawn on the body -> front/back edge
               -> centre z(h) and half depth rz(h) for h = 90..162
  At every height the arm is an ellipse (centre x, z; radii rx, rz); the
  profiles are smooth fits through the measured rows, the tip is a rounded
  cap and the upper end runs into the shoulder inside the body.
Feet: an ellipsoid each, flat on the ground, measured from the two blobs
under the body in the front view and the side silhouette below h = 20.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage as nd

from .stack import shade_top, smoothstep


def _poly(xs, ys, deg):
    c = np.polyfit(xs, ys, deg)
    return lambda h: np.polyval(c, h)


@dataclass
class ArmParams:
    # measured (sheet px, h above ground): front view centre / half width
    front_h: tuple = (112, 120, 128, 136, 144, 152, 160, 168)
    front_c: tuple = (162.0, 162.5, 161.5, 160.0, 158.5, 156.0, 152.5, 148.5)
    front_r: tuple = (20.0, 23.5, 25.5, 26.0, 25.5, 25.0, 23.5, 22.5)
    # side view: front / back edge of the U line
    side_h: tuple = (98, 106, 114, 122)
    side_front: tuple = (-63.5, -66.5, -67.5, -66.5)
    side_back: tuple = (-3.5, 2.5, 6.5, 8.5)
    tip_h: float = 92.0          # lowest point of the arm (front 98, side 86)
    tip_round: float = 20.0
    top_h: float = 215.0         # upper end, inside the body (hidden)
    top_round: float = 14.0
    merge_front: float = 118.0   # height where the arm's front edge starts sinking into the body
    merge_back: float = 158.0    # ... and the back edge (side view: 125 / 162)
    merge_len: float = 45.0
    sink: float = 34.0           # how far the arm centre moves into the body


@dataclass
class FootParams:
    centre: tuple = (80.0, 16.0, -22.0)   # right foot; left mirrored
    rx: float = 50.0
    rz: float = 60.5
    ry_down: float = 17.0
    ry_up: float = 32.0
    bottom_exp: float = 3.0      # >2 = flatter sole (front/side rows near the ground: 0.41 / 0.8 / 0.95)


@dataclass
class TailParams:
    # side view: protrudes from the back h=33..100, back edge up to z=175 at h~80;
    # back view: a round outline 52 px wide around h=37..85 on the centre line
    centre: tuple = (0.0, 67.0, 148.0)
    rx: float = 26.0
    ry: float = 33.0
    rz: float = 27.0
    tilt_deg: float = -20.0       # top leans back, as the side view's teardrop


class Limbs:
    """Implicit solids. F < 0 inside (approximately a distance in sheet px)."""

    def __init__(self, arm: ArmParams | None = None, foot: FootParams | None = None,
                 body=None, fillet: float = 12.0, tail: TailParams | None = None):
        self.tail = tail or TailParams()
        self.body = body                   # BodyShape: arms/feet blend into it with a fillet
        self.fillet = fillet
        self.arm = arm or ArmParams()
        self.foot = foot or FootParams()
        a = self.arm
        self.xc = _poly(a.front_h, a.front_c, 2)
        self.rx = _poly(a.front_h, a.front_r, 2)
        zc = [(f + b) / 2 for f, b in zip(a.side_front, a.side_back)]
        rz = [(b - f) / 2 for f, b in zip(a.side_front, a.side_back)]
        self._zc = (np.array(a.side_h, float), np.array(zc))
        self._rz = (np.array(a.side_h, float), np.array(rz))

    # ---- arm profiles (clamped extrapolation outside the measured rows) ----
    def arm_profiles(self, h):
        a = self.arm
        h = np.asarray(h, float)
        hc = np.clip(h, a.front_h[0] - 8, a.front_h[-1] + 40)
        xc = self.xc(hc)
        rx = np.clip(self.rx(np.clip(h, a.front_h[0], a.front_h[-1])), 18, None)
        zc = np.interp(h, *self._zc)
        rz = np.interp(h, *self._rz)
        # rounded ends
        lo = a.tip_h + a.tip_round
        s_lo = np.sqrt(np.clip(1 - ((lo - h) / a.tip_round) ** 2, 0, 1))
        s = np.where(h < lo, s_lo, 1.0)
        hi = a.top_h - a.top_round
        s_hi = np.sqrt(np.clip(1 - ((h - hi) / a.top_round) ** 2, 0, 1))
        s = np.where(h > hi, s_hi, s)
        s = np.where((h < a.tip_h) | (h > a.top_h), 0.0, s)
        return xc, zc, rx * s, rz * s

    def F_arm(self, X, Y, Z, side=+1):
        a = self.arm
        xc, zc, rx, rz = self.arm_profiles(Y)
        # shoulder: the arm sinks into the body along a SLANTED line, as in the side
        # view (the front edge merges at h~125, the back edge at h~165)
        zf, zb = a.side_front[-1], a.side_back[-1]
        u = np.clip((Z - zf) / (zb - zf), 0, 1)
        h0 = a.merge_front + (a.merge_back - a.merge_front) * u
        t = smoothstep(h0, h0 + a.merge_len, Y)
        xc = xc - a.sink * t
        rx = rx * (1 - 0.35 * t)
        rx = np.maximum(rx, 1e-3)
        rz = np.maximum(rz, 1e-3)
        q = np.sqrt(((side * X - xc) / rx) ** 2 + ((Z - zc) / rz) ** 2)
        return (q - 1) * np.minimum(rx, rz) + np.where(rx < 0.01, 1e3, 0)

    def F_foot(self, X, Y, Z, side=+1):
        f = self.foot
        cx, cy, cz = f.centre
        ry = np.where(Y < cy, f.ry_down, f.ry_up)
        ex = np.where(Y < cy, f.bottom_exp, 2.0)
        r_h = np.sqrt(((side * X - cx) / f.rx) ** 2 + ((Z - cz) / f.rz) ** 2)
        lim = np.sqrt(np.clip(1 - np.abs((Y - cy) / ry) ** ex, 0, 1))
        d = (r_h - lim) * min(f.rx, f.rz)
        d = np.where(np.abs((Y - cy) / ry) > 1, np.maximum(d, (np.abs(Y - cy) - ry)), d)
        return np.where(Y < 0, 1e3, d)                      # flat on the ground

    def body_dist(self, X, Y, Z):
        """Approximate signed distance to the body (sheet px)."""
        b = self.body
        f = b.F(X, Y, Z) + 1.0
        r = np.minimum(b.w(Y), np.where(Z < 0, b.df(Y), b.db(Y)))
        d = (np.power(np.maximum(f, 1e-9), 1.0 / b.p) - 1) * np.maximum(r, 1.0)
        return np.where((Y < b.bottom) | (Y > b.top), 1e3, d)

    def F_tail(self, X, Y, Z):
        t = self.tail
        cx, cy, cz = t.centre
        a = math.radians(t.tilt_deg)
        dy, dz = Y - cy, Z - cz
        yr = dy * math.cos(a) - dz * math.sin(a)          # rotate about x: top leans back
        zr = dy * math.sin(a) + dz * math.cos(a)
        q = np.sqrt(((X - cx) / t.rx) ** 2 + (yr / t.ry) ** 2 + (zr / t.rz) ** 2)
        return (q - 1) * min(t.rx, t.ry, t.rz)

    def F(self, X, Y, Z, which):
        if which == "Arms":
            fl = np.minimum(self.F_arm(X, Y, Z, +1), self.F_arm(X, Y, Z, -1))
        elif which == "Tail":
            fl = self.F_tail(X, Y, Z)
        else:
            fl = np.minimum(self.F_foot(X, Y, Z, +1), self.F_foot(X, Y, Z, -1))
        if self.body is None or self.fillet <= 0:
            return fl
        # smooth union with the body, keeping ONLY the fillet volume: outside the body
        # (deeper than 1 px inside is excluded) and only where the blend is active
        # (fl < fb + k). The part thus adds the rounded joint and nothing of the body.
        k = self.fillet
        fb = self.body_dist(X, Y, Z)
        hh = np.clip(0.5 + 0.5 * (fb - fl) / k, 0, 1)
        sm = fb * (1 - hh) + fl * hh - k * hh * (1 - hh)
        fillet = np.maximum(np.maximum(sm, -fb - 1.0), fl - fb - k)
        return np.minimum(fl, fillet)

    def grad(self, X, Y, Z, which, e=0.5):
        g = [(self.F(X + e * (k == 0), Y + e * (k == 1), Z + e * (k == 2), which)
              - self.F(X - e * (k == 0), Y - e * (k == 1), Z - e * (k == 2), which)) / (2 * e)
             for k in range(3)]
        n = np.stack(g, -1)
        return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)

    def y_range(self, which):
        if which == "Arms":
            return self.arm.tip_h, self.arm.top_h
        if which == "Tail":
            t = self.tail
            r = max(t.ry, t.rz)
            return t.centre[1] - r - 2, t.centre[1] + r + 2
        f = self.foot
        return 0.0, f.centre[1] + f.ry_up


def slice_part(limbs: Limbs, which, body_shape, bottom, dh, scale, size, pal, crease=0.05):
    """Layers on the body grid covering the part. Returns indices, albedo, normal, baked."""
    n = size
    c = (n - 1) / 2
    u, v = np.meshgrid(np.arange(n) - c, np.arange(n) - c)
    X = u / scale
    Z = -v / scale
    y0, y1 = limbs.y_range(which)
    i0 = int(math.floor((y0 - bottom) / dh))
    i1 = int(math.floor((y1 - bottom) / dh))
    # height of the part's upper surface above every pixel (for pixels seen from above)
    htop = np.full(X.shape, -1.0)
    for hh in np.linspace(y1, y0, 400):
        inside = (limbs.F(X, np.full_like(X, hh), Z, which) <= 0) & (htop < 0)
        htop[inside] = hh
    has_top = htop >= 0
    n_top = np.zeros(X.shape + (3,))
    n_top[has_top] = limbs.grad(X[has_top], htop[has_top], Z[has_top], which)
    H_body = body_shape.top - body_shape.bottom
    out = []
    for i in range(i0, i1 + 1):
        h = bottom + (i + 0.5) * dh
        Y = np.full_like(X, h)
        F = limbs.F(X, Y, Z, which) * scale                 # canvas px, < 0 inside
        alpha = np.clip(0.5 - F, 0, 1)
        A = np.zeros((n, n, 4), np.uint8)
        Nm = np.zeros((n, n, 4), np.uint8)
        Bk = np.zeros((n, n, 4), np.uint8)
        m = alpha > 0
        if m.any():
            n_rim = limbs.grad(X[m], Y[m], Z[m], which)
            d_rim = np.clip(-F[m], 0, None)
            d_top = np.clip(htop[m] - h, 0, None) * scale
            w = smoothstep(-3.0, 3.0, d_rim - d_top)[:, None]
            nrm = n_rim * (1 - w) + n_top[m] * w
            # where the limb surface runs into the body surface, use the body's normal so the
            # two coincident surfaces shade identically (no stripes in the merge zone)
            if limbs.body is not None:
                fbm = limbs.body_dist(X[m], Y[m], Z[m])
                wb = smoothstep(8.0, 0.0, np.abs(fbm))[:, None]
                nb = body_shape.grad(X[m], Y[m], Z[m])
                nrm = nrm * (1 - wb) + nb * wb
            nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
            hn = np.clip((h - body_shape.bottom) / H_body, 0, 1)
            base = np.asarray(pal["base"], np.float32) * (0.965 + 0.05 * hn)
            # soft shade in the fillet where the limb meets the body
            fb = limbs.body_dist(X[m], Y[m], Z[m]) if limbs.body is not None else np.full(m.sum(), 99.0)
            cr = smoothstep(6.0, 0.0, fb) * (fb > -0.5) * crease
            outline = np.asarray(pal["outline"], np.float32)
            col = base[None] * (1 - cr[:, None]) + outline[None] * cr[:, None]
            a8 = np.round(alpha[m] * 255).astype(np.uint8)
            A[m, :3] = np.clip(col, 0, 255).astype(np.uint8)
            A[m, 3] = a8
            ls = np.stack([nrm[:, 0], nrm[:, 2], nrm[:, 1]], -1)
            Nm[m, :3] = np.clip(np.round((ls * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
            Nm[m, 3] = a8
            tone = shade_top(nrm, hn, pal) / np.maximum(np.asarray(pal["base"], np.float32), 1)
            Bk[m, :3] = np.clip(col * tone, 0, 255).astype(np.uint8)
            Bk[m, 3] = a8
            cov = Nm[..., 3] > 0
            dd, (iy, ix) = nd.distance_transform_edt(~cov, return_indices=True)
            pad = (~cov) & (dd <= 4)
            Nm[pad, :3] = Nm[iy[pad], ix[pad], :3]
        out.append((i, A, Nm, Bk))
    return out


def _fname(i):
    return f"Layer_{i:03d}.png" if i >= 0 else f"Layer_n{-i:03d}.png"


def export_limbs(out_dir, body_shape, limbs: Limbs | None = None,
                 parts=(("Arms", "Limbs/Arms"), ("Feet", "Limbs/Feet"))):
    body_cfg = json.load(open(os.path.join(out_dir, "Body", "stack_config.json")))
    sp = body_cfg["layer_spacing_px"]
    scale = body_cfg["scale"]["canvas_px_per_sheet_px"]
    bottom = body_cfg["scale"]["sheet_px_body_bottom_above_ground"]
    size = body_cfg["image"]["width"]
    dh = sp / scale
    pal = {k: np.array(v, float) for k, v in body_cfg["lighting"]["palette_rgb"].items()}
    limbs = limbs or Limbs(body=body_shape)
    cfgs = []
    for which, rel in parts:
        d = os.path.join(out_dir, rel)
        for s in ("", "Albedo", "Normal"):
            os.makedirs(os.path.join(d, s), exist_ok=True)
            for f in os.listdir(os.path.join(d, s)):
                if f.startswith("Layer_") and f.endswith(".png"):
                    os.remove(os.path.join(d, s, f))
        files, idx = [], []
        for i, A, Nm, Bk in slice_part(limbs, which, body_shape, bottom, dh, scale, size, pal):
            if A[..., 3].max() == 0:
                continue
            fn = _fname(i)
            Image.fromarray(A, "RGBA").save(os.path.join(d, "Albedo", fn))
            Image.fromarray(Nm, "RGBA").save(os.path.join(d, "Normal", fn))
            Image.fromarray(Bk, "RGBA").save(os.path.join(d, fn))
            files.append(fn)
            idx.append(i)
        if which == "Tail":
            t = limbs.tail
            geom = {"tail": {"centre": list(t.centre), "rx": t.rx, "ry": t.ry, "rz": t.rz,
                             "tilt_deg": t.tilt_deg}}
            est = ["tail: tilted ellipsoid; width from the back view's round outline (52 px), "
                   "depth and height from the side view (back edge z=175 at h~80, h=33..100)",
                   "the back view's outline (h 37..85) and the side view (h 33..100) disagree on the "
                   "height; the ellipsoid takes the middle (h 34..94)",
                   "the joint to the body is a rounded fillet (smooth union)"]
        elif which == "Arms":
            a = limbs.arm
            geom = {"right_arm": {"front_view": {"h": list(a.front_h), "centre_x": list(a.front_c),
                                                 "half_width": list(a.front_r)},
                                  "side_view": {"h": list(a.side_h), "front_z": list(a.side_front),
                                                "back_z": list(a.side_back)},
                                  "tip_h": a.tip_h, "top_h": a.top_h},
                    "left_arm": "mirror of the right arm in x"}
            est = ["arm cross-sections are ellipses: x from the front view (outer edge + inner line), "
                   "z from the side view's U line; heights between measured rows are smooth fits",
                   "the arm's upper part merges into the body; above h=170 (front) / 125 (side front "
                   "edge) the outline is hidden and the measured shape is extended",
                   "tip at h=92: front view shows ~98, side view ~86 (averaged), rounded"]
        else:
            f = limbs.foot
            geom = {"right_foot": {"centre": list(f.centre), "rx": f.rx, "rz": f.rz,
                                   "ry_down": f.ry_down, "ry_up": f.ry_up, "flat_bottom_at": 0.0},
                    "left_foot": "mirror of the right foot in x"}
            est = ["feet: ellipsoids fitted to the two blobs under the body (front view rows h<20) and "
                   "the side silhouette below h=20; flat on the ground",
                   f"foot top (h={f.centre[1] + f.ry_up:.0f}) is hidden by the body; the side view's line "
                   "over the toe (h~35 at z=-70) is matched"]
        cfg = {
            "name": f"Mint{which}", "part": which, "version": 1,
            "grid": "same canvas, pivot, axes and layer spacing as Body; negative layer indices are "
                    "below the body's first layer (file names Layer_n###)",
            "layers": len(files), "layer_index": idx, "layer_span": 1.0,
            "files": [f"{rel}/{f}" for f in files],
            "albedo_files": [f"{rel}/Albedo/{f}" for f in files],
            "normal_files": [f"{rel}/Normal/{f}" for f in files],
            "layer_height_px": [round(k * sp, 4) for k in idx],
            "layer_spacing_px": sp,
            "image": body_cfg["image"], "pivot": body_cfg["pivot"], "axes": body_cfg["axes"],
            "render": {"two_sided": False, "gloss_scale": 0.0},
            "geometry_sheet_px": geom,
            "crease": "albedo darkened towards the outline colour where the limb meets the body",
            "estimated": est,
            "config_path": f"{rel}/stack_config.json",
        }
        with open(os.path.join(d, "stack_config.json"), "w") as fh:
            json.dump(cfg, fh, indent=2, ensure_ascii=False)
        cfgs.append(cfg)
    return cfgs


def ortho_check(limbs: Limbs, body_shape, views, mF, mS, path):
    """Outline of body + limbs (red) over the front and side art (orthographic)."""
    from PIL import ImageDraw
    tiles = []
    for view, m in (("front", mF), ("side", mS), ("back", None)):
        if m is None:
            from . import analyze
            m = analyze.measure(views["back"], "frontback")
        v = views[view]
        H, W = v.h, v.w
        rows = np.arange(H)
        hgt = (m.ground - rows).astype(float)
        cols = np.arange(W) - m.axis
        U, Hh = np.meshgrid(cols, hgt)
        occ = np.zeros((H, W), bool)
        for t in np.linspace(-200, 200, 161):            # march along the view direction
            if view == "front":
                X, Z = U, np.full_like(U, t)
            elif view == "back":
                X, Z = -U, np.full_like(U, t)
            else:
                X, Z = np.full_like(U, t), U
            inside = (limbs.F(X, Hh, Z, "Arms") <= 0) | (limbs.F(X, Hh, Z, "Feet") <= 0) | \
                     (limbs.F(X, Hh, Z, "Tail") <= 0) | (body_shape.F(X, Hh, Z) <= 0)
            occ |= inside
        im = Image.fromarray(v.rgb.astype(np.uint8)).convert("RGBA")
        over = np.zeros((H, W, 4), np.uint8)
        ring = occ & ~nd.binary_erosion(occ, iterations=2)
        over[ring] = (225, 30, 30, 255)
        im.alpha_composite(Image.fromarray(over))
        tiles.append(im)
    Wt = sum(t.width for t in tiles) + 20
    Ht = max(t.height for t in tiles)
    out = Image.new("RGBA", (Wt, Ht), (255, 255, 255, 255))
    x = 0
    for t in tiles:
        out.alpha_composite(t, (x, Ht - t.height))
        x += t.width + 20
    out.save(path)
