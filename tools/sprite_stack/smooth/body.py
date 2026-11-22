"""Body: 64 smooth horizontal cross-sections with baked soft cel shading.

Shape: at normalised height h the cross-section is a four-radius superellipse
    |dx / r_x|^p + |dz / r_z|^p = 1
with r_x = right(h) or left(h) and r_z = front(h) or back(h) depending on the
quadrant (Profile from profile.py). The exponent p is estimated from the top
view. The implicit field F(x, h, z) = f - 1 is evaluated analytically, so
outlines are anti-aliased and normals are continuous across layers (no colour
jumps between layers).

Layer image convention (same as the rest of tools/sprite_stack):
  top-down view, +X (character's left) to the right, character FRONT at the
  BOTTOM edge, canvas centre = body axis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as nd

from ..views import View
from .profile import Profile


@dataclass
class BodyParams:
    layers: int = 64
    size: int = 512
    body_width_px: float = 300.0       # widest body extent on the canvas
    p: float = 2.2                     # cross-section superellipse exponent
    base_round: float = 0.05           # rounded ground edge (fraction of height)
    light: tuple = (0.0, 1.0, 0.0)      # straight down: shading is the same from every yaw
    gloss_dir: tuple = (-0.45, 0.62, -0.64)


def flare_bottom(fn, h_round):
    """Keep the widest radius below the widest point (body flares to the ground)
    and round the very bottom edge."""
    hs = np.linspace(0, 1, 801)
    r = fn(hs)
    i_max = int(np.argmax(r[: len(hs) // 2]))
    def g(h):
        h = np.asarray(h, float)
        base = np.where(h < hs[i_max], r[i_max], fn(h))
        t = np.clip(h / h_round, 0, 1)
        return base * np.sqrt(1 - (1 - t) ** 2) ** 0.5 if h_round > 0 else base
    return g


def estimate_exponent(top: View) -> float:
    """Superellipse exponent from the top-view body outline (area / bbox area)."""
    m = top.mask & ~(top.part if top.part is not None else False)
    r = int(0.12 * min(top.h, top.w))
    disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
    m = nd.binary_opening(nd.binary_fill_holes(m), structure=disk)
    ys, xs = np.nonzero(m)
    ratio = m.sum() / ((np.ptp(xs) + 1) * (np.ptp(ys) + 1))
    best, best_p = 9, 2.0
    for p in np.linspace(1.6, 4.0, 49):
        a = math.gamma(1 + 1 / p) ** 2 / math.gamma(1 + 2 / p)
        if abs(a - ratio) < best:
            best, best_p = abs(a - ratio), p
    # bumps (arms, feet) still left in the top outline push the ratio down; a body
    # cross-section is never pinched, so keep it between an ellipse and a soft square
    return float(np.clip(best_p, 2.0, 3.0))


class Body:
    def __init__(self, prof: Profile, prm: BodyParams):
        self.prm = prm
        self.prof = prof
        hr = prm.base_round
        if getattr(prof, "shaped", False):     # v2 profile already shapes the bottom
            self.r = {k: getattr(prof, k) for k in ("left", "right", "front", "back")}
        else:
            self.r = {k: flare_bottom(getattr(prof, k), hr) for k in ("left", "right", "front", "back")}
        widest = max(float(np.max(self.r["left"](np.linspace(0, 1, 400)) + self.r["right"](np.linspace(0, 1, 400)))),
                     float(np.max(self.r["front"](np.linspace(0, 1, 400)) + self.r["back"](np.linspace(0, 1, 400)))))
        self.scale = prm.body_width_px / widest          # canvas px per front-view px
        self.height_px = prof.height * self.scale        # body height on the canvas scale
        self.spacing = self.height_px / prm.layers       # vertical px between layers
        n = prm.size
        c = (n - 1) / 2
        u, v = np.meshgrid(np.arange(n) - c, np.arange(n) - c)
        self.dx = u                 # +X to the right
        self.dz = v                 # +v (down in image) = towards the FRONT
        self.max_extent = {}

    def radii(self, h):
        s = self.scale
        return {k: max(float(fn(h)) * s, 1e-3) for k, fn in self.r.items()}

    def field(self, h):
        """f(x, z) for height h: < 1 inside. Returns f."""
        R = self.radii(h)
        p = self.prm.p
        if min(R.values()) < 1.0:            # closed (above the head top): empty layer
            return np.full(self.dx.shape, 1e6)
        rx = np.where(self.dx >= 0, R["right"], R["left"])
        rz = np.where(self.dz >= 0, R["front"], R["back"])
        return (np.abs(self.dx) / rx) ** p + (np.abs(self.dz) / rz) ** p

    def layer(self, i: int, palette: dict) -> tuple[np.ndarray, np.ndarray]:
        """Returns (RGBA uint8 image, normals) for layer i."""
        L = self.prm.layers
        h = (i + 0.5) / L
        dh = 0.5 / L
        f = self.field(h)
        f_up, f_dn = self.field(min(h + dh, 1.0)), self.field(max(h - dh, 0.0))
        g_v, g_u = np.gradient(f)                          # per canvas px
        g_y = (f_up - f_dn) / (2 * dh * self.height_px)    # per canvas px in height
        # anti-aliased coverage from the signed distance (f - 1) / |grad f|
        if f.min() >= 1e6:
            return np.zeros(f.shape + (4,), np.uint8), None
        gl = np.hypot(g_u, g_v) + 1e-9
        alpha = np.clip(0.5 - (f - 1) / gl, 0, 1)
        # normal of the level set through each pixel; x = right, y = up, z = towards the BACK
        n = np.stack([g_u, g_y, -g_v], -1)   # outward = +grad f
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        rgb = shade(n, h, palette, self.prm)
        img = np.zeros(f.shape + (4,), np.uint8)
        img[..., :3] = rgb
        img[..., 3] = np.round(alpha * 255).astype(np.uint8)
        img[img[..., 3] == 0, :3] = 0                      # no colour left in transparent pixels
        return img, n


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def shade(n, h, pal, prm: BodyParams):
    """Soft cel shading: 3 tones blended with narrow smoothsteps + gloss + ground AO."""
    L = np.array(prm.light, np.float32)
    L /= np.linalg.norm(L)
    d = n @ L
    shadow, base, light = (np.array(pal[k], np.float32) for k in ("shadow", "base", "light"))
    col = shadow + (base - shadow) * smoothstep(-0.30, -0.18, d)[..., None]   # underside / flare
    col = col + (light - col) * smoothstep(0.50, 0.62, d)[..., None]            # head dome
    # glossy highlight (the white glint on the upper-left of the head in the art)
    G = np.array(prm.gloss_dir, np.float32)
    G /= np.linalg.norm(G)
    g = smoothstep(0.90, 0.965, n @ G)[..., None]
    g = g * smoothstep(0.58, 0.68, np.float32(h))          # the glint sits on the head only
    col = col + (np.array(pal["gloss"], np.float32) - col) * g * 0.7
    # contact shadow near the ground
    ao = 1 - 0.18 * (1 - smoothstep(0.0, 0.12, np.float32(h)))
    return np.clip(col * ao, 0, 255).astype(np.uint8)


def sample_palette(views: dict[str, View]) -> dict:
    """Mint tones from the art: light / base / shadow from the back view body."""
    v = views["back"]
    m = v.mask & ~(v.part if v.part is not None else False)
    m = nd.binary_erosion(m, iterations=6)
    px = v.rgb[m]
    lum = px @ np.array([0.299, 0.587, 0.114])
    px = px[lum > np.percentile(lum, 15)]              # drop line art
    lum = px @ np.array([0.299, 0.587, 0.114])
    q = np.percentile(lum, [20, 55, 92])
    pick = lambda lo, hi: px[(lum >= lo) & (lum <= hi)].mean(0)  # noqa: E731
    shadow = pick(q[0] - 6, q[0] + 6)
    base = pick(q[1] - 6, q[1] + 6)
    light = pick(q[2] - 6, q[2] + 6)
    return {"shadow": shadow.tolist(), "base": base.tolist(), "light": light.tolist(),
            "gloss": [246.0, 252.0, 248.0], "outline": (shadow * 0.45).tolist()}
