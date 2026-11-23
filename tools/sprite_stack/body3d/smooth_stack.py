"""Continuous sprite stack: interpolate BETWEEN neighbouring layers.

The plain stack draws every layer's single cross-section (taken at the slab
centre) several times across its slab, so the outline and the normal jump at
every slab boundary; on curved parts of the body that reads as horizontal
stripes. Here every drawn copy sits at a height h between two layer centres
(j + 0.5 and j + 1.5) and blends those two layers:

  * outline : signed distance to the edge, stored per layer (Body/SDF), is
              interpolated and thresholded -> the copy's outline is the actual
              cross-section in between (exact for a linear change, no blur)
  * normal  : the two layers' normals are interpolated (a normal missing on one
              side falls back to the other)
  * colour  : the albedo of whichever layer covers the pixel

The existing Albedo / Normal / baked layers are unchanged; SDF layers are new.
SDF encoding: 8-bit grey, value = 128 + 16 * d (d = canvas px inside the edge,
clamped to +-8); coverage = clamp(0.5 + d, 0, 1).
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image
from scipy import ndimage as nd

from .model import BodyShape
from .stack import shade_top

SDF_SCALE = 16.0
SDF_RANGE = 7.9


def make_sdf(shape: BodyShape, layers=64, size=512, scale=1.2):
    n = size
    c = (n - 1) / 2
    u, v = np.meshgrid(np.arange(n) - c, np.arange(n) - c)
    X = u / scale
    Z = -v / scale
    dh = (shape.top - shape.bottom) / layers
    out = []
    for i in range(layers):
        h = shape.bottom + (i + 0.5) * dh
        f = shape.F(X, np.full_like(X, h), Z) + 1.0
        gv, gu = np.gradient(f)
        gl = np.hypot(gu, gv) + 1e-9
        d = np.clip((1 - f) / gl, -SDF_RANGE, SDF_RANGE)          # canvas px, > 0 inside
        out.append(np.clip(np.round(128 + SDF_SCALE * d), 0, 255).astype(np.uint8))
    return out


def decode(sdf8):
    return (sdf8.astype(np.float32) - 128) / SDF_SCALE


def export_sdf(shape, body_dir, layers=64, size=512, scale=1.2):
    d = os.path.join(body_dir, "SDF")
    os.makedirs(d, exist_ok=True)
    out = make_sdf(shape, layers, size, scale)
    for i, s in enumerate(out):
        Image.fromarray(s, "L").save(os.path.join(d, f"Layer_{i:03d}.png"))
    return out


# ------------------------------------------------------------------ rendering

def _f(a):
    return np.ascontiguousarray(a, np.float32)


def _prep(albedo, normal, sdf8, yaw, S, sq):
    """Rotate + squash one layer's channels once."""
    def tr(ch, resample=Image.BILINEAR):
        im = Image.fromarray(_f(ch), "F").resize((S, S), resample).rotate(yaw, resample=resample)
        return np.asarray(im.resize((S, sq), resample))
    a = albedo[..., 3] / 255.0
    rgb = np.stack([tr(albedo[..., k] / 255.0 * a) for k in range(3)], -1)
    aa = tr(a)
    rgb = np.divide(rgb, aa[..., None], out=np.zeros_like(rgb), where=aa[..., None] > 1e-3)
    nv = [tr(normal[..., k] / 255.0 * 2 - 1) for k in range(3)]
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    N = np.stack([nv[0] * c - nv[1] * s, nv[0] * s + nv[1] * c, nv[2]], -1)
    ln = np.linalg.norm(N, axis=-1, keepdims=True)
    N = np.divide(N, ln, out=np.zeros_like(N), where=ln > 0.5)    # 0 where no normal
    d = tr(decode(sdf8)) * (S / albedo.shape[0])                   # px at output scale
    return {"rgb": rgb, "a": aa, "N": N, "d": d}


def smooth_frame(albedo, normal, sdf, spacing, yaw, pal, squash=0.75, out_scale=1.0, copies=8,
                 mode="lit", shade_fn=None, outline=(38, 107, 81), outline_px=2):
    """mode 'lit' = screen-fixed light from the normals (lit.shade);
    'baked' = the rotation-invariant top light (stack.shade_top) on the same normals."""
    from . import lit
    n0 = albedo[0].shape[0]
    S = int(round(n0 * out_scale))
    sq = max(1, int(round(S * squash)))
    L = len(albedo)
    lift = spacing * out_scale
    Hc = int(math.ceil(sq + lift * L)) + 12
    canvas = np.zeros((Hc, S, 4), np.float32)
    base_y = Hc - 6 - sq
    P = [_prep(albedo[i], normal[i], sdf[i], yaw, S, sq) for i in range(L)]
    c, s = math.cos(math.radians(-yaw)), math.sin(math.radians(-yaw))
    for k in range(L * copies):
        h = k / copies                                  # height in layer units (copy bottom)
        x = h + 0.5 / copies - 0.5                      # position between layer centres
        j = int(math.floor(x))
        t = x - j
        j0, j1 = max(0, min(L - 1, j)), max(0, min(L - 1, j + 1))
        if j < 0:
            t = 0.0
        if j + 1 > L - 1:
            t = 0.0
        A, B = P[j0], P[j1]
        d = (1 - t) * A["d"] + t * B["d"]
        cov = np.clip(0.5 + d, 0, 1)
        if cov.max() <= 0:
            continue
        hasA = np.linalg.norm(A["N"], axis=-1, keepdims=True) > 0.5
        hasB = np.linalg.norm(B["N"], axis=-1, keepdims=True) > 0.5
        nA = np.where(hasA, A["N"], B["N"])
        nB = np.where(hasB, B["N"], A["N"])
        N = (1 - t) * nA + t * nB
        N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
        rgb = np.where((A["a"] > 0.5)[..., None], A["rgb"], B["rgb"])
        h01 = (j0 + t * (j1 - j0) + 0.5) / L
        if mode == "lit":
            tone = lit.shade(N, h01, pal, squash)
        else:
            # back to model space (x, up, z) for the top-light shading
            nl = N.copy()
            nx = nl[..., 0] * c - nl[..., 1] * s
            ny = nl[..., 0] * s + nl[..., 1] * c
            nm = np.stack([nx, nl[..., 2], ny], -1)
            tone = shade_top(nm, h01, pal) / np.maximum(np.asarray(pal["base"], np.float32), 1)
        col = np.clip(rgb * tone, 0, 1)
        layer = np.concatenate([col * cov[..., None], cov[..., None]], -1)
        y = int(round(base_y - h * lift))
        y0, y1 = max(y, 0), min(y + sq, Hc)
        if y1 > y0:
            src = layer[y0 - y:y1 - y]
            canvas[y0:y1] = src + canvas[y0:y1] * (1 - src[..., 3:4])
    rgbo = np.divide(canvas[..., :3], canvas[..., 3:4], out=np.zeros_like(canvas[..., :3]),
                     where=canvas[..., 3:4] > 1e-4)
    img = np.round(np.clip(np.concatenate([rgbo, canvas[..., 3:4]], -1), 0, 1) * 255).astype(np.uint8)
    if outline is not None:
        m = nd.binary_fill_holes(img[..., 3] > 110)
        ring = nd.binary_dilation(m, iterations=outline_px) & ~m
        img[ring] = list(outline) + [255]
    return Image.fromarray(img)


def band_energy(frame: Image.Image, spacing_px: float):
    """Stripe strength: brightness changes along vertical lines through the body at the
    layer frequency (period = one layer on screen), relative to the smooth shading.
    Returns the mean amplitude (0..255 levels) of the high-pass signal."""
    a = np.asarray(frame).astype(np.float32)
    m = nd.binary_erosion(a[..., 3] > 250, iterations=4)
    lum = a[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    smooth = nd.gaussian_filter1d(lum, max(1.0, spacing_px), axis=0)
    hp = (lum - smooth)[m]
    return {"rms": float(np.sqrt(np.mean(hp ** 2))), "p99": float(np.percentile(np.abs(hp), 99))}
