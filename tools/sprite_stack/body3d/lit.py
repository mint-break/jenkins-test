"""Screen-fixed lighting for the sprite stack (normal maps), mirrored 1:1 by
Unity/SpriteStackLit.shader so the Python preview shows what Unity renders.

Space used for lighting = the stack's screen plane before squashing:
  x = screen right, y = screen up inside the ground plane (away from the
  viewer), z = world up. A layer rotated by `yaw` (counter-clockwise on screen)
  has its layer-space normal xy rotated by the same yaw.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage as nd

# shared with the shader defaults (stack_config.json "lit" block)
PARAMS = {
    "light_dir": [-0.45, -0.55, 0.70],   # from the viewer's upper left, towards the character
    "base_from": 0.05, "base_to": 0.22,  # shadow -> base
    "light_from": 0.72, "light_to": 0.84,  # base -> light
    "gloss_from": 0.955, "gloss_to": 0.99, "gloss_strength": 0.6,
    "gloss_min_height": 0.55, "gloss_full_height": 0.65,
    "rim_strength": 0.10,
}


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def view_dir(squash):
    """Camera direction (towards the viewer) for a given vertical squash = sin(pitch)."""
    pitch = math.asin(max(min(squash, 1.0), 0.0))
    return np.array([0.0, -math.cos(pitch), math.sin(pitch)])


def shade(n, h01, pal, squash, P=PARAMS, two_sided=None, gloss_scale=1.0):
    """n: (...,3) normals in the lighting space. Returns tone colour (...,3) 0..255,
    expressed relative to the palette 'base' so albedo variations survive.
    two_sided: mask of thin surfaces (leaves) lit from the side facing the viewer."""
    L = np.asarray(P["light_dir"], float)
    L /= np.linalg.norm(L)
    V = view_dir(squash)
    if two_sided is not None:
        flip = two_sided & ((n @ V) < 0)
        n = np.where(flip[..., None], -n, n)
    d = n @ L
    sh, ba, li, gl = (np.asarray(pal[k], np.float32) for k in ("shadow", "base", "light", "gloss"))
    col = sh + (ba - sh) * _ss(P["base_from"], P["base_to"], d)[..., None]
    col = col + (li - col) * _ss(P["light_from"], P["light_to"], d)[..., None]
    Hv = L + V
    Hv /= np.linalg.norm(Hv)
    g = _ss(P["gloss_from"], P["gloss_to"], n @ Hv) * _ss(P["gloss_min_height"], P["gloss_full_height"], h01)
    col = col + (gl - col) * (P["gloss_strength"] * gloss_scale * g)[..., None]
    rim = _ss(0.35, 0.0, np.clip(n @ V, 0, 1))
    col = col * (1 - P["rim_strength"] * rim)[..., None]
    return col / np.maximum(ba, 1)      # tone multiplier per channel


def _rot(img2d, yaw):
    return np.asarray(Image.fromarray(np.ascontiguousarray(img2d, np.float32), "F")
                      .rotate(yaw, resample=Image.BILINEAR))


def lit_frame(albedo, normals, spacing, yaw, pal, squash=0.75, out_scale=0.5,
              outline=(38, 107, 81), outline_px=2, heights01=None, two_sided=None, gloss=None):
    """two_sided: per-layer bool masks (thin surfaces, e.g. leaves) or None;
    gloss: per-layer gloss multipliers or None."""
    n0 = albedo[0].shape[0]
    S = int(round(n0 * out_scale))
    sq = max(1, int(round(S * squash)))
    lift = spacing * out_scale
    Hc = int(math.ceil(sq + lift * len(albedo))) + 12
    canvas = np.zeros((Hc, S, 4), np.float32)
    base_y = Hc - 6 - sq
    c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    L = len(albedo)
    for i in range(L):
        A = albedo[i]
        if A[..., 3].max() == 0:
            continue
        small = lambda ch: np.asarray(Image.fromarray(np.ascontiguousarray(ch, np.float32), "F")  # noqa: E731
                                      .resize((S, S), Image.BILINEAR))
        a = small(A[..., 3] / 255.0)
        rgb = [small(A[..., k] / 255.0 * A[..., 3] / 255.0) for k in range(3)]       # premultiplied
        nm = [small(normals[i][..., k] / 255.0 * 2 - 1) for k in range(3)]
        a = _rot(a, yaw)
        rgb = [_rot(x, yaw) for x in rgb]
        nm = [_rot(x, yaw) for x in nm]
        nx = nm[0] * c - nm[1] * s_          # rotate the vectors with the image (CCW)
        ny = nm[0] * s_ + nm[1] * c
        N = np.stack([nx, ny, nm[2]], -1)
        N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
        h01 = heights01[i] if heights01 is not None else i / max(L - 1, 1)
        ts = None
        if two_sided is not None and two_sided[i] is not None and two_sided[i].any():
            ts = _rot(small(two_sided[i].astype(np.float32)), yaw) > 0.5
        tone = shade(N, h01, pal, squash, two_sided=ts,
                     gloss_scale=1.0 if gloss is None else gloss[i])
        prem = np.stack(rgb, -1) * tone
        layer = np.concatenate([np.clip(prem, 0, 1), a[..., None]], -1)
        # squash vertically
        layer = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(layer[..., k], np.float32), "F")
                                     .resize((S, sq), Image.BILINEAR)) for k in range(4)], -1)
        copies = max(1, int(math.ceil(lift)))
        for k in range(copies):
            y = int(round(base_y - i * lift - k * lift / copies))
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


def lit_items(items, spacing, yaw, pal, squash=0.75, out_scale=1.0, outline=(38, 107, 81), outline_px=2,
              copies_per_layer=8):
    """General stack renderer for several parts on one grid.

    items: dicts with albedo, normal (RGBA uint8), index (float, body-grid units),
    span (grid units covered by the layer: 1 for body/leaves, 1/4 for face),
    h01, two_sided (bool), gloss (float). Every layer is drawn ceil(span * lift)
    times across its span, and ALL copies of all parts are drawn in order of
    height (ties: list order), so thin decal layers interleave correctly."""
    n0 = items[0]["albedo"].shape[0]
    S = int(round(n0 * out_scale))
    sq = max(1, int(round(S * squash)))
    lift = spacing * out_scale
    top = max(it["index"] + it["span"] for it in items)
    low = min(0.0, min(it["index"] for it in items))
    Hc = int(math.ceil(sq + lift * (top - low))) + 12
    canvas = np.zeros((Hc, S, 4), np.float32)
    base_y = Hc - 6 - sq
    c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    small = lambda ch: np.asarray(Image.fromarray(np.ascontiguousarray(ch, np.float32), "F")  # noqa: E731
                                  .resize((S, S), Image.BILINEAR))
    draws = []
    for j, it in enumerate(items):
        A = it["albedo"]
        if A[..., 3].max() == 0:
            continue
        a = _rot(small(A[..., 3] / 255.0), yaw)
        rgb = [_rot(small(A[..., k] / 255.0 * A[..., 3] / 255.0), yaw) for k in range(3)]
        nm = [_rot(small(it["normal"][..., k] / 255.0 * 2 - 1), yaw) for k in range(3)]
        N = np.stack([nm[0] * c - nm[1] * s_, nm[0] * s_ + nm[1] * c, nm[2]], -1)
        N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
        ts = (a > 0) if it.get("two_sided") else None
        tone = shade(N, it["h01"], pal, squash, two_sided=ts, gloss_scale=it.get("gloss", 1.0))
        layer = np.concatenate([np.clip(np.stack(rgb, -1) * tone, 0, 1), a[..., None]], -1)
        layer = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(layer[..., k], np.float32), "F")
                                     .resize((S, sq), Image.BILINEAR)) for k in range(4)], -1)
        copies = max(1, int(round(it["span"] * copies_per_layer)))
        for k in range(copies):
            draws.append((round(it["index"] + it["span"] * k / copies, 6), j, layer))
    draws.sort(key=lambda t: (t[0], t[1]))
    for h, _, layer in draws:
        y = int(round(base_y - (h - low) * lift))
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
