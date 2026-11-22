"""STEP 3: orthographic previews of the 3D body (not the sprite stack).

Front / back / side are ray-cast analytically per image row (the cross-section
at that height is a superellipse, so the first hit has a closed form); the top
view searches the highest surface point per pixel. Normals come from the
implicit surface gradient, shading is soft cel with a light placed at the
viewer's upper left (as in the art), so each view reads like the reference.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd

from .model import BodyShape


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def palette_from_art(v, line_lum=150):
    lum = v.rgb @ np.array([0.299, 0.587, 0.114])
    body = v.mask & (lum >= line_lum)
    body = nd.binary_erosion(body, iterations=4)
    px = v.rgb[body]
    l = px @ np.array([0.299, 0.587, 0.114])
    q = np.percentile(l, [12, 50, 93])
    pick = lambda c: px[np.abs(l - c) < 4].mean(0)  # noqa: E731
    lines = v.rgb[v.mask & (lum < 110)]
    return {"shadow": pick(q[0]), "base": pick(q[1]), "light": pick(q[2]),
            "gloss": np.array([250.0, 254.0, 252.0]), "outline": np.median(lines, 0)}


def shade(n, view_dir, pal, light_cam=(-0.45, 0.55, 0.7), hnorm=None, right=None, up=None):
    """n: model-space normals; view_dir: unit vector from the surface to the viewer;
    right / up: model-space directions of the image's right and up. The light is
    given in camera space (x right, y up, z towards the viewer)."""
    vz = np.asarray(view_dir, float)
    right = np.asarray(right, float)
    up = np.asarray(up, float)
    L = light_cam[0] * right + light_cam[1] * up + light_cam[2] * vz
    L /= np.linalg.norm(L)
    d = n @ L
    sh, ba, li = (np.asarray(pal[k], np.float32) for k in ("shadow", "base", "light"))
    col = sh + (ba - sh) * smoothstep(0.05, 0.22, d)[..., None]
    col = col + (li - col) * smoothstep(0.72, 0.84, d)[..., None]
    # specular glint
    Hv = L + vz
    Hv /= np.linalg.norm(Hv)
    g = smoothstep(0.982, 0.993, n @ Hv)
    if hnorm is not None:                      # the glint belongs to the head, as in the art
        g = g * smoothstep(0.55, 0.65, hnorm)
    g = g[..., None]
    col = col + (np.asarray(pal["gloss"], np.float32) - col) * g * 0.8
    # soft rim towards the silhouette
    rim = smoothstep(0.35, 0.0, np.clip(n @ vz, 0, 1))[..., None]
    col = col * (1 - 0.10 * rim)
    return np.clip(col, 0, 255)


def _finish(rgb, mask, pal, outline_px=3, bg=(255, 255, 255)):
    out = np.zeros(mask.shape + (4,), np.uint8)
    out[..., :3] = rgb.astype(np.uint8)
    out[..., 3] = mask * 255
    if outline_px:
        ring = nd.binary_dilation(mask, iterations=outline_px) & ~mask
        out[ring, :3] = np.asarray(pal["outline"]).astype(np.uint8)
        out[ring, 3] = 255
    return Image.fromarray(out)


def render_side_on(shape: BodyShape, view: str, pal, px=1.0, pad=20):
    """front: viewer at -Z; back: viewer at +Z; side: viewer at -X (face to the left)."""
    W = float(np.max(shape.w(np.linspace(shape.bottom, shape.top, 600))))
    D = float(np.max(np.maximum(shape.df(np.linspace(shape.bottom, shape.top, 600)),
                                shape.db(np.linspace(shape.bottom, shape.top, 600)))))
    half = W if view in ("front", "back") else D
    nx = int(2 * (half + pad) / px)
    ny = int((shape.top - shape.bottom + 2 * pad) / px)
    us = (np.arange(nx) + 0.5) * px - (half + pad)          # image x (left -> right)
    ys = shape.top + pad - (np.arange(ny) + 0.5) * px       # height
    U, Y = np.meshgrid(us, ys)
    w = shape.w(Y)
    p = shape.p
    if view == "front":            # image right = +X; first hit at the front (z < 0)
        X = U
        Z = -shape.df(Y) * np.clip(1 - (np.abs(X) / np.maximum(w, 1e-6)) ** p, 0, 1) ** (1 / p)
        inside = np.abs(X) < w
        vdir, rgt, upv = (0, 0, -1), (1, 0, 0), (0, 1, 0)
    elif view == "back":           # viewer at +Z: image right = -X
        X = -U
        Z = shape.db(Y) * np.clip(1 - (np.abs(X) / np.maximum(w, 1e-6)) ** p, 0, 1) ** (1 / p)
        inside = np.abs(X) < w
        vdir, rgt, upv = (0, 0, 1), (-1, 0, 0), (0, 1, 0)
    else:                          # side, viewer at -X: image right = +Z (back), face on the left
        Z = U
        d = np.where(Z < 0, shape.df(Y), shape.db(Y))
        X = -w * np.clip(1 - (np.abs(Z) / np.maximum(d, 1e-6)) ** p, 0, 1) ** (1 / p)
        inside = np.abs(Z) < d
        vdir, rgt, upv = (-1, 0, 0), (0, 0, 1), (0, 1, 0)
    inside &= (Y >= shape.bottom) & (Y <= shape.top)
    n = shape.grad(X, Y, Z)
    rgb = shade(n, np.array(vdir, float), pal, hnorm=(Y - shape.bottom) / (shape.top - shape.bottom),
                right=rgt, up=upv)
    return _finish(rgb, inside, pal), (half + pad, shape.top + pad)


def render_top(shape: BodyShape, pal, px=1.0, pad=20):
    hs = np.linspace(shape.bottom, shape.top, 700)
    W = float(np.max(shape.w(hs)))
    Df = float(np.max(shape.df(hs)))
    Db = float(np.max(shape.db(hs)))
    nx = int(2 * (W + pad) / px)
    nz = int((Df + Db + 2 * pad) / px)
    xs = (np.arange(nx) + 0.5) * px - (W + pad)
    zs = -(Db + pad) + (np.arange(nz) + 0.5) * px          # image down = towards the front (-Z)
    Xg, Zg = np.meshgrid(xs, -zs)                           # row 0 = back
    Zg = -Zg
    Zg = Zg[::-1]                                           # row 0 = back (+Z), last row = front
    Xg = Xg
    hit = np.full(Xg.shape, -1.0)
    for h in hs[::-1]:
        w = shape.w(h)
        d = np.where(Zg < 0, shape.df(h), shape.db(h))
        if w <= 0:
            continue
        f = (np.abs(Xg) / w) ** shape.p + (np.abs(Zg) / np.maximum(d, 1e-6)) ** shape.p
        new = (f <= 1) & (hit < 0)
        hit[new] = h
    inside = hit >= 0
    n = shape.grad(Xg, np.where(inside, hit, shape.bottom), Zg)
    rgb = shade(n, np.array([0.0, 1.0, 0.0]), pal, hnorm=(hit - shape.bottom) / (shape.top - shape.bottom),
                right=(1, 0, 0), up=(0, 0, 1))
    return _finish(rgb, inside, pal)


def overlay(art_view, measure, shape: BodyShape, view: str):
    """Reference art with the model's silhouette (red) and the measured edge
    points used for fitting (green) / ignored (grey)."""
    im = Image.fromarray(art_view.rgb.astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(im)
    g, a = measure.ground, measure.axis
    hs = np.linspace(shape.bottom, shape.top, 600)
    if view in ("front", "back"):
        r1 = r2 = shape.w(hs)
    else:
        r1, r2 = shape.df(hs), shape.db(hs)
    for h in range(0, len(measure.h)):
        pass
    rows = np.arange(len(measure.h))
    for r in rows[::3]:
        for edge, raw, sgn in ((measure.left, measure.raw_left, -1), (measure.right, measure.raw_right, 1)):
            if np.isnan(raw[r]):
                continue
            x = a + sgn * raw[r]
            col = (40, 170, 70, 255) if not np.isnan(edge[r]) else (150, 150, 150, 180)
            d.ellipse([x - 2, r - 2, x + 2, r + 2], fill=col)
    d.line([(a - rr, g - h) for rr, h in zip(r1, hs)], fill=(225, 30, 30, 255), width=3)
    d.line([(a + rr, g - h) for rr, h in zip(r2, hs)], fill=(225, 30, 30, 255), width=3)
    return im


def overlay_top(art_top, shape: BodyShape, measure_side_axis=None):
    im = Image.fromarray(art_top.rgb.astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(im)
    m = art_top.mask
    ys, xs = np.nonzero(m)
    cx = (xs.min() + xs.max()) / 2
    hs = np.linspace(shape.bottom, shape.top, 400)
    hw = hs[int(np.argmax(shape.w(hs)))]
    w, df, db = float(shape.w(hw)), float(shape.df(hw)), float(shape.db(hw))
    cy = ys.min() + db                          # back edge of the art = top of the mask
    pts = []
    for t in np.linspace(0, 2 * np.pi, 400):
        c, s = np.cos(t), np.sin(t)
        dz = df if s > 0 else db                 # image down = front
        rr = 1 / ((abs(c) / w) ** shape.p + (abs(s) / dz) ** shape.p) ** (1 / shape.p)
        pts.append((cx + rr * c, cy + rr * s))
    d.line(pts + [pts[0]], fill=(225, 30, 30, 255), width=3)
    return im


def profile_chart(shape: BodyShape, front, back, side, path, scale=1.6):
    """Half width / half depth vs height: measured rows (dots) and the fitted body (lines)."""
    H = int((shape.top + 30) * scale)
    Wd = int(560 * scale)
    im = Image.new("RGB", (Wd, H), (250, 252, 251))
    d = ImageDraw.Draw(im)
    ox = Wd // 2
    hs = np.linspace(shape.bottom, shape.top, 600)
    for name, col, xs_, sgn in (("width", (40, 120, 200), [front.left, front.right, back.left, back.right], -1),
                                ("depth", (200, 110, 40), [side.left, side.right], 1)):
        hh = front.h if name == "width" else side.h
        if name == "width":
            hh_list = [front.h, front.h, back.h, back.h]
        else:
            hh_list = [side.h, side.h]
        for arr, hrow in zip(xs_, hh_list):
            for r in range(0, len(arr), 2):
                if not np.isnan(arr[r]):
                    x = ox + sgn * arr[r] * scale
                    y = H - 15 - hrow[r] * scale
                    d.point((x, y), fill=col)
    d.line([(ox - shape.w(h) * scale, H - 15 - h * scale) for h in hs], fill=(20, 70, 140), width=2)
    d.line([(ox + shape.df(h) * scale, H - 15 - h * scale) for h in hs], fill=(160, 70, 20), width=2)
    d.line([(ox + shape.db(h) * scale, H - 15 - h * scale) for h in hs], fill=(220, 150, 60), width=2)
    d.line([(ox, 0), (ox, H)], fill=(200, 200, 200))
    d.line([(0, H - 15 - shape.bottom * scale), (Wd, H - 15 - shape.bottom * scale)], fill=(200, 200, 200))
    d.text((8, 8), "left: half width (front+back, dots = measured rows)", fill=(20, 70, 140))
    d.text((8, 22), "right: half depth front (dark) / back (light) from side view", fill=(160, 70, 20))
    im.save(path)
