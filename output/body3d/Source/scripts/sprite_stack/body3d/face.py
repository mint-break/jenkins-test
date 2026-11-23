"""Face part: eyes, mouth and cheeks as a thin decal shell on the body's front.

1. Decal texture: the features are cut from the FRONT view. For every pixel
   the body colour underneath is in-painted from the surrounding skin, and the
   feature strength is alpha = projection of (pixel - skin) onto
   (feature colour - skin). Colours are flat per feature (soft cel look).
   Saved as Face/face_texture.png (sheet scale, origin in the config).
2. Shell layers: the body's cross-section at fine heights (a quarter of the
   body layer spacing, so the 11 px mouth line keeps its curve), restricted to
   the outer `shell_px` of the front half. Each shell pixel takes the decal
   value at its front-view position (orthographic projection along -Z, i.e.
   exactly what the front view shows) and the body's surface normal there, so
   it is lit like the skin under it.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from PIL import Image
from scipy import ndimage as nd

from . import analyze, model
from .stack import shade_top, smoothstep

FEATURES = {
    # name: (selector on the front art, flat colour)
    "eyes_mouth": (lambda r, g, b, lum: (r > g + 5) & (lum < 150), None),
    "cheeks": (lambda r, g, b, lum: (r > g + 15) & (lum >= 150), None),
}


def build_shape(views):
    mF = analyze.measure(views["front"], "frontback")
    mB = analyze.measure(views["back"], "frontback")
    mS = analyze.measure(views["side"], "side")
    shape = model.build(mF, mB, mS, 2.0)
    hs = np.linspace(shape.bottom, shape.top, 500)
    hw = hs[int(np.argmax(shape.w(hs)))]
    shape.p, _, _ = analyze.top_view_exponent(views["top"], float(shape.w(hw)), float(shape.db(hw)))
    return shape, mF, mS


def decal(views, mF):
    """Returns (alpha HxW float, colour HxWx3, info) in front-view pixels."""
    v = views["front"]
    rgb = v.rgb.astype(float)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    alpha = np.zeros(v.mask.shape)
    colour = np.zeros(v.mask.shape + (3,))
    info = {}
    feats = {}
    for name, (sel, _) in FEATURES.items():
        core = v.mask & sel(r, g, b, lum)
        lab, k = nd.label(nd.binary_closing(core, iterations=1))
        sizes = nd.sum(core, lab, range(1, k + 1))
        core = np.isin(lab, np.flatnonzero(sizes > 15) + 1) & core
        feats[name] = core
    anyf = feats["eyes_mouth"] | feats["cheeks"]
    region = nd.binary_dilation(anyf, iterations=4)
    # skin under the features: nearest colour outside the (grown) feature region
    _, (iy, ix) = nd.distance_transform_edt(region, return_indices=True)
    skin = rgb[iy, ix]
    skin = np.stack([nd.uniform_filter(skin[..., c], 9) for c in range(3)], -1)
    for name, core in feats.items():
        strong = core & nd.binary_erosion(core, iterations=1)
        K = np.median(rgb[strong if strong.sum() > 20 else core], 0)
        grow = nd.binary_dilation(core, iterations=3)
        d = K - skin
        a = np.sum((rgb - skin) * d, -1) / np.maximum(np.sum(d * d, -1), 1)
        a = np.clip(a, 0, 1) * grow
        if name == "cheeks":                       # cheeks are soft blush in the art
            a = nd.gaussian_filter(a, 1.0) * grow
        take = a > alpha
        alpha[take] = a[take]
        colour[take] = K
        ys, xs = np.nonzero(core)
        info[name] = {"colour_rgb": [int(round(c)) for c in K],
                      "x_px": [float(xs.min() - mF.axis), float(xs.max() - mF.axis)],
                      "h_px": [float(mF.ground - ys.max()), float(mF.ground - ys.min())]}
    return alpha, colour, info


def shell_layers(shape, alpha, colour, mF, heights, scale, size, shell_px=4.0, foot_heights=None):
    """Face layers at the given heights (sheet px). The decal is sampled at
    `heights`; the shell follows the body cross-section at `foot_heights` (the
    centre of the body layer that is drawn at the same height), so the face
    sits exactly on that body layer's rim. Returns albedo + normal RGBA lists."""
    n = size
    c = (n - 1) / 2
    u, vv = np.meshgrid(np.arange(n) - c, np.arange(n) - c)
    X = u / scale
    Z = -vv / scale
    H, W = alpha.shape
    alb, nor = [], []
    foot_heights = heights if foot_heights is None else foot_heights
    for h, hf in zip(heights, foot_heights):
        Y = np.full_like(X, h)
        f = shape.F(X, np.full_like(X, hf), Z) + 1.0
        gv, gu = np.gradient(f)
        gl = np.hypot(gu, gv) + 1e-9
        cov = np.clip(0.5 - (f - 1) / gl, 0, 1)
        d_in = (1 - f) / gl                                  # canvas px inside the edge
        shell = cov * smoothstep(shell_px + 1, shell_px - 1, d_in) * (Z < 0)
        # front-view position: column = axis + x, row = ground - h
        col = mF.axis + X
        row = np.full_like(X, mF.ground - h)
        ok = (shell > 0) & (col >= 0) & (col < W - 1) & (row >= 0) & (row < H - 1)
        A = np.zeros((n, n, 4), np.uint8)
        Nm = np.zeros((n, n, 4), np.uint8)
        if ok.any():
            pts = [row[ok], col[ok]]
            a = nd.map_coordinates(alpha, pts, order=1) * shell[ok]
            rgbv = np.stack([nd.map_coordinates(colour[..., k], pts, order=0) for k in range(3)], -1)
            keep = a > 1.5 / 255
            if keep.any():
                yy, xx = np.nonzero(ok)
                yy, xx = yy[keep], xx[keep]
                A[yy, xx, :3] = np.clip(rgbv[keep], 0, 255).astype(np.uint8)
                A[yy, xx, 3] = np.round(a[keep] * 255).astype(np.uint8)
                nrm = shape.grad(X[yy, xx], Y[yy, xx], Z[yy, xx])
                ls = np.stack([nrm[:, 0], nrm[:, 2], nrm[:, 1]], -1)
                Nm[yy, xx, :3] = np.clip(np.round((ls * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
                Nm[yy, xx, 3] = A[yy, xx, 3]
                covd = Nm[..., 3] > 0
                dd, (iy, ix) = nd.distance_transform_edt(~covd, return_indices=True)
                pad = (~covd) & (dd <= 3)
                Nm[pad, :3] = Nm[iy[pad], ix[pad], :3]
        alb.append(A)
        nor.append(Nm)
    return alb, nor


def export_face(out_dir, views, sub=8, shell_px=4.0):
    body_cfg = json.load(open(os.path.join(out_dir, "Body", "stack_config.json")))
    sp = body_cfg["layer_spacing_px"]
    scale = body_cfg["scale"]["canvas_px_per_sheet_px"]
    bottom = body_cfg["scale"]["sheet_px_body_bottom_above_ground"]
    size = body_cfg["image"]["width"]
    dh = sp / scale
    pal = {k: np.array(v, float) for k, v in body_cfg["lighting"]["palette_rgb"].items()}
    shape, mF, mS = build_shape(views)
    alpha, colour, info = decal(views, mF)

    d = os.path.join(out_dir, "Face")
    for s in ("", "Albedo", "Normal"):
        os.makedirs(os.path.join(d, s), exist_ok=True)
        for f in os.listdir(os.path.join(d, s)):
            if f.startswith("Layer_") and f.endswith(".png"):
                os.remove(os.path.join(d, s, f))
    # decal texture (front-view crop)
    ys, xs = np.nonzero(alpha > 0.01)
    y0, y1, x0, x1 = ys.min() - 4, ys.max() + 5, xs.min() - 4, xs.max() + 5
    tex = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    tex[..., :3] = np.clip(colour[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    tex[..., 3] = np.round(alpha[y0:y1, x0:x1] * 255).astype(np.uint8)
    tex[tex[..., 3] == 0, :3] = 0
    Image.fromarray(tex, "RGBA").save(os.path.join(d, "face_texture.png"))

    hmin = min(v["h_px"][0] for v in info.values()) - 4
    hmax = max(v["h_px"][1] for v in info.values()) + 4
    step = dh / sub
    k0 = int(math.floor((hmin - bottom) / step))
    k1 = int(math.ceil((hmax - bottom) / step))
    heights = [bottom + (k + 0.5) * step for k in range(k0, k1 + 1)]
    foot = [bottom + (math.floor(k / sub) + 0.5) * dh for k in range(k0, k1 + 1)]
    alb, nor = shell_layers(shape, alpha, colour, mF, heights, scale, size, shell_px, foot)
    files, index = [], []
    for j, k in enumerate(range(k0, k1 + 1)):
        if alb[j][..., 3].max() == 0:
            continue
        fn = f"Layer_{k:03d}.png"
        Image.fromarray(alb[j], "RGBA").save(os.path.join(d, "Albedo", fn))
        Image.fromarray(nor[j], "RGBA").save(os.path.join(d, "Normal", fn))
        m = alb[j][..., 3] > 0
        baked = alb[j].copy()
        nv = nor[j][m][:, :3].astype(float) / 255 * 2 - 1
        tone = shade_top(np.stack([nv[:, 0], nv[:, 2], nv[:, 1]], -1), (heights[j] - bottom) /
                         (body_cfg["layers"] * dh), pal) / np.maximum(pal["base"], 1)
        baked[m, :3] = np.clip(alb[j][m][:, :3] * tone, 0, 255).astype(np.uint8)
        Image.fromarray(baked, "RGBA").save(os.path.join(d, fn))
        files.append(fn)
        index.append(round(k / sub, 6))           # body-grid index (fractional)

    # where do the features land in the side view? (consistency check)
    side_check = {}
    for name, v in info.items():
        hc = sum(v["h_px"]) / 2
        xc = sum(v["x_px"]) / 2 if name != "eyes_mouth" else max(abs(v["x_px"][0]), abs(v["x_px"][1])) * 0.8
        w = float(shape.w(hc))
        z = -float(shape.df(hc)) * max(0.0, 1 - (abs(xc) / w) ** shape.p) ** (1 / shape.p)
        side_check[name] = {"height_px": round(hc, 1), "model_depth_px": round(z, 1)}
    cfg = {
        "name": "MintFace", "part": "Face", "version": 1,
        "grid": "same canvas, pivot and axes as Body; layer index is FRACTIONAL on the body grid "
                f"(1/{sub} of the body spacing): height_px = index * layer_spacing_px",
        "layers": len(files),
        "layer_index": index,
        "layer_span": 1.0 / sub,
        "files": [f"Face/{f}" for f in files],
        "albedo_files": [f"Face/Albedo/{f}" for f in files],
        "normal_files": [f"Face/Normal/{f}" for f in files],
        "layer_height_px": [round(i * sp, 4) for i in index],
        "layer_spacing_px": sp,
        "image": body_cfg["image"], "pivot": body_cfg["pivot"], "axes": body_cfg["axes"],
        "texture": {"file": "Face/face_texture.png",
                    "origin_in_front_view_px": {"axis_col": float(mF.axis - x0),
                                                "ground_row": float(mF.ground - y0)},
                    "scale": "1 texture px = 1 sheet px; projected along -Z onto the body"},
        "features": info,
        "side_view_check": {"art_side_eye_depth_px": [-91.5, -72.5], "art_side_cheek_depth_px": [-79.5, -50.5],
                            "model": side_check},
        "render": {"two_sided": False, "gloss_scale": 0.0,
                   "note": "decal shell on the body's front surface; lit with the body's normals"},
        "draw_order": "every part is drawn copies_per_layer (8) times per body layer; face layer k/8 "
                      "is drawn right after the body copy at the same height and lies on that "
                      "copy's rim (same cross-section)",
        "estimated": [
            "feature shapes and positions come from the front view only (projected along -Z); "
            "the side view is used as a check",
            "skin colour under the features is in-painted from the surrounding skin",
            "colours are flat per feature (eyes+mouth brown, cheeks pink) with soft edges",
        ],
    }
    with open(os.path.join(d, "stack_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    return cfg, shape
