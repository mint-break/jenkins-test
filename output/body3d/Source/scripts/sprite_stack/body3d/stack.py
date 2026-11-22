"""STEP 5-6: slice the 3D body into 64 horizontal layers and export them.

Each layer is the body's cross-section at the CENTRE height of its slab,
evaluated from the same implicit surface as the mesh (no image cutting):
  * coverage  : anti-aliased from the signed distance (f - 1) / |grad f|
  * colour    : soft cel shading of the 3D surface normal at that point, with
                ONE light for the whole body. Baked lighting turns with the
                sprite stack, so the light comes from straight above: every
                yaw then reads the same (lit crown, mint sides, shaded base).
  * no outline in the layers: it is added once after stacking (per-layer
    outlines would leave rings inside the stack)

Layer image convention: top-down view, canvas centre = body axis, image
right = the character's left (+X), image bottom = the character's FRONT (-Z).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from PIL import Image
from scipy import ndimage as nd

from .model import BodyShape


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def shade_top(n, hnorm, pal):
    """Rotation-invariant soft cel: depends only on how much the normal faces up.
    The tone changes are spread over a wide range of slopes so that neighbouring
    layers never differ by a visible step (no rings on the crown / base)."""
    sh, ba, li = (np.asarray(pal[k], np.float32) for k in ("shadow", "base", "light"))
    up = n[..., 1]
    col = sh + (ba - sh) * smoothstep(-0.75, -0.05, up)[..., None]      # underside / base
    col = col + (li - col) * smoothstep(0.15, 0.95, up)[..., None]      # crown
    # very gentle vertical gradient, lighter towards the top (as in the art)
    col = col * (0.965 + 0.05 * np.clip(hnorm, 0, 1))[..., None]
    return np.clip(col, 0, 255)


def make_layers(shape: BodyShape, pal, layers=64, size=512, scale=1.2):
    n = size
    c = (n - 1) / 2
    u, v = np.meshgrid(np.arange(n) - c, np.arange(n) - c)
    X = u / scale                       # sheet px, +X to the image right
    Z = -v / scale                      # image down = front = -Z
    H = shape.top - shape.bottom
    dh = H / layers
    out, heights = [], []
    for i in range(layers):
        h = shape.bottom + (i + 0.5) * dh
        Y = np.full_like(X, h)
        f = shape.F(X, Y, Z) + 1.0       # superellipse value, < 1 inside
        gv, gu = np.gradient(f)          # per canvas px
        gl = np.hypot(gu, gv) + 1e-9
        alpha = np.clip(0.5 - (f - 1) / gl, 0, 1)
        img = np.zeros((n, n, 4), np.uint8)
        m = alpha > 0
        if m.any():
            nrm = shape.grad(X[m], Y[m], Z[m])
            img[m, :3] = shade_top(nrm, (h - shape.bottom) / H, pal).astype(np.uint8)
            img[m, 3] = np.round(alpha[m] * 255).astype(np.uint8)
        img[img[..., 3] == 0, :3] = 0    # nothing left in transparent pixels
        out.append(img)
        heights.append(i * dh * scale)  # bottom of the slab, canvas px above the first layer
    return out, np.array(heights), dh * scale


# ---------------------------------------------------------------- preview

def _premul(img):
    a = img.astype(np.float32) / 255
    a[..., :3] *= a[..., 3:4]
    return a


def stack_frame(layers, spacing, yaw, squash=0.75, out_scale=0.5, outline=(38, 107, 81), outline_px=2):
    """Rotate every layer about the canvas centre, squash vertically and draw it
    once per pixel of its slab (no gaps), bottom first; outline once at the end."""
    n = layers[0].shape[0]
    S = int(round(n * out_scale))
    sq = max(1, int(round(S * squash)))
    lift = spacing * out_scale
    Hc = int(math.ceil(sq + lift * len(layers))) + 12
    canvas = np.zeros((Hc, S, 4), np.float32)
    base = Hc - 6 - sq
    for i, L in enumerate(layers):
        if L[..., 3].max() == 0:
            continue
        im = Image.fromarray(np.round(_premul(L) * 255).astype(np.uint8), "RGBa")
        im = im.resize((S, S), Image.BILINEAR).rotate(yaw, resample=Image.BILINEAR)
        a = np.asarray(im.resize((S, sq), Image.BILINEAR), np.float32) / 255
        copies = max(1, int(math.ceil(lift)))
        for k in range(copies):
            y = int(round(base - i * lift - k * lift / copies))
            y0, y1 = max(y, 0), min(y + sq, Hc)
            if y1 > y0:
                src = a[y0 - y:y1 - y]
                canvas[y0:y1] = src + canvas[y0:y1] * (1 - src[..., 3:4])
    rgb = np.divide(canvas[..., :3], canvas[..., 3:4], out=np.zeros_like(canvas[..., :3]),
                    where=canvas[..., 3:4] > 1e-4)
    img = np.concatenate([rgb, canvas[..., 3:4]], -1)
    img = np.round(img * 255).astype(np.uint8)
    if outline is not None:
        m = nd.binary_fill_holes(img[..., 3] > 110)
        ring = nd.binary_dilation(m, iterations=outline_px) & ~m
        img[ring] = list(outline) + [255]
    return Image.fromarray(img)


def turntable(layers, spacing, n=16, **kw):
    return [stack_frame(layers, spacing, 360.0 * i / n, **kw) for i in range(n)]


def sheet(frames, cols=8, bg=(255, 255, 255, 255)):
    w = max(f.width for f in frames)
    h = max(f.height for f in frames)
    rows = math.ceil(len(frames) / cols)
    out = Image.new("RGBA", (w * cols, h * rows), bg)
    for i, f in enumerate(frames):
        out.alpha_composite(f, ((i % cols) * w + (w - f.width) // 2, (i // cols) * h + h - f.height))
    return out


def save_gif(frames, path, bg=(250, 252, 251), ms=70):
    flat = []
    for f in frames:
        b = Image.new("RGB", f.size, bg)
        b.paste(f, mask=f.getchannel("A"))
        flat.append(b.quantize(255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    flat[0].save(path, save_all=True, append_images=flat[1:], duration=ms, loop=0, disposal=2)


# ---------------------------------------------------------------- quality checks

def verify(layers, shape: BodyShape, scale, spacing):
    """Checks on the exported pixels themselves."""
    res = {}
    res["count"] = len(layers)
    res["all_512_rgba"] = all(L.shape == (512, 512, 4) for L in layers)
    res["colour_in_transparent_px"] = int(sum(int((L[L[..., 3] == 0][:, :3] != 0).sum()) for L in layers))
    res["empty_layers"] = [i for i, L in enumerate(layers) if L[..., 3].max() == 0]
    # silhouette extents per layer vs. the 3D profiles
    H = shape.top - shape.bottom
    errs = []
    c = (512 - 1) / 2
    for i, L in enumerate(layers):
        m = L[..., 3] >= 128
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        h = shape.bottom + (i + 0.5) * H / len(layers)
        w_img = (xs.max() - xs.min() + 1) / 2 / scale
        front_img = (ys.max() - c + 0.5) / scale
        back_img = (c - ys.min() + 0.5) / scale
        errs.append([abs(w_img - shape.w(h)), abs(front_img - shape.df(h)), abs(back_img - shape.db(h))])
    errs = np.array(errs)
    res["layer_extent_error_sheet_px"] = {"mean": float(errs.mean()), "max": float(errs.max())}
    # same centre in every layer: centroid of each layer's coverage
    cents = []
    for L in layers:
        a = L[..., 3].astype(np.float64)
        if a.sum() > 0:
            ys, xs = np.indices(a.shape)
            cents.append((float((xs * a).sum() / a.sum()), float((ys * a).sum() / a.sum())))
    cx = np.array([p[0] for p in cents])
    res["centre_x_drift_px"] = float(np.ptp(cx))
    # neighbouring layers: area change (no sudden jumps on the side wall)
    areas = np.array([(L[..., 3] / 255).sum() for L in layers])
    rel = np.abs(np.diff(areas)) / np.maximum(areas[1:], 1)
    res["max_area_change_mid_layers"] = float(rel[4:-4].max())
    # colour continuity: mean rim colour of neighbouring layers
    rims = []
    for L in layers:
        m = L[..., 3] > 200
        if m.sum() < 50:
            rims.append(None)
            continue
        ring = m & ~nd.binary_erosion(m, iterations=3)
        rims.append(L[ring][:, :3].astype(float).mean(0))
    steps = [float(np.abs(a - b).max()) for a, b in zip(rims[:-1], rims[1:]) if a is not None and b is not None]
    res["max_rim_colour_step_between_layers"] = max(steps) if steps else 0.0
    return res


def stripe_metric(frame: Image.Image):
    """Horizontal stripes show up as fast up/down changes of brightness along a
    vertical line through the stacked body. Returns the 95th percentile of the
    absolute second difference of brightness inside the body (0..255 scale)."""
    a = np.asarray(frame).astype(float)
    m = a[..., 3] > 250
    cols = np.flatnonzero(m.any(0))
    vals = []
    for cx in np.linspace(cols[0] + 0.3 * np.ptp(cols), cols[0] + 0.7 * np.ptp(cols), 9).astype(int):
        rows = np.flatnonzero(m[:, cx])
        if len(rows) < 20:
            continue
        lum = a[rows[5]:rows[-5], cx, :3] @ np.array([0.299, 0.587, 0.114])
        vals.append(np.abs(np.diff(lum, 2)))
    v = np.concatenate(vals)
    return {"p95_second_diff": float(np.percentile(v, 95)), "max_second_diff": float(v.max())}


# ---------------------------------------------------------------- export

def export(shape: BodyShape, pal, out_dir, layers=64, size=512, scale=1.2, squash=0.75):
    body_dir = os.path.join(out_dir, "Body")
    prev_dir = os.path.join(out_dir, "Preview")
    os.makedirs(body_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)
    imgs, heights, spacing = make_layers(shape, pal, layers, size, scale)
    files = []
    for i, L in enumerate(imgs):
        fn = f"Layer_{i:03d}.png"
        Image.fromarray(L, "RGBA").save(os.path.join(body_dir, fn))
        files.append(fn)
    # read back what was written: every check and preview uses the PNG files
    disk = [np.asarray(Image.open(os.path.join(body_dir, f)).convert("RGBA")) for f in files]

    frames = turntable(disk, spacing, 16, squash=squash)
    sheet(frames, 8).save(os.path.join(prev_dir, "stacked_turntable_16.png"))
    save_gif(turntable(disk, spacing, 48, squash=squash), os.path.join(prev_dir, "stacked_turntable.gif"))
    hero = [stack_frame(disk, spacing, yaw, squash=squash, out_scale=0.75) for yaw in (0, -45, -90, 180)]
    sheet(hero, 4).save(os.path.join(prev_dir, "stacked_preview.png"))

    checks = verify(disk, shape, scale, spacing)
    checks["stripes_front"] = stripe_metric(stack_frame(disk, spacing, 0, squash=squash, out_scale=1.0,
                                                        outline=None))
    checks["stripes_back"] = stripe_metric(stack_frame(disk, spacing, 180, squash=squash, out_scale=1.0,
                                                       outline=None))
    # silhouette area over a full turn (should vary smoothly, no popping)
    areas = []
    for yaw in range(0, 360, 10):
        fr = np.asarray(stack_frame(disk, spacing, yaw, squash=squash, out_scale=0.5, outline=None))
        areas.append(float((fr[..., 3] > 128).sum()))
    areas = np.array(areas)
    checks["turn_area_max_jump"] = float(np.max(np.abs(np.diff(np.r_[areas, areas[0]]))) / areas.mean())

    H = shape.top - shape.bottom
    cfg = {
        "name": "MintBody",
        "part": "Body",
        "version": 1,
        "layers": layers,
        "files": [f"Body/{f}" for f in files],
        "image": {"width": size, "height": size, "format": "PNG RGBA, straight alpha",
                  "background": "fully transparent"},
        "pivot": {"x_px": size / 2, "y_px": size / 2,
                  "meaning": "image centre = body axis (character centre) in every layer"},
        "axes": {"image_right": "+X (character's left)", "image_down": "-Z (character's front)",
                 "stack_up": "+Y"},
        "draw_order": "ascending layer index (Layer_000 = bottom, drawn first)",
        "layer_spacing_px": spacing,
        "layer_height_px": [round(float(h), 4) for h in heights],
        "slab_thickness_px": spacing,
        "scale": {"canvas_px_per_sheet_px": scale,
                  "sheet_px_body_bottom_above_ground": shape.bottom,
                  "ground_offset_px": shape.bottom * scale,
                  "note": "ground is the feet bottom in the reference sheet; the body starts "
                          "ground_offset_px above it (feet fill the gap)"},
        "rendering": {
            "recommended_squash": squash,
            "fill_between_layers": f"draw each layer about {math.ceil(spacing)} times at 1px offsets "
                                   "(or display at 1/layer_spacing_px scale so the spacing is 1 screen px)",
            "outline": "not baked; add once after stacking (render the stack to a texture, then outline)",
            "rotation": "rotate every layer by the same angle about the pivot",
            "positions": "round screen positions to whole pixels",
            "sorting": "sort whole stacks by their ground point, not layer by layer",
        },
        "lighting": {"model": "soft cel, baked, light from straight above (rotation invariant)",
                     "palette_rgb": {k: [int(round(x)) for x in v] for k, v in pal.items()}},
        "parts": {"Body": {"layers": layers, "parent": None, "offset_px": [0, 0, 0]},
                  "planned": ["Leaves", "Face", "Arms", "Feet", "Tail"],
                  "attach_note": "future parts use the same canvas size, pivot and spacing; "
                                 "their layer_height_px is measured from the same base"},
        "source": {"mesh": "Source/body.obj", "profiles": "Source/body_shape.json",
                   "analysis": "Source/analysis.json"},
        "body_height_sheet_px": H,
        "quality_checks": checks,
        "estimated": shape.estimated,
    }
    with open(os.path.join(body_dir, "stack_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    return cfg
