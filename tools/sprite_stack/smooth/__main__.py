"""python -m tools.sprite_stack.smooth --input input/views_pixel.png --out output/unity_stack

Step 1 (this tool so far): BODY ONLY - 64 smooth cross-sections at 512x512.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

from ..views import load_views, segment_part
from .body import Body, BodyParams, estimate_exponent, sample_palette
from .profile import estimate, estimate_v2, top_exponent
from . import render


def compare_with_art(views, prof, body, path):
    """Estimated body outline (red) over the front / side / back / top art."""
    from PIL import ImageDraw
    from .profile import _half_extent
    Hb = prof.raw["width"][2]
    tiles = []
    for name, (a_, b_) in (("front", ("left", "right")), ("side", ("front", "back")),
                           ("back", ("right", "left"))):
        v = views[name]
        _, _, _, _, _, axis, H = _half_extent(v)
        bottom = np.flatnonzero(v.mask.any(1))[-1]
        k = H / Hb                                       # view px per profile px
        im = Image.fromarray(v.rgb.astype(np.uint8))
        d = ImageDraw.Draw(im)
        hs = np.linspace(0, 1, 500)
        ys = bottom - hs * prof.height * k
        d.line([(axis - body.r[a_](h) * k, y) for h, y in zip(hs, ys)], fill=(230, 40, 40), width=3)
        d.line([(axis + body.r[b_](h) * k, y) for h, y in zip(hs, ys)], fill=(230, 40, 40), width=3)
        tiles.append(im)
    # top view: widest cross-section
    v = views["top"]
    im = Image.fromarray(v.rgb.astype(np.uint8))
    d = ImageDraw.Draw(im)
    ys_, xs_ = np.nonzero(v.mask)
    cy = (ys_.min() + ys_.max()) / 2
    cx = (xs_.min() + xs_.max()) / 2
    k = v.w / max(v.w, 1)
    hs = np.linspace(0, 0.6, 60)
    hw = hs[np.argmax([body.r["left"](h) for h in hs])]
    kx = (cy - ys_.min()) / max(body.r["back"](hw), 1)   # top view scale from the clean back side
    pts = []
    for t in np.linspace(0, 2 * np.pi, 360):
        c, s_ = np.cos(t), np.sin(t)
        rx = body.r["right"](hw) if c >= 0 else body.r["left"](hw)
        rz = body.r["front"](hw) if s_ >= 0 else body.r["back"](hw)
        rr = 1 / ((abs(c) / rx) ** body.prm.p + (abs(s_) / rz) ** body.prm.p) ** (1 / body.prm.p)
        pts.append((cx + rr * c * kx, cy + rr * s_ * kx))
    d.line(pts + [pts[0]], fill=(230, 40, 40), width=3)
    tiles.append(im)
    W = sum(t.width for t in tiles) + 10 * len(tiles)
    Hh = max(t.height for t in tiles)
    out = Image.new("RGB", (W, Hh))
    x = 0
    for t in tiles:
        out.paste(t, (x, Hh - t.height))
        x += t.width + 10
    out.save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sprite_stack.smooth")
    ap.add_argument("--input", default="input/views_pixel.png")
    ap.add_argument("--out", default="output/unity_stack")
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--body-width", type=float, default=300.0, help="widest body extent in px")
    ap.add_argument("--p", type=float, default=0.0, help="cross-section exponent (0 = from top view)")
    ap.add_argument("--squash", type=float, default=0.55, help="preview pitch")
    a = ap.parse_args(argv)

    for d in ("Body", "Preview", "Config"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    views = load_views(a.input)
    segment_part(views, "warm")
    prof = estimate_v2(views)
    p = a.p or top_exponent(views["top"])
    prm = BodyParams(layers=a.layers, size=a.size, body_width_px=a.body_width, p=p)
    body = Body(prof, prm)
    pal = sample_palette(views)
    print(f"cross-section exponent p={p:.2f}  body height {body.height_px:.1f}px  "
          f"layer spacing {body.spacing:.3f}px")

    layers = []
    for i in range(a.layers):
        img, _ = body.layer(i, pal)
        if img[..., 3].max() == 0:
            print(f"warning: layer {i} is empty")
        Image.fromarray(img).save(os.path.join(a.out, "Body", f"Layer_{i:03d}.png"))
        layers.append(img)

    # previews
    P = lambda n: os.path.join(a.out, "Preview", n)  # noqa: E731
    frames = render.turntable(layers, body.spacing, 16, squash=a.squash)
    render.sheet(frames, 8).save(P("body_turntable_16.png"))
    render.save_gif(render.turntable(layers, body.spacing, 48, squash=a.squash), P("body_turntable.gif"))
    # orthographic side-on views (pitch 0) to compare with the art silhouettes
    ortho = [render.stack_frame(layers, body.spacing, yaw, squash=0.02, out_scale=0.5, fill=True)
             for yaw in (0, 90, 180)]
    render.sheet(ortho, 3).save(P("body_ortho_front_side_back.png"))
    compare_with_art(views, prof, body, P("body_vs_art.png"))
    top = Image.fromarray(layers[0]).copy()
    for L in layers[1:]:
        top.alpha_composite(Image.fromarray(L))
    top.resize((a.size // 2, a.size // 2), Image.LANCZOS).save(P("body_top.png"))

    cfg = {
        "character": "mint",
        "stage": "body prototype",
        "canvas": {"width": a.size, "height": a.size, "pivot_px": [a.size / 2, a.size / 2],
                   "origin": "canvas centre = body axis (character centre)",
                   "image_axes": "x right = character's left (+X); image bottom = character front (-Z)"},
        "units": {"px_per_source_px": body.scale, "body_height_px": body.height_px},
        "parts": [{
            "name": "Body", "folder": "Body", "layers": a.layers,
            "files": [f"Body/Layer_{i:03d}.png" for i in range(a.layers)],
            "layer_height_px": [round(i * body.spacing, 4) for i in range(a.layers)],
            "layer_spacing_px": body.spacing,
            "position_px": [0, 0, 0],
            "draw_order": "bottom to top (index 0 first)",
            "cross_section": {"type": "four-radius superellipse", "exponent": p},
        }],
        "shading": {"palette": {k: [round(c) for c in v] for k, v in pal.items()},
                    "light_model_space": prm.light, "style": "soft cel, baked"},
        "estimated": prof.estimated + [
            f"cross-section roundness p={p:.2f}: superellipse through the top view's back-diagonal radius",
        ],
    }
    with open(os.path.join(a.out, "Config", "stack_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    print("done")


if __name__ == "__main__":
    main()
