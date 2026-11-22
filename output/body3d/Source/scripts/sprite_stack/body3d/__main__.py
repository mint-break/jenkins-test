"""python -m tools.sprite_stack.body3d --input input/views_apose.png --out output/body3d

STEP 1-3: analyse the 4 views, build the 3D body, render front / side / back /
top previews of the 3D shape.
With --stack also STEP 4-6: shape checks, slice into 64 layers, export PNGs +
stack_config.json + previews + Source/ (mesh, profiles, scripts).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

from ..views import load_views
from . import analyze, checks, model, preview, stack


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sprite_stack.body3d")
    ap.add_argument("--input", default="input/views_apose.png")
    ap.add_argument("--order", default="front,side,back,top", help="views in reading order")
    ap.add_argument("--out", default="output/body3d")
    ap.add_argument("--head-aspect", type=float, default=1.0,
                    help="head dome: 1 = round, smaller = flatter")
    ap.add_argument("--round-top", type=float, default=0.0,
                    help="0..1: narrow the crown width towards the depth (0 = off)")
    ap.add_argument("--top-h", type=float, default=0.0,
                    help="fixed crown height above the ground in sheet px (overrides --top-above-leaf)")
    ap.add_argument("--top-above-leaf", type=float, default=0.04,
                    help="crown height = leaf base + this fraction of the body height (0 = from --head-aspect)")
    ap.add_argument("--width-top-margin", type=float, default=0.28,
                    help="stop measuring the width this far (fraction of height) below the leaf base")
    ap.add_argument("--stack", action="store_true", help="also run STEP 4-6 (checks, 64 layers, export)")
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--scale", type=float, default=1.2, help="canvas px per sheet px")
    ap.add_argument("--squash", type=float, default=0.75, help="preview camera pitch (vertical squash)")
    a = ap.parse_args(argv)
    for d in ("Preview", "Source"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    P = lambda *n: os.path.join(a.out, *n)  # noqa: E731

    views = load_views(a.input, order=a.order)
    # STEP 1
    mF = analyze.measure(views["front"], "frontback")
    mB = analyze.measure(views["back"], "frontback")
    mS = analyze.measure(views["side"], "side")
    # STEP 2 (shape first with a provisional roundness, then fit it to the top view)
    shape = model.build(mF, mB, mS, 2.0, head_aspect=a.head_aspect, round_top=a.round_top,
                        width_top_margin=a.width_top_margin, top_h=a.top_h,
                        top_above_leaf=a.top_above_leaf)
    hs = np.linspace(shape.bottom, shape.top, 500)
    hw = hs[int(np.argmax(shape.w(hs)))]
    p, p_err, _ = analyze.top_view_exponent(views["top"], float(shape.w(hw)), float(shape.db(hw)))
    shape.p = p
    shape.estimated.append(f"cross-section roundness p={p:.2f}: fitted to the top view's back half "
                           f"(+-60 deg, rms {p_err:.1f}px); width/depth come from the other 3 views")
    nv, nf = shape.write_obj(P("Source", "body.obj"))
    shape.to_json(P("Source", "body_shape.json"))
    # STEP 3
    pal = preview.palette_from_art(views["front"])
    imgs = {}
    for v in ("front", "side", "back"):
        imgs[v], _ = preview.render_side_on(shape, v, pal)
        imgs[v].save(P("Preview", f"{v}.png"))
    imgs["top"] = preview.render_top(shape, pal)
    imgs["top"].save(P("Preview", "top.png"))
    # checks against the art
    ov = [preview.overlay(views["front"], mF, shape, "front"),
          preview.overlay(views["side"], mS, shape, "side"),
          preview.overlay(views["back"], mB, shape, "back"),
          preview.overlay_top(views["top"], shape)]
    Wt = sum(o.width for o in ov) + 30
    Ht = max(o.height for o in ov)
    sheet = Image.new("RGBA", (Wt, Ht), (255, 255, 255, 255))
    x = 0
    for o in ov:
        sheet.alpha_composite(o, (x, Ht - o.height))
        x += o.width + 10
    sheet.save(P("Preview", "compare_with_art.png"))
    # 4 views side by side on white
    Wt = sum(i.width for i in imgs.values()) + 60
    Ht = max(i.height for i in imgs.values())
    sheet = Image.new("RGBA", (Wt, Ht), (255, 255, 255, 255))
    x = 0
    for k in ("front", "side", "back", "top"):
        sheet.alpha_composite(imgs[k], (x, (Ht - imgs[k].height) // 2 if k == "top" else Ht - imgs[k].height))
        x += imgs[k].width + 20
    sheet.save(P("Preview", "four_views.png"))
    preview.profile_chart(shape, mF, mB, mS, P("Preview", "profile_chart.png"))

    report = {"step": "1-3 (3D shape only, no sprite stack yet)", "mesh": {"vertices": nv, "faces": nf},
              "body_height_px": shape.top - shape.bottom, "bottom_px": shape.bottom, "top_px": shape.top,
              "max_half_width_px": float(np.max(shape.w(np.linspace(shape.bottom, shape.top, 500)))),
              "max_half_depth_front_px": float(np.max(shape.df(np.linspace(shape.bottom, shape.top, 500)))),
              "max_half_depth_back_px": float(np.max(shape.db(np.linspace(shape.bottom, shape.top, 500)))),
              "superellipse_p": p, "measured": shape.measured, "estimated": shape.estimated}
    with open(P("Source", "analysis.json"), "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    if not a.stack:
        print(json.dumps(report, indent=1, ensure_ascii=False))
        return
    # STEP 4
    chk = checks.run(shape, mF, mB, mS)
    with open(P("Source", "shape_checks.json"), "w") as fh:
        json.dump(chk, fh, indent=2, ensure_ascii=False)
    print("STEP 4 problems:", chk["problems"] or "none")
    if chk["problems"]:
        raise SystemExit("shape checks failed; fix the shape before slicing")
    # STEP 5-6
    cfg = stack.export(shape, pal, a.out, a.layers, a.size, a.scale, a.squash)
    copy_sources(P("Source", "scripts"))
    print(json.dumps(cfg["quality_checks"], indent=1))


def copy_sources(dst):
    """Ship the generator scripts with the assets (Source/scripts)."""
    import shutil
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)
    os.makedirs(os.path.join(dst, "sprite_stack", "body3d"), exist_ok=True)
    for f in ("__init__.py", "views.py"):
        shutil.copy(os.path.join(pkg, f), os.path.join(dst, "sprite_stack", f))
    for f in os.listdir(here):
        if f.endswith(".py"):
            shutil.copy(os.path.join(here, f), os.path.join(dst, "sprite_stack", "body3d", f))
    with open(os.path.join(dst, "README.md"), "w") as fh:
        fh.write("# Generator scripts\n\n"
                 "Copied from tools/sprite_stack. Requires Python 3, numpy, scipy, Pillow.\n\n"
                 "```\npython -m sprite_stack.body3d --input views_apose.png --stack --out out\n```\n"
                 "Run from this folder (so that `sprite_stack` is importable).\n")


if __name__ == "__main__":
    main()
