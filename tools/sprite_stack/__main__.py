"""CLI:  python -m tools.sprite_stack --input input/views.png --res 64 --layers 9 --out output

Pipeline: view sheet -> visual hull voxels -> layers (thick body tiers + fine
appendage layers) -> slice strip -> turntable previews -> OBJ.
The slices are always cut from the voxel data; nothing is drawn per layer.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

from . import mesh, render, slices
from .views import load_views, remove_part, resize_view, segment_part
from .voxelize import build, quantize


def debug_views(views, path: str, h: int = 160) -> None:
    tiles = []
    for name in ("top", "front", "side", "back"):
        v = views[name]
        w = max(1, round(v.w * h / v.h))
        m, c = resize_view(v, w, h)
        a = np.where(m[..., None], c, np.array([255.0, 0, 255]))
        if v.part is not None:
            from .views import protrusion_mask, resize_mask
            pm = resize_mask(v.part, w, h)
            qm = resize_mask(protrusion_mask(v), w, h)
            a[pm] = a[pm] * 0.5 + np.array([0, 60, 255.0]) * 0.5   # part over the body
            a[qm] = a[qm] * 0.4 + np.array([255.0, 40, 0]) * 0.6   # part sticking out
        a = a.astype(np.uint8)
        tiles.append(a)
        tiles.append(np.full((h, 8, 3), 30, np.uint8))
    Image.fromarray(np.concatenate(tiles, 1)).save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sprite_stack", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="input/views.png")
    ap.add_argument("--out", default="output")
    ap.add_argument("--res", type=int, default=64, help="voxel height (shape resolution)")
    ap.add_argument("--layers", type=int, default=9,
                    help="number of body tiers (thick layers); 0 = one layer per voxel")
    ap.add_argument("--top-layers", type=int, default=0,
                    help="layers for the appendage on top (0 = one per voxel, keeps leaves thin)")
    ap.add_argument("--grid", default="auto", help="COLSxROWS of the sheet, e.g. 4x1 or 2x2")
    ap.add_argument("--order", default="top,front,side,back", help="view names, row-major")
    ap.add_argument("--bg-threshold", type=int, default=40,
                    help="max(RGB) <= this is background (ignored if the PNG has alpha)")
    ap.add_argument("--appendage", default="auto",
                    help="thin part on top (leaves etc.): auto | off | fraction-from-top")
    ap.add_argument("--plate", type=int, default=2, help="appendage plate thickness (voxels)")
    ap.add_argument("--round", type=float, default=0.0, dest="round_exp",
                    help="carve body layers to a superellipse with this exponent (0=off, 2=ellipse)")
    ap.add_argument("--part", default="none",
                    help="split a colour-separable part before carving: none | warm | numpy expr of r,g,b")
    ap.add_argument("--remove-part", action="store_true",
                    help="erase the --part object instead of carving it (e.g. no sandwich)")
    ap.add_argument("--colors", type=int, default=0, help="quantise voxel colours (0=off)")
    ap.add_argument("--dirs", type=int, default=16, help="turntable directions")
    ap.add_argument("--scale", type=int, default=6, help="preview pixel scale")
    ap.add_argument("--smooth", action="store_true",
                    help="rotate upscaled layers instead of pixel-art (1px per voxel) rendering")
    ap.add_argument("--ridge", type=float, default=0.25, help="tier line / rim strength (0=off)")
    ap.add_argument("--no-outline", action="store_true")
    ap.add_argument("--squash", type=float, default=0.6, help="vertical squash of layers (pitch)")
    ap.add_argument("--no-mesh", action="store_true")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    P = lambda name: os.path.join(a.out, name)  # noqa: E731

    views = load_views(a.input, a.grid, a.order, a.bg_threshold)
    if a.part != "none":
        segment_part(views, a.part)
        if a.remove_part:
            remove_part(views)
    debug_views(views, P("debug_views.png"))
    model = build(views, a.res, a.appendage, a.plate, a.round_exp)
    if a.colors:
        quantize(model, a.colors)
    neck = model.meta.get("neck_y")
    bands = slices.plan_layers(model.occ.shape[1], a.layers, neck, a.top_layers)
    st = slices.cut(model.occ, model.rgb, bands)
    print(f"layers: {st.L} ({sum(1 for b in bands if neck is None or b[0] < neck)} body tiers"
          f" + {sum(1 for b in bands if neck is not None and b[0] >= neck)} top), "
          f"thickness {sorted(set(int(t) for t in st.thick))} voxels")

    slices.write_strip(st, P("slices.png"), model.meta)
    back = slices.read_strip(P("slices.png"))
    same = (np.array_equal(back.occ, st.occ) and np.array_equal(back.rgb[back.occ], st.rgb[st.occ])
            and np.allclose(back.bottom, st.bottom) and np.allclose(back.thick, st.thick))
    print(f"slice strip round-trip identical: {same}")

    kw = dict(scale=a.scale, squash=a.squash, pixel=not a.smooth, ridge=a.ridge,
              line=not a.no_outline)
    flat = render.turntable(model.occ, model.rgb, bands, a.dirs, **kw)
    litf = render.turntable(model.occ, model.rgb, bands, a.dirs, lit=True, **kw)
    for frames, tag in ((flat, ""), (litf, "_lit")):
        render.sheet(frames, cols=min(8, a.dirs)).save(P(f"turntable_{a.dirs}{tag}.png"))
        render.save_gif(frames, P(f"turntable{tag}.gif"))
    picks = [0, a.dirs // 8, a.dirs // 4, a.dirs // 2]
    render.sheet([fr[i] for fr in (flat, litf) for i in picks], cols=4).save(
        P("compare_flat_vs_lit.png"))

    report = {"voxels": int(model.occ.sum()), "layers": st.L,
              "layer_thickness": [int(t) for t in st.thick], **model.meta,
              "slices_roundtrip_identical": bool(same)}
    if not a.no_mesh:
        # mesh the layer stack itself (each layer extruded over its thickness)
        quads = mesh.greedy_quads(st.occ, st.rgb)
        area = sum(q[3] for q in quads)
        exp = mesh.exposed_face_count(st.occ)
        heights = list(st.bottom) + [st.top]
        quads = mesh.map_heights(quads, heights)
        mesh.write_obj(quads, P("model.obj"), st.occ.shape)
        print(f"mesh: {len(quads)} quads, face count {area} vs exposed layer faces {exp} "
              f"-> {'OK' if area == exp else 'MISMATCH'}")
        report.update(mesh_quads=len(quads), mesh_faces=area, exposed_faces=exp)
        X, _, Z = st.occ.shape
        row = []
        for i in picks:
            row += [flat[i], mesh.render_quads(quads, X, Z, st.top, 360 * i / a.dirs,
                                               scale=a.scale, squash=a.squash, pixel=not a.smooth)]
        render.sheet(row, cols=2).save(P("compare_stack_vs_mesh.png"))
    with open(P("report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"done -> {a.out}/")


if __name__ == "__main__":
    main()
