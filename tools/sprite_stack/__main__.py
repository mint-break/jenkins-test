"""CLI:  python -m tools.sprite_stack --input input/views.png --layers 32 --out output

Pipeline: view sheet -> visual hull voxels -> slice strip -> turntable previews -> OBJ.
The slices are always cut from the voxel data; nothing is drawn per layer.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

from . import mesh, render, slices
from .views import load_views, resize_view, segment_part
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
    ap.add_argument("--layers", type=int, default=32, help="number of slices (= voxel height)")
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
    ap.add_argument("--colors", type=int, default=0, help="quantise voxel colours (0=off)")
    ap.add_argument("--dirs", type=int, default=16, help="turntable directions")
    ap.add_argument("--scale", type=int, default=4, help="preview pixel scale")
    ap.add_argument("--squash", type=float, default=0.6, help="vertical squash of layers (pitch)")
    ap.add_argument("--no-mesh", action="store_true")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    P = lambda name: os.path.join(a.out, name)  # noqa: E731

    views = load_views(a.input, a.grid, a.order, a.bg_threshold)
    if a.part != "none":
        segment_part(views, a.part)
    debug_views(views, P("debug_views.png"))
    model = build(views, a.layers, a.appendage, a.plate, a.round_exp)
    if a.colors:
        quantize(model, a.colors)

    slices.write_strip(model, P("slices.png"))
    occ2, rgb2 = slices.read_strip(P("slices.png"), a.layers)
    same = np.array_equal(occ2, model.occ) and np.array_equal(rgb2[occ2], model.rgb[model.occ])
    print(f"slice strip round-trip identical to voxels: {same}")
    # stack/mesh renders use the slices read back from disk, i.e. the deliverable itself
    occ, rgb = occ2, rgb2

    kw = dict(scale=a.scale, squash=a.squash)
    flat = render.turntable(occ, rgb, a.dirs, **kw)
    litf = render.turntable(occ, rgb, a.dirs, lit=True, **kw)
    lit26 = render.turntable(occ, rgb, a.dirs, lit=True, normal_sigma=0, **kw)
    for frames, tag in ((flat, ""), (litf, "_lit")):
        render.sheet(frames, cols=min(8, a.dirs)).save(P(f"turntable_{a.dirs}{tag}.png"))
        render.save_gif(frames, P(f"turntable{tag}.gif"))
    # rows: flat / lit (raw 26-neighbour normals) / lit (smoothed normals), 4 directions
    picks = [0, a.dirs // 8, a.dirs // 4, a.dirs // 2]
    render.sheet([fr[i] for fr in (flat, lit26, litf) for i in picks], cols=4).save(
        P("compare_flat_vs_lit.png"))

    report = {"voxels": int(occ.sum()), **model.meta, "slices_roundtrip_identical": bool(same)}
    if not a.no_mesh:
        quads = mesh.greedy_quads(occ, rgb)
        mesh.write_obj(quads, P("model.obj"), occ.shape)
        area = sum(q[3] for q in quads)
        exp = mesh.exposed_face_count(occ)
        print(f"mesh: {len(quads)} quads, face area {area} vs exposed voxel faces {exp} "
              f"-> {'OK' if area == exp else 'MISMATCH'}")
        report.update(mesh_quads=len(quads), mesh_area=area, exposed_faces=exp)
        size = flat[0].size
        row = []
        for i in picks:
            row += [flat[i], mesh.render_quads(quads, occ.shape, 360 * i / a.dirs, size=size, **kw)]
        render.sheet(row, cols=2).save(P("compare_stack_vs_mesh.png"))
    with open(P("report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"done -> {a.out}/")


if __name__ == "__main__":
    main()
