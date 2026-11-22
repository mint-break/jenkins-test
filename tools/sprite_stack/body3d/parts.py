"""Export extra parts (leaves, ...) on the body's layer grid and the
character-level config that merges every part's layers into one draw order.

python -m tools.sprite_stack.body3d.parts --out output/body3d   (after the body export)
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as nd

from ..views import load_views
from . import analyze, leaves, lit
from .stack import shade_top


def _rd(path):
    return np.asarray(Image.open(path).convert("RGBA"))


def _over(top, bot):
    """Straight-alpha 'over' for preview compositing of two layers on one canvas."""
    ta = top[..., 3:4] / 255.0
    ba = bot[..., 3:4] / 255.0
    oa = ta + ba * (1 - ta)
    rgb = np.where(oa > 0, (top[..., :3] * ta + bot[..., :3] * ba * (1 - ta)) / np.maximum(oa, 1e-6), 0)
    out = np.zeros_like(top)
    out[..., :3] = np.clip(rgb, 0, 255)
    out[..., 3] = np.clip(np.round(oa[..., 0] * 255), 0, 255)
    return out.astype(np.uint8)


def export_leaves(out_dir, params: leaves.LeafParams | None = None, views=None):
    body_cfg = json.load(open(os.path.join(out_dir, "Body", "stack_config.json")))
    sp = body_cfg["layer_spacing_px"]
    scale = body_cfg["scale"]["canvas_px_per_sheet_px"]
    bottom = body_cfg["scale"]["sheet_px_body_bottom_above_ground"]
    size = body_cfg["image"]["width"]
    dh = sp / scale
    pal = {k: np.array(v, float) for k, v in body_cfg["lighting"]["palette_rgb"].items()}
    p = params or leaves.LeafParams()
    pts = leaves.all_points(p)
    y = pts["pos"][:, 1]
    k0 = int(math.floor((y.min() - bottom) / dh))
    k1 = int(math.floor((y.max() - bottom) / dh))
    heights = [bottom + k * dh for k in range(k0, k1 + 1)]
    alb, nor = leaves.slice_layers(pts, heights, dh, scale, size)

    d = os.path.join(out_dir, "Leaves")
    for sub in ("", "Albedo", "Normal"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
        for f in os.listdir(os.path.join(d, sub)):          # no stale layers from an earlier run
            if f.startswith("Layer_") and f.endswith(".png"):
                os.remove(os.path.join(d, sub, f))
    files, idx = [], []
    for j, k in enumerate(range(k0, k1 + 1)):
        if alb[j][..., 3].max() == 0:
            continue
        fn = f"Layer_{k:03d}.png"
        Image.fromarray(alb[j], "RGBA").save(os.path.join(d, "Albedo", fn))
        Image.fromarray(nor[j], "RGBA").save(os.path.join(d, "Normal", fn))
        # baked fallback (light from straight above, like the body's baked layers)
        m = alb[j][..., 3] > 0
        baked = alb[j].copy()
        if m.any():
            nv = nor[j][m][:, :3].astype(float) / 255 * 2 - 1
            nmodel = np.stack([nv[:, 0], np.abs(nv[:, 2]), nv[:, 1]], -1)   # two-sided: upper face
            tone = shade_top(nmodel, 1.0, pal) / np.maximum(pal["base"], 1)
            baked[m, :3] = np.clip(alb[j][m][:, :3] * tone, 0, 255).astype(np.uint8)
        Image.fromarray(baked, "RGBA").save(os.path.join(d, fn))
        files.append(fn)
        idx.append(k)

    B, dvec, w, n, L = leaves._frame(p)
    cfg = {
        "name": "MintLeaves", "part": "Leaves", "version": 1,
        "grid": "same canvas, pivot, axes and layer spacing as Body; layer index k sits at "
                "Body layer_height_px[0] + k * layer_spacing_px",
        "layers": len(files),
        "layer_index": idx,
        "layer_span": 1.0,
        "files": [f"Leaves/{f}" for f in files],
        "albedo_files": [f"Leaves/Albedo/{f}" for f in files],
        "normal_files": [f"Leaves/Normal/{f}" for f in files],
        "layer_height_px": [round(k * sp, 4) for k in idx],
        "layer_spacing_px": sp,
        "image": body_cfg["image"], "pivot": body_cfg["pivot"], "axes": body_cfg["axes"],
        "draw_order": "by layer index; at the same index draw Body first, then Leaves",
        "render": {"two_sided": True, "gloss_scale": 0.35,
                   "note": "thin surface: light the side facing the viewer (flip the normal when "
                           "dot(n, view) < 0). Normals store the leaf's upper face."},
        "geometry_sheet_px": {
            "right_leaf": {"base": list(p.base), "tip": list(p.tip), "length_3d": round(L, 2),
                           "true_width": p.width, "roll_deg": p.roll_deg,
                           "width_dir": [round(float(c), 4) for c in w],
                           "upper_normal": [round(float(c), 4) for c in n]},
            "left_leaf": "mirror of the right leaf in x",
            "bend": p.bend, "cup": p.cup, "thickness": p.thickness,
            "width_profile": {"u": list(p.profile_u), "w": list(p.profile_w)}},
        "colours_rgb": p.colours,
        "estimated": [
            "leaf direction from the front tip (+-150, 436) and the side tip (back 108, 436): "
            "the leaves lean back, which only the side view shows",
            f"blade roll {p.roll_deg} deg: the only flat-blade angle that matches both the front "
            "(100 px) and side (104 px) widths; the top view (an estimated drawing) shows the "
            "blades lying flatter, which the other views do not allow",
            "true width 113 px solved from the front + side apparent widths",
            "bend, cupping, thickness (plush leaf), serration and vein pattern are drawn to "
            "match the art's style, not measured",
            "stem: one shared stem on the axis, ~5 px above the crown; both blades start at its top (lower end inside the head)",
        ],
    }
    with open(os.path.join(d, "stack_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    return cfg, pts


def character_config(out_dir, parts_cfgs):
    body = json.load(open(os.path.join(out_dir, "Body", "stack_config.json")))
    sp = body["layer_spacing_px"]
    entries = []
    for i, f in enumerate(body["lighting"]["lit"]["albedo_files"]):
        entries.append({"part": "Body", "index": i, "span": 1.0, "albedo": f,
                        "normal": body["lighting"]["lit"]["normal_files"][i], "baked": body["files"][i]})
    part_rank = {"Body": 0}
    for r, pc in enumerate(parts_cfgs, start=1):
        part_rank[pc["part"]] = r
        for k, a, nfile, bfile in zip(pc["layer_index"], pc["albedo_files"], pc["normal_files"], pc["files"]):
            entries.append({"part": pc["part"], "index": k, "span": pc.get("layer_span", 1.0),
                            "albedo": a, "normal": nfile, "baked": bfile})
    entries.sort(key=lambda e: (e["index"], part_rank[e["part"]]))
    for o, e in enumerate(entries):
        e["draw_order"] = o
        e["height_px"] = round(e["index"] * sp, 4)
    top_index = int(math.ceil(max(e["index"] + e["span"] for e in entries))) - 1
    cfg = {
        "name": "Mint", "version": 1,
        "canvas": body["image"], "pivot": body["pivot"], "axes": body["axes"],
        "layer_spacing_px": sp, "grid_layers": top_index + 1,
        "height01_divisor": body["layers"],
        "parts": {"Body": {"config": "Body/stack_config.json", "two_sided": False, "gloss_scale": 1.0}},
        "draw": entries,
        "copies_per_layer": 8,
        "draw_rule": "each entry covers [index, index + span) on the body grid and is drawn "
                     "round(span * copies_per_layer) times at equal steps across it; sort ALL drawn "
                     "copies of all parts by height (ties: the order of this list, Body first)",
        "lighting": body["lighting"]["lit"]["shader_params"],
        "palette_rgb": body["lighting"]["palette_rgb"],
        "rendering": body["rendering"],
    }
    for pc in parts_cfgs:
        cfg["parts"][pc["part"]] = {"config": f"{pc['part']}/stack_config.json",
                                    "two_sided": pc["render"]["two_sided"],
                                    "gloss_scale": pc["render"]["gloss_scale"],
                                    "span": pc.get("layer_span", 1.0)}
    cfg["parts"]["Body"]["span"] = 1.0
    cfg["planned_parts"] = [p for p in ["Face", "Arms", "Feet", "Tail"] if p not in cfg["parts"]]
    with open(os.path.join(out_dir, "character_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    return cfg


def merged_layers(out_dir, char_cfg):
    """Per grid index: composite all parts (preview only). Returns albedo, normal,
    two-sided masks, gloss multipliers."""
    n = char_cfg["canvas"]["width"]
    K = char_cfg["grid_layers"]
    A = [np.zeros((n, n, 4), np.uint8) for _ in range(K)]
    N = [np.zeros((n, n, 4), np.uint8) for _ in range(K)]
    TS = [np.zeros((n, n), bool) for _ in range(K)]
    G = [1.0] * K
    for e in char_cfg["draw"]:
        k = e["index"]
        a = _rd(os.path.join(out_dir, e["albedo"]))
        nm = _rd(os.path.join(out_dir, e["normal"]))
        m = a[..., 3] > 0
        A[k] = _over(a, A[k])
        N[k][m] = nm[m]
        part = char_cfg["parts"][e["part"]]
        if part["two_sided"]:
            TS[k] |= m
        if m.any() and part["gloss_scale"] < G[k]:
            G[k] = part["gloss_scale"]
    return A, N, TS, G


def load_items(out_dir, char_cfg, parts=None):
    items = []
    for e in char_cfg["draw"]:
        if parts is not None and e["part"] not in parts:
            continue
        part = char_cfg["parts"][e["part"]]
        items.append({"albedo": _rd(os.path.join(out_dir, e["albedo"])),
                      "normal": _rd(os.path.join(out_dir, e["normal"])),
                      "index": e["index"], "span": e["span"],
                      "h01": (e["index"] + 0.5 * e["span"]) / char_cfg["height01_divisor"],
                      "two_sided": part["two_sided"], "gloss": part["gloss_scale"]})
    return items


def previews(out_dir, char_cfg, views, leaf_pts=None, squash=0.75, tag="character"):
    prev = os.path.join(out_dir, "Preview")
    os.makedirs(prev, exist_ok=True)
    pal = {k: np.array(v, float) for k, v in char_cfg["palette_rgb"].items()}
    sp = char_cfg["layer_spacing_px"]
    items = load_items(out_dir, char_cfg)

    def frame(yaw, out=0.5):
        # render at full size and downsample: thin parts (leaves) stay solid
        f = lit.lit_items(items, sp, yaw, pal, squash, 1.0, outline_px=int(round(2 / out)))
        return f.resize((max(1, int(f.width * out)), max(1, int(f.height * out))), Image.LANCZOS)

    from .stack import sheet, save_gif
    sheet([frame(y, 0.75) for y in (0, -45, -90, 180)], 4).save(os.path.join(prev, f"{tag}_preview.png"))
    sheet([frame(360.0 * i / 16) for i in range(16)], 8).save(os.path.join(prev, f"{tag}_turntable_16.png"))
    save_gif([frame(360.0 * i / 24) for i in range(24)], os.path.join(prev, f"{tag}_turntable.gif"), ms=110)
    if leaf_pts is not None:
        leaves_vs_art(out_dir, views, leaf_pts)


def leaves_vs_art(out_dir, views, leaf_pts):
    prev = os.path.join(out_dir, "Preview")
    font = ImageFont.load_default(size=20)
    tiles = []
    for view, kind in (("front", "frontback"), ("side", "side"), ("back", "frontback")):
        v = views[view]
        m = analyze.measure(v, kind)
        im = Image.fromarray(v.rgb.astype(np.uint8)).convert("RGBA")
        msk, img = leaves.ortho(leaf_pts, view, size=(v.w, v.h), origin=(m.axis, m.ground))
        over = np.zeros((v.h, v.w, 4), np.uint8)
        ring = msk & ~nd.binary_erosion(msk, iterations=2)
        over[ring] = (225, 30, 30, 255)
        im.alpha_composite(Image.fromarray(over))
        ImageDraw.Draw(im).text((6, 6), view, fill=(30, 30, 30, 255), font=font)
        tiles.append(im)
    W = sum(t.width for t in tiles) + 10 * len(tiles)
    H = max(t.height for t in tiles)
    out = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    x = 0
    for t in tiles:
        out.alpha_composite(t, (x, H - t.height))
        x += t.width + 10
    out.save(os.path.join(prev, "leaves_vs_art.png"))


def face_vs_art(out_dir, char_cfg, views, squash_front=0.15):
    """Face check: the stack seen straight from the front / side next to the art."""
    pal = {k: np.array(v, float) for k, v in char_cfg["palette_rgb"].items()}
    items = load_items(out_dir, char_cfg, parts=("Body", "Face"))
    sp = char_cfg["layer_spacing_px"]
    tiles = []
    for yaw, view in ((0, "front"), (-90, "side")):
        f = lit.lit_items(items, sp, yaw, pal, squash_front, 1.0, outline_px=2)
        a = np.asarray(f)
        ys, xs = np.nonzero(a[..., 3] > 0)
        f = f.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        v = views[view]
        art = Image.fromarray(v.rgb.astype(np.uint8)).convert("RGBA")
        art = art.resize((int(art.width * f.height / art.height * 1.1), int(f.height * 1.1)))
        tiles += [art, f]
    W = sum(t.width for t in tiles) + 20 * len(tiles)
    H = max(t.height for t in tiles)
    out = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    x = 0
    for t in tiles:
        out.alpha_composite(t, (x, H - t.height))
        x += t.width + 20
    out.save(os.path.join(out_dir, "Preview", "face_vs_art.png"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sprite_stack.body3d.parts")
    ap.add_argument("--input", default="input/views_apose.png")
    ap.add_argument("--order", default="front,side,back,top")
    ap.add_argument("--out", default="output/body3d")
    ap.add_argument("--roll", type=float, default=80.0, help="leaf blade roll about the midrib (deg)")
    ap.add_argument("--parts", default="leaves,face", help="parts to (re)export; others are reused")
    a = ap.parse_args(argv)
    views = load_views(a.input, order=a.order)
    todo = a.parts.split(",")
    cfgs, pts = [], None
    if "leaves" in todo:
        lcfg, pts = export_leaves(a.out, leaves.LeafParams(roll_deg=a.roll))
    else:
        lcfg = json.load(open(os.path.join(a.out, "Leaves", "stack_config.json")))
    lcfg.setdefault("layer_span", 1.0)
    cfgs.append(lcfg)
    from . import face
    if "face" in todo:
        fcfg, _ = face.export_face(a.out, views)
    else:
        fcfg = json.load(open(os.path.join(a.out, "Face", "stack_config.json")))
    cfgs.append(fcfg)
    ccfg = character_config(a.out, cfgs)
    previews(a.out, ccfg, views, pts)
    face_vs_art(a.out, ccfg, views)
    print(f"leaves: {lcfg['layers']} layers; face: {fcfg['layers']} layers "
          f"(index {fcfg['layer_index'][0]}..{fcfg['layer_index'][-1]}); "
          f"character draw entries: {len(ccfg['draw'])}")


if __name__ == "__main__":
    main()
