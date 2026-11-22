"""Cut the voxel model into sprite-stack layers and do slice-strip I/O.

A layer is a top-down X x Z image (character front = BOTTOM edge of the image)
with a bottom height and a thickness in voxel units. Thick layers are what give
the "tiered" sprite-stack look: the body is cut into a few thick tiers while thin
parts (leaves) can keep 1-voxel layers. Layers run left -> right in the strip.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Stack:
    occ: np.ndarray        # bool [X, L, Z]
    rgb: np.ndarray        # uint8 [X, L, Z, 3]
    bottom: np.ndarray     # float [L]  bottom of each layer (voxels)
    thick: np.ndarray      # float [L]  thickness of each layer (voxels)
    src_y: list            # voxel rows each layer was cut from (for re-colouring)

    @property
    def L(self) -> int:
        return self.occ.shape[1]

    @property
    def top(self) -> float:
        return float(self.bottom[-1] + self.thick[-1])


def plan_layers(Y: int, layers: int, top_from: int | None = None, top_layers: int = 0):
    """Split [0, Y) into bands. The body [0, top_from) gets `layers` bands;
    the appendage [top_from, Y) gets `top_layers` bands (0 = one per voxel)."""
    def split(a, b, n):
        if b <= a:
            return []
        n = (b - a) if n <= 0 else min(n, b - a)
        edges = np.unique(np.round(np.linspace(a, b, n + 1)).astype(int))
        return list(zip(edges[:-1], edges[1:]))

    if top_from is None or top_from >= Y:
        return split(0, Y, layers)
    return split(0, top_from, layers) + split(top_from, Y, top_layers)


def cut(occ: np.ndarray, rgb: np.ndarray, bands, salient: float = 45.0,
        ground_skip: int = 2) -> Stack:
    """Each band becomes one layer: its shape is the band's middle cross-section.

    A thick layer shows its outer rim as the tier's side wall, so the rim must
    carry what the real surface shows over the whole band height. For every
    voxel row in the band we project that row's surface colours onto the layer
    (nearest surface pixel), then per pixel keep the row whose colour stands out
    most from the band's typical colour (eyes, cheeks, line art), falling back
    to the middle row. Small features thus survive between tiers.
    """
    from scipy import ndimage as nd
    X, _, Z = occ.shape
    L = len(bands)
    o = np.zeros((X, L, Z), bool)
    c = np.zeros((X, L, Z, 3), np.uint8)
    src = []
    for i, (a, b) in enumerate(bands):
        mid = (a + b - 1) // 2
        src.append(list(range(a, b)))
        o[:, i] = occ[:, mid]
        cands, used = [], []
        for y in range(a, b):
            if y < ground_skip and b > ground_skip:
                continue            # ground contact rows carry the drawing's bottom outline
            m = occ[:, y]
            if not m.any():
                continue
            surf = m & ~nd.binary_erosion(m)
            _, (ix, iz) = nd.distance_transform_edt(~surf, return_indices=True)
            cands.append(rgb[:, y][ix, iz].astype(np.float32))
            used.append(y)
        if not cands:
            continue
        cands = np.stack(cands, 0)                                # t, X, Z, 3
        midc = cands[int(np.argmin(np.abs(np.array(used) - mid)))]
        if len(cands) > 1:
            vals = cands[:, o[:, i]]
            med = np.median(vals.reshape(-1, 3), axis=0) if vals.size else midc.mean((0, 1))
            d = np.linalg.norm(cands - med, axis=-1)                # t, X, Z
            k = np.argmax(d, axis=0)
            pick = np.take_along_axis(cands, k[None, ..., None], 0)[0]
            use = (np.take_along_axis(d, k[None], 0)[0] > salient)[..., None]
            col = np.where(use, pick, midc)
        else:
            col = midc
        # interior of the layer (seen from above on the tier's top) keeps the mid colour
        rim = o[:, i] & ~nd.binary_erosion(o[:, i], iterations=2)
        col = np.where(rim[..., None], col, rgb[:, mid].astype(np.float32))
        c[:, i] = np.clip(col, 0, 255).astype(np.uint8)
    bottom = np.array([a for a, _ in bands], float)
    thick = np.array([b - a for a, b in bands], float)
    return Stack(o, c, bottom, thick, src)


def layer_rgba(occ: np.ndarray, rgb: np.ndarray, i: int) -> np.ndarray:
    """[Z rows (back..front), X cols, 4]"""
    a = np.zeros((occ.shape[2], occ.shape[0], 4), np.uint8)
    a[..., :3] = rgb[:, i].transpose(1, 0, 2)[::-1]
    a[..., 3] = occ[:, i].T[::-1] * 255
    return a


def write_strip(st: Stack, path: str, meta: dict, preview_scale: int = 8) -> None:
    X, L, Z = st.occ.shape
    strip = np.concatenate([layer_rgba(st.occ, st.rgb, i) for i in range(L)], axis=1)
    Image.fromarray(strip).save(path)
    Image.fromarray(strip).resize((strip.shape[1] * preview_scale, strip.shape[0] * preview_scale),
                                  Image.NEAREST).save(path.replace(".png", f"_x{preview_scale}.png"))
    info = {"layers": L, "layer_width": X, "layer_height": Z,
            "order": "bottom->top, left->right", "front": "bottom edge of each layer",
            "pivot": [X / 2, Z / 2],
            "layer_bottom": [float(v) for v in st.bottom],
            "layer_thickness": [float(v) for v in st.thick],
            "units": "layer_bottom/thickness in voxels = layer pixels", **meta}
    with open(path.replace(".png", ".json"), "w") as fh:
        json.dump(info, fh, indent=2)


def read_strip(path: str) -> Stack:
    with open(path.replace(".png", ".json")) as fh:
        info = json.load(fh)
    L = info["layers"]
    a = np.asarray(Image.open(path).convert("RGBA"))
    Z, total = a.shape[:2]
    X = total // L
    occ = np.zeros((X, L, Z), bool)
    rgb = np.zeros((X, L, Z, 3), np.uint8)
    for i in range(L):
        l = a[:, i * X:(i + 1) * X][::-1].transpose(1, 0, 2)
        occ[:, i] = l[..., 3] > 127
        rgb[:, i] = l[..., :3]
    return Stack(occ, rgb, np.array(info["layer_bottom"]), np.array(info["layer_thickness"]), [])
