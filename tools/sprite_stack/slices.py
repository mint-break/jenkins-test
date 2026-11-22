"""Slice strip I/O. Layer i (0 = ground) is an X x Z top-down image; the character's
front (z = 0) is at the BOTTOM of each layer image. Layers run left -> right."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from .voxelize import VoxelModel


def layer_rgba(occ: np.ndarray, rgb: np.ndarray, y: int) -> np.ndarray:
    """[Z rows (back..front), X cols, 4]"""
    a = np.zeros((occ.shape[2], occ.shape[0], 4), np.uint8)
    a[..., :3] = rgb[:, y].transpose(1, 0, 2)[::-1]
    a[..., 3] = occ[:, y].T[::-1] * 255
    return a


def write_strip(model: VoxelModel, path: str, preview_scale: int = 8) -> None:
    X, Y, Z = model.shape
    strip = np.concatenate([layer_rgba(model.occ, model.rgb, y) for y in range(Y)], axis=1)
    Image.fromarray(strip).save(path)
    big = Image.fromarray(strip).resize((strip.shape[1] * preview_scale,
                                         strip.shape[0] * preview_scale), Image.NEAREST)
    big.save(path.replace(".png", f"_x{preview_scale}.png"))
    meta = {"layers": Y, "layer_width": X, "layer_height": Z, "order": "bottom->top, left->right",
            "front": "bottom edge of each layer", "pivot": [X / 2, Z / 2], **model.meta}
    with open(path.replace(".png", ".json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def read_strip(path: str, layers: int):
    a = np.asarray(Image.open(path).convert("RGBA"))
    Z, total = a.shape[:2]
    X = total // layers
    occ = np.zeros((X, layers, Z), bool)
    rgb = np.zeros((X, layers, Z, 3), np.uint8)
    for y in range(layers):
        l = a[:, y * X:(y + 1) * X][::-1].transpose(1, 0, 2)
        occ[:, y] = l[..., 3] > 127
        rgb[:, y] = l[..., :3]
    return occ, rgb
