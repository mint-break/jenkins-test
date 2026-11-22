"""Sprite-stack turntable rendering (with optional normal lighting + voxel AO)."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage as nd

from .slices import layer_rgba
from .voxelize import normals26


def ambient_occlusion(occ: np.ndarray, radius: int = 2, strength: float = 1.1) -> np.ndarray:
    """Voxel AO: share of filled voxels in a (2r+1)^3 box. Flat surface ~0.5."""
    frac = nd.uniform_filter(occ.astype(np.float32), size=2 * radius + 1, mode="constant")
    return np.clip(1.0 - strength * (frac - 0.5), 0.55, 1.15)


def lit_colours(occ, rgb, normals, ao, yaw_deg: float, light=(-0.5, 0.75, -0.45),
                ambient=0.45, diffuse=0.65) -> np.ndarray:
    """Light is fixed relative to the camera, so it is rotated by -yaw in model space."""
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    a = math.radians(-yaw_deg)
    # rotation around Y matching the on-screen rotation of the layers
    Lx = L[0] * math.cos(a) - L[2] * math.sin(a)
    Lz = L[0] * math.sin(a) + L[2] * math.cos(a)
    Lm = np.array([Lx, L[1], Lz], np.float32)
    shade = ambient + diffuse * np.clip(normals @ Lm, 0, 1)
    out = rgb.astype(np.float32) * (shade * ao)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def render_stack(occ, rgb, yaw_deg: float, scale: int = 4, squash: float = 0.6,
                 step: float = 1.0, canvas=None, bg=(0, 0, 0, 0)) -> Image.Image:
    """Rotate each layer around the shared pivot, squash vertically, lift `step`
    voxel units per layer, draw bottom to top."""
    X, Y, Z = occ.shape
    diag = math.ceil(math.hypot(X, Z) * scale) + 4
    if canvas is None:
        canvas = (diag + 8, int(diag * squash + Y * step * scale) + 16)
    out = Image.new("RGBA", canvas, bg)
    base_x = canvas[0] / 2
    base_y = canvas[1] - 8 - diag * squash / 2
    for y in range(Y):
        layer = layer_rgba(occ, rgb, y)
        if not layer[..., 3].any():
            continue
        im = Image.fromarray(layer).resize((X * scale, Z * scale), Image.NEAREST)
        pad = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
        ox, oy = (diag - X * scale) // 2, (diag - Z * scale) // 2
        pad.paste(im, (ox, oy))
        cx, cy = ox + X * scale / 2, oy + Z * scale / 2
        # place pivot at the centre of the padded square, then rotate around it
        rot = pad.rotate(yaw_deg, resample=Image.NEAREST, center=(cx, cy),
                         translate=(diag / 2 - cx, diag / 2 - cy))
        sq = rot.resize((diag, max(1, round(diag * squash))), Image.NEAREST)
        px = round(base_x - diag / 2)
        py = round(base_y - sq.height / 2 - y * step * scale)
        out.alpha_composite(sq, (px, py))
    return out


def turntable(occ, rgb, n: int = 16, lit: bool = False, normal_sigma: float = 1.5, **kw):
    frames = []
    normals = ao = None
    if lit:
        normals = _smooth_normals(occ, normal_sigma)
        ao = ambient_occlusion(occ, radius=3)
    for i in range(n):
        yaw = 360.0 * i / n
        col = lit_colours(occ, rgb, normals, ao, yaw) if lit else rgb
        frames.append(render_stack(occ, col, yaw, **kw))
    return frames


def _smooth_normals(occ: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Normals from the gradient of a Gaussian-blurred occupancy field.

    Plain 26-neighbour normals (voxelize.normals26) only see one voxel around,
    so every 1-voxel layer step shows up as a stripe when lit; blurring first
    spreads the estimate over ~2 voxels.
    """
    if sigma <= 0:
        return normals26(occ)
    g = nd.gaussian_filter(occ.astype(np.float32), sigma)
    n = -np.stack(np.gradient(g), -1) * occ[..., None]
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    return np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-6)


def sheet(frames, cols: int = 8, label: bool = True) -> Image.Image:
    w, h = frames[0].size
    rows = math.ceil(len(frames) / cols)
    out = Image.new("RGBA", (w * cols, h * rows), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        out.alpha_composite(f, ((i % cols) * w, (i // cols) * h))
    return out


def save_gif(frames, path: str, bg=(40, 44, 52), ms: int = 100) -> None:
    flat = []
    for f in frames:
        b = Image.new("RGB", f.size, bg)
        b.paste(f, mask=f.getchannel("A"))
        flat.append(b.quantize(255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    flat[0].save(path, save_all=True, append_images=flat[1:], duration=ms, loop=0, disposal=2)
