"""Sprite-stack turntable rendering (with optional normal lighting + voxel AO)."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage as nd

from .slices import Stack, cut, layer_rgba
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
    Lx = L[0] * math.cos(a) - L[2] * math.sin(a)
    Lz = L[0] * math.sin(a) + L[2] * math.cos(a)
    Lm = np.array([Lx, L[1], Lz], np.float32)
    shade = ambient + diffuse * np.clip(normals @ Lm, 0, 1)
    out = rgb.astype(np.float32) * (shade * ao)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _tint(a: np.ndarray, k: float) -> np.ndarray:
    b = a.copy()
    b[..., :3] = np.clip(b[..., :3].astype(np.float32) * k, 0, 255).astype(np.uint8)
    return b


def outline(img: Image.Image, k: float = 0.42) -> Image.Image:
    """1px outline outside the silhouette, coloured as a dark version of the edge."""
    a = np.asarray(img).copy()
    m = a[..., 3] > 0
    ring = nd.binary_dilation(m, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]]) & ~m
    _, (iy, ix) = nd.distance_transform_edt(~m, return_indices=True)
    a[ring, :3] = (a[iy, ix][ring, :3] * k).astype(np.uint8)
    a[ring, 3] = 255
    return Image.fromarray(a)


def render_stack(st: Stack, yaw_deg: float, scale: int = 4, squash: float = 0.6,
                 pixel: bool = True, ridge: float = 0.25, line: bool = True,
                 canvas=None, bg=(0, 0, 0, 0)) -> Image.Image:
    """Rotate every layer around the shared pivot, squash it vertically and draw it
    once per voxel of its thickness, bottom to top.

    pixel=True renders at 1 px per voxel and upscales the frame with nearest
    neighbour (chunky pixel-art look); otherwise layers are upscaled first.
    ridge darkens the lowest and brightens the highest copy of each thick layer,
    which draws the line between tiers.
    """
    X, L, Z = st.occ.shape
    s = 1 if pixel else scale
    diag = math.ceil(math.hypot(X, Z) * s) + 4
    H = st.top
    if canvas is None:
        canvas = (diag + 8, int(diag * squash + H * s) + 16)
        if pixel:
            canvas = (math.ceil(canvas[0]), math.ceil(canvas[1]))
    out = Image.new("RGBA", canvas, bg)
    base_x = canvas[0] / 2
    base_y = canvas[1] - 8 - diag * squash / 2
    for i in range(L):
        layer = layer_rgba(st.occ, st.rgb, i)
        if not layer[..., 3].any():
            continue
        t = max(1, int(round(st.thick[i])))
        variants = {}
        for k in range(t * s):
            tone = 1.0
            if t > 1 and ridge > 0:
                if k < s:
                    tone = 1.0 - ridge            # bottom copy: groove between tiers
                elif k >= (t - 1) * s:
                    tone = 1.0 + ridge * 0.4      # top copy: rim highlight
            if tone not in variants:
                im = Image.fromarray(_tint(layer, tone)).resize((X * s, Z * s), Image.NEAREST)
                pad = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
                ox, oy = (diag - X * s) // 2, (diag - Z * s) // 2
                pad.paste(im, (ox, oy))
                cx, cy = ox + X * s / 2, oy + Z * s / 2
                rot = pad.rotate(yaw_deg, resample=Image.NEAREST, center=(cx, cy),
                                 translate=(diag / 2 - cx, diag / 2 - cy))
                variants[tone] = rot.resize((diag, max(1, round(diag * squash))), Image.NEAREST)
            sq = variants[tone]
            px = round(base_x - diag / 2)
            py = round(base_y - sq.height / 2 - st.bottom[i] * s - k)
            out.alpha_composite(sq, (px, py))
    if line:
        out = outline(out)
    if pixel and scale > 1:
        out = out.resize((out.width * scale, out.height * scale), Image.NEAREST)
    return out


def turntable(occ, rgb, bands, n: int = 16, lit: bool = False, normal_sigma: float = 1.5, **kw):
    """Render n directions. Lighting is computed on the full voxel model and then
    cut into the same layers as the slice strip."""
    frames = []
    normals = ao = None
    if lit:
        normals = _smooth_normals(occ, normal_sigma)
        ao = ambient_occlusion(occ, radius=3)
    base = cut(occ, rgb, bands)
    for i in range(n):
        yaw = 360.0 * i / n
        st = cut(occ, lit_colours(occ, rgb, normals, ao, yaw), bands) if lit else base
        frames.append(render_stack(st, yaw, **kw))
    return frames


def _smooth_normals(occ: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Normals from the gradient of a Gaussian-blurred occupancy field
    (sigma=0: plain 26-neighbour normals)."""
    if sigma <= 0:
        return normals26(occ)
    g = nd.gaussian_filter(occ.astype(np.float32), sigma)
    n = -np.stack(np.gradient(g), -1) * occ[..., None]
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    return np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-6)


def sheet(frames, cols: int = 8) -> Image.Image:
    w = max(f.width for f in frames)
    h = max(f.height for f in frames)
    rows = math.ceil(len(frames) / cols)
    out = Image.new("RGBA", (w * cols, h * rows), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        out.alpha_composite(f, ((i % cols) * w + (w - f.width) // 2, (i // cols) * h + h - f.height))
    return out


def save_gif(frames, path: str, bg=(40, 44, 52), ms: int = 100) -> None:
    flat = []
    for f in frames:
        b = Image.new("RGB", f.size, bg)
        b.paste(f, mask=f.getchannel("A"))
        flat.append(b.quantize(255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    flat[0].save(path, save_all=True, append_images=flat[1:], duration=ms, loop=0, disposal=2)
