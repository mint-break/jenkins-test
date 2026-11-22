"""Preview renderer for smooth layer stacks (premultiplied alpha, sub-layer fill)."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage as nd


def _premul(img: np.ndarray) -> Image.Image:
    a = img.astype(np.float32)
    a[..., :3] *= a[..., 3:4] / 255
    return Image.fromarray(np.round(a).astype(np.uint8), "RGBa")


def stack_frame(layers, spacing: float, yaw: float, squash: float = 0.55, out_scale: float = 0.5,
                outline: tuple | None = (40, 80, 62), outline_px: int = 2, fill: bool = True,
                layer_offsets=None) -> Image.Image:
    """layers: list of RGBA uint8 arrays (same size, pivot at the centre), bottom first.
    Each layer is rotated about the canvas centre, squashed vertically, and lifted by
    `spacing` px per layer; with fill=True it is also drawn at every pixel in between so
    the stack has no gaps."""
    n = layers[0].shape[0]
    S = int(round(n * out_scale))
    sq_h = max(1, int(round(S * squash)))
    lift = spacing * out_scale
    H = int(math.ceil(sq_h + lift * len(layers))) + 8
    canvas = np.zeros((H, S, 4), np.float32)          # premultiplied RGBA, 0..1
    base_y = H - 4 - sq_h
    for i, L in enumerate(layers):
        if L[..., 3].max() == 0:
            continue
        im = _premul(L).resize((S, S), Image.BILINEAR)
        im = im.rotate(yaw, resample=Image.BILINEAR)
        im = np.asarray(im.resize((S, sq_h), Image.BILINEAR), np.float32) / 255
        y0 = base_y - (layer_offsets[i] if layer_offsets is not None else i * spacing) * out_scale
        copies = max(1, int(math.ceil(lift))) if fill else 1
        for k in range(copies):
            y = int(round(y0 - k * lift / copies))
            a, b = max(y, 0), min(y + sq_h, H)
            if b <= a:
                continue
            src = im[a - y:b - y]
            dst = canvas[a:b]
            canvas[a:b] = src + dst * (1 - src[..., 3:4])
    rgb = np.divide(canvas[..., :3], canvas[..., 3:4], out=np.zeros_like(canvas[..., :3]),
                    where=canvas[..., 3:4] > 1e-4)
    out = Image.fromarray(np.round(np.concatenate([rgb, canvas[..., 3:4]], -1) * 255).astype(np.uint8), "RGBA")
    if outline is not None:
        a = np.asarray(out).copy()
        from scipy.ndimage import binary_fill_holes
        m = binary_fill_holes(a[..., 3] > 96)
        ring = nd.binary_dilation(m, iterations=outline_px) & ~m
        a[ring] = list(outline) + [255]
        out = Image.fromarray(a)
    return out


def turntable(layers, spacing, n=16, **kw):
    return [stack_frame(layers, spacing, 360.0 * i / n, **kw) for i in range(n)]


def sheet(frames, cols=8, bg=None):
    w = max(f.width for f in frames)
    h = max(f.height for f in frames)
    rows = math.ceil(len(frames) / cols)
    out = Image.new("RGBA", (w * cols, h * rows), bg or (0, 0, 0, 0))
    for i, f in enumerate(frames):
        out.alpha_composite(f, ((i % cols) * w + (w - f.width) // 2, (i // cols) * h + h - f.height))
    return out


def save_gif(frames, path, bg=(34, 40, 38), ms=80):
    flat = []
    for f in frames:
        b = Image.new("RGB", f.size, bg)
        b.paste(f, mask=f.getchannel("A"))
        flat.append(b.quantize(255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    flat[0].save(path, save_all=True, append_images=flat[1:], duration=ms, loop=0, disposal=2)
