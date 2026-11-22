"""Load an orthographic view sheet, split it into views and extract silhouettes.

Coordinate convention used across the package (right-handed, Y up):
  X : character's left  (= viewer's right in the FRONT view)
  Y : up
  Z : backwards        (z = 0 is the front face, FRONT viewer sits at -Z)

Expected view orientation in the sheet:
  front : seen from -Z, upright
  back  : seen from +Z, upright (mirrored internally so that columns map to +X)
  side  : seen from -X, upright, character facing the viewer's LEFT (front = left)
  top   : seen from +Y, character's front pointing DOWN in the image
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as nd

VIEW_NAMES = ("top", "front", "side", "back")


@dataclass
class View:
    name: str
    rgb: np.ndarray   # float32 HxWx3, cropped to silhouette bbox
    mask: np.ndarray  # bool HxW
    part: np.ndarray | None = None  # bool HxW, pixels of a separable part (optional)

    @property
    def h(self) -> int:
        return self.mask.shape[0]

    @property
    def w(self) -> int:
        return self.mask.shape[1]


def _silhouette(cell: np.ndarray, bg_threshold: int, keep_ratio: float) -> np.ndarray:
    if cell.shape[2] == 4 and cell[..., 3].min() < 250:
        mask = cell[..., 3] > 127
    else:
        mask = cell[..., :3].max(-1) > bg_threshold
    lab, n = nd.label(mask)
    if n == 0:
        raise ValueError("empty view cell")
    sizes = nd.sum(mask, lab, range(1, n + 1))
    keep = np.flatnonzero(sizes >= sizes.max() * keep_ratio) + 1
    mask = np.isin(lab, keep)
    return nd.binary_fill_holes(mask)


def load_views(path: str, grid: str = "auto", order: str = ",".join(VIEW_NAMES),
               bg_threshold: int = 40, keep_ratio: float = 0.25) -> dict[str, View]:
    """Split the sheet into a COLSxROWS grid (row-major `order`) and crop each view.

    Labels / captions are dropped because only components that are at least
    `keep_ratio` of the largest component survive.
    """
    img = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32)
    H, W = img.shape[:2]
    names = [s.strip() for s in order.split(",")]
    if sorted(names) != sorted(VIEW_NAMES):
        raise ValueError(f"order must list {VIEW_NAMES} once each")
    if grid in ("auto", "blobs"):
        views = _load_blobs(img, names, bg_threshold, keep_ratio)
        if views is not None:
            return views
        if grid == "blobs":
            raise ValueError("could not find exactly 4 separate views in the sheet")
    if grid == "auto":
        cols, rows = (4, 1) if W / H >= 3 else (2, 2)
    else:
        cols, rows = (int(v) for v in grid.lower().split("x"))
    if len(names) != cols * rows:
        raise ValueError(f"order must have {cols * rows} entries for a {cols}x{rows} grid")

    views = {}
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        cell = img[r * H // rows:(r + 1) * H // rows, c * W // cols:(c + 1) * W // cols]
        mask = _silhouette(cell, bg_threshold, keep_ratio)
        ys, xs = np.nonzero(mask)
        sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        views[name] = View(name, cell[sl][..., :3].copy(), mask[sl].copy())
    return views


def _load_blobs(img, names, bg_threshold, keep_ratio):
    """Find the 4 views as the 4 largest separate blobs, whatever the layout.

    Blobs are ordered in reading order (rows by vertical centre, then x).
    Captions are smaller blobs and are ignored. Returns None if the sheet does
    not split into exactly 4 large blobs.
    """
    if img[..., 3].min() < 250:
        mask = img[..., 3] > 127
    else:
        mask = img[..., :3].max(-1) > bg_threshold
    # bridge 1-2 px gaps inside a view (thin outlines) but not the space between views
    closed = nd.binary_closing(mask, iterations=3)
    lab, n = nd.label(closed)
    if n < 4:
        return None
    sizes = nd.sum(closed, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    big = [k + 1 for k in order if sizes[k] >= sizes[order[0]] * keep_ratio]
    if len(big) != 4:
        return None
    boxes = nd.find_objects(lab)
    items = []
    for k in big:
        sl = boxes[k - 1]
        cy = (sl[0].start + sl[0].stop) / 2
        items.append((cy, sl[1].start, k, sl))
    items.sort()
    # group into rows: a new row starts when the centre jumps by > 1/3 of a view height
    rows, cur = [], [items[0]]
    for it in items[1:]:
        h = cur[-1][3][0].stop - cur[-1][3][0].start
        if abs(it[0] - cur[-1][0]) > h / 3:
            rows.append(cur)
            cur = [it]
        else:
            cur.append(it)
    rows.append(cur)
    ordered = [it for r in rows for it in sorted(r, key=lambda t: t[1])]
    views = {}
    for name, (_, _, k, sl) in zip(names, ordered):
        m = (lab[sl] == k) & mask[sl]
        m = nd.binary_fill_holes(m)
        ys, xs = np.nonzero(m)
        sub = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        views[name] = View(name, img[sl][sub][..., :3].copy(), m[sub].copy())
    return views


def resize_view(view: View, out_w: int, out_h: int, mirror: bool = False):
    """Area-downsample a view. Returns (mask bool[h,w], rgb float[h,w,3]).

    Colours are averaged premultiplied by coverage, so the background never
    bleeds into edge pixels. Pixels outside the silhouette get the colour of the
    nearest covered pixel (used when a voxel projects just outside a view).
    """
    def area(a: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(a.astype(np.float32), "F").resize((out_w, out_h), Image.BOX))

    m = view.mask.astype(np.float32)
    cov = area(m)
    rgb = np.stack([area(view.rgb[..., k] * m) for k in range(3)], -1)
    valid = cov > 0.05
    rgb[valid] /= cov[valid][:, None]
    _, (iy, ix) = nd.distance_transform_edt(~valid, return_indices=True)
    rgb = rgb[iy, ix]
    mask = cov >= 0.5
    if mirror:
        mask, rgb = mask[:, ::-1], rgb[:, ::-1]
    return mask, rgb


def detect_neck(view: View, lo: float = 0.08, hi: float = 0.45) -> float:
    """Fraction from the top of the narrowest row in [lo, hi] of the height.

    Used to find where a thin appendage (leaves, antenna, ...) meets the body.
    """
    widths = view.mask.sum(1).astype(float)
    a, b = int(view.h * lo), int(view.h * hi)
    widths = nd.uniform_filter1d(widths, 5)
    return (a + int(np.argmin(widths[a:b]))) / view.h


PART_PRESETS = {
    # warm / orange-brown pixels (bread, wood, skin...) on a cool-coloured body
    "warm": "(r > g + 15) & (r > b + 40)",
}


def segment_part(views: dict[str, "View"], rule: str, close_px: int = 8,
                 outline_px: int = 6) -> None:
    """Mark a colour-separable part (e.g. a held object) in every view.

    `rule` is a preset name or a numpy expression of r, g, b (0..255).
    Only the largest matching blob per view is kept, grown by `outline_px` to
    include the line art around it.
    """
    expr = PART_PRESETS.get(rule, rule)
    for v in views.values():
        r, g, b = (v.rgb[..., k] for k in range(3))
        m = eval(expr, {"np": np}, {"r": r, "g": g, "b": b}) & v.mask  # noqa: S307 (CLI input)
        m = nd.binary_fill_holes(nd.binary_closing(m, iterations=close_px)) & v.mask
        lab, n = nd.label(m)
        if n:
            sizes = nd.sum(m, lab, range(1, n + 1))
            m = lab == (np.argmax(sizes) + 1)
        # swallow the part's own dark outline so it is not mistaken for body
        v.part = nd.binary_dilation(m, iterations=outline_px) & v.mask


def body_span(v: "View") -> np.ndarray:
    """Estimated body silhouette with the part removed.

    Per image row the body spans from its first to its last body pixel. Where
    the part sits on the outline (the row's extreme pixel is a part pixel) the
    body edge is hidden, so it is interpolated from the nearest clean rows
    above and below (smoothness prior) instead of trusting the part's edge.
    """
    body = v.mask & ~v.part
    H, W = body.shape
    rows = np.arange(H)
    L = np.full(H, np.nan)
    R = np.full(H, np.nan)
    hidL = np.zeros(H, bool)
    hidR = np.zeros(H, bool)
    for r in rows:
        xs = np.flatnonzero(body[r])
        ms = np.flatnonzero(v.mask[r])
        if len(xs) == 0:
            continue
        L[r], R[r] = xs[0], xs[-1]
        hidL[r] = v.part[r, ms[0]]
        hidR[r] = v.part[r, ms[-1]]
    have = ~np.isnan(L)
    for E, hid in ((L, hidL), (R, hidR)):
        clean = have & ~hid
        if clean.sum() >= 2:
            fix = have & hid
            E[fix] = np.interp(rows[fix], rows[clean], E[clean])
    span = np.zeros_like(body)
    for r in rows[have]:
        span[r, int(round(min(L[r], R[r]))):int(round(max(L[r], R[r]))) + 1] = True
    return (span & v.mask) | nd.binary_fill_holes(body)


def protrusion_mask(v: "View") -> np.ndarray:
    """Part pixels that stick out past the (estimated) body outline."""
    if v.part is None:
        return np.zeros_like(v.mask)
    return v.part & ~body_span(v)


def resize_mask(mask: np.ndarray, out_w: int, out_h: int, mirror: bool = False) -> np.ndarray:
    cov = np.asarray(Image.fromarray(mask.astype(np.float32), "F").resize((out_w, out_h), Image.BOX))
    m = cov >= 0.5
    return m[:, ::-1] if mirror else m


def remove_part(views: dict[str, "View"], iters: int = 300, grow_px: int = 10,
                min_lum: float = 120) -> None:
    """Erase the segmented part from every view (e.g. "no sandwich" variant).

    The silhouette becomes the estimated body outline (protrusions dropped) and
    the pixels the part covered are repainted by harmonic in-filling from the
    surrounding body colours, so the carving and colouring see a bare body.
    """
    for v in views.values():
        if v.part is None or not v.part.any():
            continue
        mask = nd.binary_fill_holes(body_span(v))
        lab, n = nd.label(mask)
        if n > 1:
            mask = lab == (np.argmax(nd.sum(mask, lab, range(1, n + 1))) + 1)
        # grow past the part's line art, and fill only from bright body pixels
        hole = nd.binary_dilation(v.part, iterations=grow_px) & mask
        lum = v.rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        src = mask & ~hole & (lum > min_lum)
        _, (iy, ix) = nd.distance_transform_edt(~src, return_indices=True)
        rgb = v.rgb.copy()
        rgb[hole] = rgb[iy, ix][hole]          # nearest colour as a starting guess
        for _ in range(iters):                  # then relax towards a smooth fill
            blur = np.stack([nd.uniform_filter(rgb[..., k], 5) for k in range(3)], -1)
            rgb[hole] = blur[hole]
        v.rgb, v.mask, v.part = rgb, mask, None
