"""Greedy meshing of the voxel model -> OBJ (vertex colours) + a quick software render."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw


def greedy_quads(occ: np.ndarray, rgb: np.ndarray):
    """Yield (corners[4,3], normal[3], colour[3]) merging coplanar faces of equal colour."""
    cid = (rgb[..., 0].astype(np.int64) << 16) | (rgb[..., 1].astype(np.int64) << 8) | rgb[..., 2]
    cid = np.where(occ, cid, -1)
    dims = occ.shape
    quads = []
    for a in range(3):
        u, v = (a + 1) % 3, (a + 2) % 3
        for s in (1, -1):
            for i in range(dims[a]):
                sl = [slice(None)] * 3
                sl[a] = i
                cur = cid[tuple(sl)]
                j = i + s
                if 0 <= j < dims[a]:
                    sl[a] = j
                    nb = occ[tuple(sl)]
                else:
                    nb = np.zeros_like(cur, bool)
                # 2D array indexed by the remaining axes in increasing order
                face = np.where((cur >= 0) & ~nb, cur, -1)
                if u > v:  # make face indexed [u, v]
                    face = face.T
                face = face.copy()
                nu, nv = face.shape
                c = i + (1 if s > 0 else 0)
                for pu in range(nu):
                    pv = 0
                    while pv < nv:
                        col = face[pu, pv]
                        if col < 0:
                            pv += 1
                            continue
                        w = 1
                        while pv + w < nv and face[pu, pv + w] == col:
                            w += 1
                        h = 1
                        while pu + h < nu and np.all(face[pu + h, pv:pv + w] == col):
                            h += 1
                        face[pu:pu + h, pv:pv + w] = -1
                        corners = []
                        for du, dv in ((0, 0), (h, 0), (h, w), (0, w)):
                            p = [0, 0, 0]
                            p[a], p[u], p[v] = c, pu + du, pv + dv
                            corners.append(p)
                        if s < 0:
                            corners = corners[::-1]
                        n = [0, 0, 0]
                        n[a] = s
                        rgbc = ((col >> 16) & 255, (col >> 8) & 255, col & 255)
                        quads.append((np.array(corners, float), np.array(n), rgbc, h * w))
                        pv += w
    return quads


def write_obj(quads, path: str, shape) -> None:
    X, _, Z = shape
    with open(path, "w") as fh:
        fh.write("# sprite_stack voxel mesh (greedy). Y up, front faces -Z, 1 unit = 1 voxel\n")
        fh.write("# vertex colours: v x y z r g b\n")
        vi = 1
        faces = []
        for corners, _, col, _ in quads:
            for p in corners:
                fh.write(f"v {p[0] - X / 2:g} {p[1]:g} {p[2] - Z / 2:g} "
                         f"{col[0] / 255:.4f} {col[1] / 255:.4f} {col[2] / 255:.4f}\n")
            faces.append(f"f {vi} {vi + 1} {vi + 2} {vi + 3}\n")
            vi += 4
        fh.writelines(faces)


def exposed_face_count(occ: np.ndarray) -> int:
    p = np.pad(occ, 1)
    return int(sum((p & ~np.roll(p, sh, ax)).sum() for ax in range(3) for sh in (1, -1)))


def map_heights(quads, heights):
    """Turn layer-index y coordinates into heights (layer boundaries in voxels)."""
    h = np.asarray(heights, float)
    out = []
    for corners, n, col, area in quads:
        c = corners.copy()
        c[:, 1] = h[c[:, 1].astype(int)]
        out.append((c, n, col, area))
    return out


def render_quads(quads, X, Z, H, yaw_deg, scale=4, squash=0.6, pixel=True) -> Image.Image:
    """Same projection and canvas as render.render_stack, drawing mesh polygons."""
    from .render import outline
    s = 1 if pixel else scale
    diag = math.ceil(math.hypot(X, Z) * s) + 4
    size = (diag + 8, int(diag * squash + H * s) + 16)
    base_x = size[0] / 2
    base_y = size[1] - 8 - diag * squash / 2
    t = math.radians(yaw_deg)
    ct, st = math.cos(t), math.sin(t)
    ce = math.sqrt(max(0.0, 1 - squash ** 2))

    def proj(p):
        u, v = p[..., 0] - X / 2, Z / 2 - p[..., 2]
        ur, vr = u * ct + v * st, -u * st + v * ct
        sx = base_x + ur * s
        sy = base_y + vr * squash * s - p[..., 1] * s
        return sx, sy, vr * ce + p[..., 1] * squash

    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    items = []
    for corners, n, col, _ in quads:
        vrn = -n[0] * st + (-n[2]) * ct
        if vrn * ce + n[1] * squash <= 0:   # back-face culling
            continue
        sx, sy, dep = proj(corners)
        items.append((dep.mean(), list(zip(sx, sy)), col))
    for _, pts, col in sorted(items, key=lambda it: it[0]):
        d.polygon(pts, fill=tuple(int(v) for v in col) + (255,))
    img = outline(img)
    if pixel and scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    return img
