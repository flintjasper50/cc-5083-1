"""Synthetic scenes and the forward rolling-shutter imager used for verification."""
import numpy as np

from . import lie
from .corrector import bilinear_sample


def scene_grid(width, height, spacing=40, line_w=2, dot_every=2, dot_r=3):
    """White background, gray grid, red control points at every `dot_every`-th intersection."""
    img = np.ones((height, width, 3))
    gray = 0.25
    xs = list(range(spacing, width, spacing))
    ys = list(range(spacing, height, spacing))
    for x in xs:
        img[:, max(0, x - line_w // 2):x - line_w // 2 + line_w] = gray
    for y in ys:
        img[max(0, y - line_w // 2):y - line_w // 2 + line_w, :] = gray
    points = []
    for x in xs[::dot_every]:
        for y in ys[::dot_every]:
            points.append((float(x), float(y)))
    for (x, y) in points:
        xi, yi = int(round(x)), int(round(y))
        yy, xx = np.ogrid[:height, :width]
        m = (xx - xi) ** 2 + (yy - yi) ** 2 <= dot_r ** 2
        img[m] = (0.85, 0.10, 0.10)
    return img, points


def scene_lines(width, height, spacing=80, bar_w=3):
    """Vertical bars (slant under yaw motion), one diagonal, one horizontal bar."""
    img = np.ones((height, width, 3))
    dark = 0.15
    bars = list(range(spacing, width, spacing))
    for x in bars:
        img[:, max(0, x - bar_w // 2):x - bar_w // 2 + bar_w] = dark
    yy, xx = np.mgrid[:height, :width]
    diag = np.abs((xx - 40) * (40 - 440) - (yy - 440) * (600 - 40)) / np.hypot(600 - 40, 40 - 440)
    img[diag < 1.2] = 0.45
    y0 = height // 2
    img[max(0, y0 - 1):y0 + 2, :] = 0.6
    return img, bars


def forward_render(ref_img, K, readout, track, q_ref, t_ref, plane=None, mover=None):
    """Render the rolling-shutter input image from a reference (global-shutter at t_ref) image.

    Each input line r is exposed at readout.line_time(r) with pose track(t).
    plane: dict(normal, dist, center0, velocity) — plane n^T X = d in reference frame,
           camera center C(t) = center0 + velocity*(t - t_ref).
    mover: dict(x, y, w, h, vx, vy) — independently moving rect (px, px/s in reference coords).
    Returns (img_in, fmap_uv (H,W,2), valid (H,W,bool)).
    """
    H, W = readout.height, readout.width
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    r = readout.readout_index(uu, vv)
    t = readout.line_time(r)
    rays = K.rays(W, H)
    q_t, outside = track.interp(t.ravel())
    q_t = q_t.reshape(H, W, 4)
    outside = outside.reshape(H, W)

    if plane is None:
        world = lie.quat_rotate(lie.quat_conj(q_t), rays)
        ref_cam = lie.quat_rotate(q_ref, world)
        geo_ok = np.ones((H, W), dtype=bool)
    else:
        n = np.asarray(plane["normal"], dtype=np.float64)
        n = n / np.linalg.norm(n)
        n_w = lie.quat_rotate(lie.quat_conj(q_ref), n)  # plane normal into world frame
        d = float(plane["dist"])
        C = np.asarray(plane["center0"], dtype=np.float64) \
            + (t - t_ref)[..., None] * np.asarray(plane["velocity"], dtype=np.float64)
        wdir = lie.quat_rotate(lie.quat_conj(q_t), rays)
        den = np.sum(wdir * n_w, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = np.where(np.abs(den) > 1e-12, (d - np.sum(C * n_w, axis=-1)) / den, np.nan)
        X = C + mu[..., None] * wdir
        ref_cam = lie.quat_rotate(q_ref, X)
        geo_ok = (den > 1e-12) & (mu > 0)

    uv_ref, z = K.project(ref_cam.reshape(-1, 3))
    uv_ref = uv_ref.reshape(H, W, 2)
    z = z.reshape(H, W)

    uv_sample = uv_ref.copy()
    if mover is not None:
        o = (t - t_ref)[..., None] * np.array([mover["vx"], mover["vy"]], dtype=np.float64)
        rel = uv_ref - o
        inside = (rel[..., 0] >= mover["x"]) & (rel[..., 0] < mover["x"] + mover["w"]) \
            & (rel[..., 1] >= mover["y"]) & (rel[..., 1] < mover["y"] + mover["h"])
        uv_sample[inside] = rel[inside]

    img_in = bilinear_sample(ref_img, uv_sample)
    in_bounds = (uv_sample[..., 0] >= 0) & (uv_sample[..., 0] <= W - 1) \
        & (uv_sample[..., 1] >= 0) & (uv_sample[..., 1] <= H - 1)
    valid = geo_ok & (z > 1e-9) & (~outside) & in_bounds
    img_in[~valid] = 0.0
    return img_in, uv_ref, valid
