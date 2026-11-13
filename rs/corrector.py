"""Inverse sampling map construction for rolling-shutter correction.

For every output pixel (reference time t_ref) we solve the exposure time t at which
the rolling-shutter camera actually imaged that ray, by fixed-point iteration:
project the ray into the input image with pose(t), read off the line index, and
update t from the line time. The result is a per-pixel map, not a single homography.
"""
from dataclasses import dataclass, field

import numpy as np

REASON_BOUNDS = 1    # sample falls outside the input image
REASON_POSE = 2      # exposure time outside pose-track coverage
REASON_BEHIND = 4    # point behind camera / ray parallel to plane
REASON_FOLD = 8      # local folding of the mapping (Jacobian det <= 0)
REASON_MOVING = 16   # user-masked independently moving object (unsupported)

REASON_NAMES = {
    REASON_BOUNDS: "out_of_bounds",
    REASON_POSE: "pose_coverage",
    REASON_BEHIND: "behind_camera",
    REASON_FOLD: "folding",
    REASON_MOVING: "independent_motion",
}


@dataclass
class InverseMap:
    uv: np.ndarray            # (H,W,2) input-image sample coordinates
    reasons: np.ndarray       # (H,W) uint8 bitmask of REASON_*
    row_times: np.ndarray     # (n_lines,) exposure time of each readout line (s)
    iterations: int
    max_dt_residual: float    # seconds
    pose_report: dict = field(default_factory=dict)

    @property
    def valid(self):
        return self.reasons == 0

    def reason_counts(self):
        return {name: int(np.count_nonzero(self.reasons & bit)) for bit, name in REASON_NAMES.items()}


def detect_folding(map_uv):
    mu = map_uv[..., 0]
    mv = map_uv[..., 1]
    dmu_dx = np.gradient(mu, axis=1)
    dmu_dy = np.gradient(mu, axis=0)
    dmv_dx = np.gradient(mv, axis=1)
    dmv_dy = np.gradient(mv, axis=0)
    det = dmu_dx * dmv_dy - dmu_dy * dmv_dx
    return det <= 0.0


def build_inverse_map(K, readout, scene, t_ref, track, moving_mask=None,
                      max_iter=60, tol=1e-9, fold_check=True):
    H, W = readout.height, readout.width
    rays = K.rays(W, H).reshape(-1, 3)
    t = np.full(rays.shape[0], float(t_ref))
    residual = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        uv, z, in_cov = scene.to_input(rays, t)
        r = readout.readout_index(uv[:, 0], uv[:, 1])
        t_new = readout.line_time(r)
        residual = float(np.max(np.abs(t_new - t)))
        t = t_new
        if residual < tol:
            break
    uv, z, in_cov = scene.to_input(rays, t)
    uv = uv.reshape(H, W, 2)
    z = z.reshape(H, W)
    in_cov = in_cov.reshape(H, W)

    reasons = np.zeros((H, W), dtype=np.uint8)
    in_bounds = (uv[..., 0] >= 0.0) & (uv[..., 0] <= W - 1) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= H - 1)
    reasons[~in_bounds] |= REASON_BOUNDS
    reasons[~(z > 1e-9)] |= REASON_BEHIND
    reasons[~in_cov] |= REASON_POSE
    if fold_check:
        reasons[detect_folding(uv)] |= REASON_FOLD
    if moving_mask is not None:
        reasons[np.asarray(moving_mask, dtype=bool)] |= REASON_MOVING

    t_lo, t_hi = readout.time_range()
    pose_report = track.adequacy(t_lo, t_hi)
    if not (track.times[0] <= t_ref <= track.times[-1]):
        pose_report["ok"] = False
        pose_report["problems"].append({
            "type": "t_ref_outside",
            "t_ref_s": float(t_ref),
            "msg": "参考时刻超出姿态轨迹覆盖范围",
        })

    row_times = readout.line_time(np.arange(readout.n_lines))
    return InverseMap(uv=uv, reasons=reasons, row_times=row_times,
                      iterations=it, max_dt_residual=residual, pose_report=pose_report)


def bilinear_sample(img, uv):
    """Sample img (H,W) or (H,W,C) at float coordinates uv (...,2). Out-of-range is clamped;
    callers must mask invalid pixels themselves."""
    img = np.asarray(img, dtype=np.float64)
    H, W = img.shape[:2]
    uv = np.asarray(uv, dtype=np.float64)
    flat = uv.reshape(-1, 2)
    x = np.clip(flat[:, 0], 0.0, W - 1.0)
    y = np.clip(flat[:, 1], 0.0, H - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    wx = x - x0
    wy = y - y0
    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]
    wa = (1.0 - wx) * (1.0 - wy)
    wb = wx * (1.0 - wy)
    wc = (1.0 - wx) * wy
    wd = wx * wy
    if img.ndim == 3:
        wa, wb, wc, wd = (w[:, None] for w in (wa, wb, wc, wd))
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out.reshape(uv.shape[:-1] + img.shape[2:])


def apply_inverse_map(img_in, inv):
    """Sample the input image through the inverse map; invalid pixels become 0 (never inpainted)."""
    out = bilinear_sample(img_in, inv.uv)
    out[~inv.valid] = 0.0
    return out
