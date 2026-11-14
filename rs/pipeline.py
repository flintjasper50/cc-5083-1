"""End-to-end pipeline shared by the CLI demo and the web UI.

Forward imaging uses a dense analytic pose track (ground truth); correction only
sees gyro-rate samples of that track, so interpolation error is exercised honestly.
"""
import numpy as np

from . import corrector, synth
from .camera import Intrinsics
from .model import PlaneScene, ReadoutSpec, RotationScene
from .params import Params
from .poses import PoseTrack

TRUE_OFFSET_S = 0.002  # hidden sensor-vs-pose clock offset used by the synthesizer

OMEGAS = {
    "static": (0.0, 0.0, 0.0),  # zero angular velocity: correction must be a no-op
    "yaw": (0.0, 0.5, 0.0),
    "roll": (0.0, 0.0, 0.35),
    "pitch": (0.3, 0.0, 0.0),
    "mixed": (0.12, 0.5, 0.08),
    "extreme": (15.0, 8.0, 0.0),  # fast enough to fold the mapping
}

PLANE = {"normal": (0.0, 0.0, 1.0), "dist": 2.0,
         "center0": (0.0, 0.0, 0.0), "velocity": (0.5, 0.2, 0.0)}

MOVER = {"x": 400.0, "y": 120.0, "w": 110.0, "h": 110.0, "vx": -900.0, "vy": 250.0}

DEFAULT_CFG = {
    "scene": "grid",
    "direction": "top_to_bottom",
    "readout_ms": 32.0,
    "t_offset_ms": 2.0,
    "t_ref_ms": 16.0,
    "gyro_hz": 500.0,
    "motion": "mixed",
    "model": "rotation",
    "moving": False,
    "width": 640,
    "height": 480,
}


def _erode(mask, r=2):
    out = mask.copy()
    H, W = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ys = slice(max(0, dy), H + min(0, dy))
            xs = slice(max(0, dx), W + min(0, dx))
            src = np.zeros_like(mask)
            src[ys, xs] = mask[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
            out &= src
    return out


def fit_homography(src, dst):
    """Normalized DLT: src,dst (N,2) -> 3x3 homography mapping src to dst."""
    def norm(pts):
        c = pts.mean(axis=0)
        s = np.sqrt(2.0) / np.mean(np.linalg.norm(pts - c, axis=1))
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    Ts, Td = norm(src), norm(dst)
    sn = (Ts @ np.column_stack([src, np.ones(len(src))]).T).T
    dn = (Td @ np.column_stack([dst, np.ones(len(dst))]).T).T
    A = []
    for (x, y, _), (u, v, _) in zip(sn, dn):
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    _, _, vt = np.linalg.svd(np.asarray(A))
    Hn = vt[-1].reshape(3, 3)
    return np.linalg.inv(Td) @ Hn @ Ts


def _line_wobble(img, bars, half=4, thresh=0.4):
    """Std (px) of per-row dark-centroid positions of vertical bars; 0 = perfectly straight."""
    gray = img.mean(axis=-1)
    H = gray.shape[0]
    wobbles = []
    for x in bars:
        x0, x1 = max(0, x - half), min(gray.shape[1], x + half + 1)
        band = gray[:, x0:x1]
        dark = band < thresh
        rows = np.where(dark.sum(axis=1) >= 2)[0]
        if rows.size < H * 0.5:
            continue
        cent = []
        for y in rows:
            cols = np.where(dark[y])[0]
            cent.append(x0 + cols.mean())
        wobbles.append(float(np.std(cent)))
    return float(np.mean(wobbles)) if wobbles else None


def run_photo(img, params, track):
    """Photo correction chain shared by the CLI and the web UI.

    img: float HxWx3 in [0,1]; params: Params (camera/readout/t_ref/scene);
    track: PoseTrack on the same clock as the readout times.
    Returns (corrected, inv, stats).
    """
    cam = params.camera
    K = Intrinsics(cam["fx_px"], cam["fy_px"], cam["cx_px"], cam["cy_px"])
    if img.shape[1] != cam["width_px"] or img.shape[0] != cam["height_px"]:
        raise ValueError(
            f"照片尺寸 {img.shape[1]}x{img.shape[0]} 与参数 "
            f"{cam['width_px']}x{cam['height_px']} 不一致")
    rd = params.readout
    readout = ReadoutSpec(cam["width_px"], cam["height_px"], rd["readout_time_s"],
                          rd["direction"], rd.get("t0_s", 0.0), rd.get("t_offset_s", 0.0))
    q_ref, _ = track.interp(np.array(params.t_ref))
    sc = params.scene
    if sc["model"] == "plane":
        scene = PlaneScene(track, q_ref, K, sc["normal"], sc["dist_m"],
                           sc["center0_m"], sc["velocity_m_s"], params.t_ref)
    else:
        scene = RotationScene(track, q_ref, K)
    inv = corrector.build_inverse_map(K, readout, scene, params.t_ref, track)
    corrected = corrector.apply_inverse_map(img, inv)
    stats = {"version": params.version(),
             "valid_fraction": float(inv.valid.mean()),
             "reason_counts": inv.reason_counts(),
             "pose_report": inv.pose_report,
             "iterations": inv.iterations,
             "max_dt_residual_s": inv.max_dt_residual}
    return corrected, inv, stats


def run_synthetic(cfg):
    cfg = {**DEFAULT_CFG, **cfg}
    W, H = int(cfg["width"]), int(cfg["height"])
    K = Intrinsics(fx=560.0, fy=560.0, cx=(W - 1) / 2.0, cy=(H - 1) / 2.0)
    readout_s = float(cfg["readout_ms"]) / 1000.0
    t_ref = float(cfg["t_ref_ms"]) / 1000.0
    omega = OMEGAS[cfg["motion"]]
    use_plane = cfg["scene"] == "plane"

    # ground truth: dense analytic track, sensor clock offset by TRUE_OFFSET_S
    t_lo, t_hi = -0.025, readout_s + 0.025
    dense_t = np.arange(t_lo, t_hi + 1e-6, 1e-4)
    track_true = PoseTrack.from_body_rate(dense_t, omega)
    readout_true = ReadoutSpec(W, H, readout_s, cfg["direction"], t0=0.0, t_offset=TRUE_OFFSET_S)

    # correction side: gyro samples only (endpoint appended so the span is always covered)
    gyro_t = np.arange(t_lo, t_hi, 1.0 / float(cfg["gyro_hz"]))
    if gyro_t.size == 0 or gyro_t[-1] < t_hi - 1e-12:
        gyro_t = np.append(gyro_t, t_hi)
    track_corr = PoseTrack.from_body_rate(gyro_t, omega)
    readout_corr = ReadoutSpec(W, H, readout_s, cfg["direction"], t0=0.0,
                               t_offset=float(cfg["t_offset_ms"]) / 1000.0)

    q_ref_true, _ = track_true.interp(np.array(t_ref))
    q_ref_corr, _ = track_corr.interp(np.array(t_ref))

    plane = PLANE if use_plane else None
    if cfg["scene"] == "lines":
        ref_img, bars = synth.scene_lines(W, H)
        control_points = [(float(x), float(y)) for x in bars for y in range(60, H, 120)]
    else:
        ref_img, control_points = synth.scene_grid(W, H)
        bars = None

    mover = None
    if cfg["moving"]:
        mover = {"x": 0.60 * W, "y": 0.25 * H, "w": 0.18 * W, "h": 0.30 * H,
                 "vx": MOVER["vx"], "vy": MOVER["vy"]}
    img_in, fmap, fvalid = synth.forward_render(
        ref_img, K, readout_true, track_true, q_ref_true, t_ref, plane=plane, mover=mover)

    moving_mask = None
    if mover is not None:
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        moving_mask = (uu >= mover["x"]) & (uu < mover["x"] + mover["w"]) \
            & (vv >= mover["y"]) & (vv < mover["y"] + mover["h"])

    if cfg["model"] == "plane" and use_plane:
        scene_model = PlaneScene(track_corr, q_ref_corr, K, plane["normal"], plane["dist"],
                                 plane["center0"], plane["velocity"], t_ref)
    else:
        scene_model = RotationScene(track_corr, q_ref_corr, K)

    inv = corrector.build_inverse_map(K, readout_corr, scene_model, t_ref, track_corr,
                                      moving_mask=moving_mask)
    corrected = corrector.apply_inverse_map(img_in, inv)

    eval_mask = _erode(inv.valid & fvalid, 2)
    if moving_mask is not None:
        eval_mask &= ~moving_mask
    diff = np.abs(corrected - ref_img).sum(axis=-1)
    mae = float(diff[eval_mask].mean()) if eval_mask.any() else float("nan")
    p95 = float(np.percentile(diff[eval_mask], 95)) if eval_mask.any() else float("nan")
    mae_uncorrected = float(np.abs(img_in - ref_img).sum(axis=-1)[eval_mask].mean()) \
        if eval_mask.any() else float("nan")

    # flat-region MAE isolates geometric error from double-resampling blur on edges
    gray_ref = ref_img.mean(axis=-1)
    gx, gy = np.gradient(gray_ref)
    flat = (np.hypot(gx, gy) < 0.02) & eval_mask
    mae_flat = float(diff[flat].mean()) if flat.any() else float("nan")

    # round-trip: forward map sampled at the corrected position must return the point itself
    cp_err = []
    for (x, y) in control_points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H and inv.valid[yi, xi] and fvalid[yi, xi]:
            q = inv.uv[yi, xi]
            back = corrector.bilinear_sample(fmap, q[None, :])[0]
            cp_err.append(np.linalg.norm(back - np.array([x, y])))
    cp_stats = {"n": len(cp_err),
                "mean_px": float(np.mean(cp_err)) if cp_err else None,
                "max_px": float(np.max(cp_err)) if cp_err else None}

    # single-global-homography baseline: the shortcut this corrector must beat
    step = 10
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    ok = fvalid[ys, xs] & inv.valid[ys, xs]
    src = np.column_stack([xs[ok], ys[ok]])
    dst = fmap[ys, xs][ok]
    mae_global = None
    if len(src) >= 8:
        Hg = fit_homography(src, dst)
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        pts = np.column_stack([uu.ravel(), vv.ravel(), np.ones(H * W)])
        proj = (Hg @ pts.T).T
        uv_g = (proj[:, :2] / proj[:, 2:3]).reshape(H, W, 2)
        img_g = corrector.bilinear_sample(img_in, uv_g)
        img_g[~inv.valid] = 0.0
        mae_global = float(np.abs(img_g - ref_img).sum(axis=-1)[eval_mask].mean())

    wobble = None
    if bars is not None:
        wobble = {"input_px": _line_wobble(img_in, bars),
                  "corrected_px": _line_wobble(np.where(inv.valid[..., None], corrected, 1.0), bars)}

    scene_desc = ({"model": "plane",
                   "normal": list(plane["normal"]), "dist_m": plane["dist"],
                   "center0_m": list(plane["center0"]),
                   "velocity_m_s": list(plane["velocity"])}
                  if cfg["model"] == "plane" and use_plane else {"model": "rotation"})
    if mover is not None:
        scene_desc["moving_object"] = {k: round(float(v), 6) for k, v in mover.items()}

    params = Params(
        camera={"fx_px": K.fx, "fy_px": K.fy, "cx_px": K.cx, "cy_px": K.cy,
                "width_px": W, "height_px": H},
        readout={"readout_time_s": readout_s, "direction": cfg["direction"],
                 "t0_s": 0.0, "t_offset_s": readout_corr.t_offset},
        t_ref=t_ref,
        scene=scene_desc,
        motion={"type": "body_rate", "omega_rad_s": list(omega), "gyro_hz": float(cfg["gyro_hz"]),
                "pose_span_s": [float(gyro_t[0]), float(gyro_t[-1])]},
        notes="synthetic verification run",
    )

    converged = inv.max_dt_residual < 1e-6
    if not converged:
        inv.pose_report["ok"] = False
        inv.pose_report["problems"].append({
            "type": "not_converged",
            "max_dt_residual_s": inv.max_dt_residual,
            "msg": "曝光时刻求解未收敛（运动过快或映射折叠），相关区域按无效标记",
        })

    stats = {
        "version": params.version(),
        "valid_fraction": float(inv.valid.mean()),
        "reason_counts": inv.reason_counts(),
        "pose_report": inv.pose_report,
        "iterations": inv.iterations,
        "max_dt_residual_s": inv.max_dt_residual,
        "converged": converged,
        "mae_valid": mae,
        "mae_flat": mae_flat,
        "mae_uncorrected": mae_uncorrected,
        "p95_valid": p95,
        "mae_global_homography": mae_global,
        "control_points": cp_stats,
        "line_wobble": wobble,
        "true_offset_s": TRUE_OFFSET_S,
        "moving_note": ("独立运动物体不受纯旋转/平面模型约束，校正结果在该区域无效，已按掩码标记"
                        if mover is not None else None),
    }
    return {
        "cfg": cfg, "params": params, "stats": stats,
        "images": {"reference": ref_img, "input": img_in, "corrected": corrected},
        "inv": inv, "track_corr": track_corr, "K": K,
        "fmap": fmap, "fvalid": fvalid, "moving_mask": moving_mask,
    }
