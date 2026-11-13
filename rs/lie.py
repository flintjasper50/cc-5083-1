"""SO(3) helpers. Quaternions are [x, y, z, w]; rotation vectors are axis*angle in radians."""
import numpy as np


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_from_rotvec(rv):
    rv = np.asarray(rv, dtype=np.float64)
    theta = np.linalg.norm(rv, axis=-1, keepdims=True)
    half = 0.5 * theta
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(theta < 1e-12, 0.5 - theta * theta / 48.0, np.sin(half) / theta)
    return np.concatenate([rv * scale, np.cos(half)], axis=-1)


def quat_to_rotvec(q):
    q = quat_normalize(q)
    xyz = q[..., :3]
    w = np.clip(q[..., 3:4], -1.0, 1.0)
    s = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, w)
    s_safe = np.where(s < 1e-12, 1.0, s)
    axis = np.where(s < 1e-12, np.array([1.0, 0.0, 0.0]), xyz / s_safe)
    return axis * angle


def quat_mul(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_conj(q):
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([-q[..., :3], q[..., 3:]], axis=-1)


def quat_rotate(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    xyz = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def slerp(q0, q1, u):
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    u = np.asarray(u, dtype=np.float64)
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0.0, -q1, q1)
    d = np.clip(np.abs(d), 0.0, 1.0)
    theta = np.arccos(d)
    st = np.sin(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(st < 1e-9, 1.0 - u, np.sin((1.0 - u) * theta) / st)
        b = np.where(st < 1e-9, u, np.sin(u * theta) / st)
    return quat_normalize(a * q0 + b * q1)


def quat_angle_between(a, b):
    d = np.clip(np.abs(np.sum(quat_normalize(a) * quat_normalize(b), axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(d)


def quat_to_matrix(q):
    q = quat_normalize(q)
    x, y, z, w = np.moveaxis(q, -1, 0)
    m = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1)
    return m.reshape(q.shape[:-1] + (3, 3))
