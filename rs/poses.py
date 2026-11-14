"""Pose tracks: time-stamped world->camera rotations with interpolation and adequacy checks.

Units: times in seconds, quaternions [x,y,z,w] (unitless), angles in radians.
"""
import math
from dataclasses import dataclass

import numpy as np

from . import lie


@dataclass
class PoseTrack:
    times: np.ndarray  # (N,) seconds, strictly increasing
    quats: np.ndarray  # (N,4) [x,y,z,w], world -> camera

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64)
        self.quats = lie.quat_normalize(np.asarray(self.quats, dtype=np.float64))
        if self.times.ndim != 1 or self.quats.shape != (self.times.size, 4):
            raise ValueError("times (N,) and quats (N,4) shape mismatch")
        if self.times.size < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError("times must be strictly increasing, need >= 2 samples")

    @classmethod
    def from_dict(cls, d):
        """Load from a poses-file dict. Two accepted shapes, both with explicit
        timestamps on the pose clock (seconds):

          {"times_s": [...], "quats_xyzw": [[x,y,z,w], ...]}
          {"gyro_times_s": [...], "gyro_rad_s": [[wx,wy,wz], ...], "q0_xyzw": [...]}

        Files without an explicit time base are rejected — the sampling rate is
        never guessed.
        """
        if "times_s" in d and "quats_xyzw" in d:
            return cls(np.asarray(d["times_s"], float), np.asarray(d["quats_xyzw"], float))
        if "gyro_times_s" in d and "gyro_rad_s" in d:
            return cls.integrate_gyro(np.asarray(d["gyro_times_s"], float),
                                      np.asarray(d["gyro_rad_s"], float),
                                      d.get("q0_xyzw", (0, 0, 0, 1)))
        raise ValueError(
            "姿态文件缺少时间基准：需要 times_s+quats_xyzw 或 "
            "gyro_times_s+gyro_rad_s（秒），不会按采样率猜测时间轴")

    @classmethod
    def from_body_rate(cls, times, omega, q0=(0.0, 0.0, 0.0, 1.0)):
        """Analytic track for constant body-frame angular velocity omega (rad/s)."""
        times = np.asarray(times, dtype=np.float64)
        rv = np.outer(times - times[0], np.asarray(omega, dtype=np.float64))
        dq = lie.quat_from_rotvec(rv)
        q0 = lie.quat_normalize(np.asarray(q0, dtype=np.float64))
        return cls(times, lie.quat_mul(np.broadcast_to(q0, dq.shape), dq))

    @classmethod
    def integrate_gyro(cls, times, omegas, q0=(0.0, 0.0, 0.0, 1.0)):
        """Trapezoidal integration of body-frame gyro samples (rad/s)."""
        times = np.asarray(times, dtype=np.float64)
        omegas = np.asarray(omegas, dtype=np.float64)
        quats = np.empty((times.size, 4))
        quats[0] = lie.quat_normalize(np.asarray(q0, dtype=np.float64))
        for k in range(times.size - 1):
            dt = times[k + 1] - times[k]
            w = 0.5 * (omegas[k] + omegas[k + 1])
            quats[k + 1] = lie.quat_mul(quats[k], lie.quat_from_rotvec(w * dt))
        return cls(times, lie.quat_normalize(quats))

    def interp(self, t):
        """SLERP at times t (array). Returns (quats (...,4), outside_mask (...))."""
        t = np.asarray(t, dtype=np.float64)
        idx = np.clip(np.searchsorted(self.times, t) - 1, 0, self.times.size - 2)
        t0 = self.times[idx]
        t1 = self.times[idx + 1]
        u = (t - t0) / (t1 - t0)
        q = lie.slerp(self.quats[idx], self.quats[idx + 1], np.expand_dims(u, -1))
        outside = (t < self.times[0]) | (t > self.times[-1])
        return q, outside

    def max_step_angle(self):
        return float(np.max(lie.quat_angle_between(self.quats[:-1], self.quats[1:])))

    def adequacy(self, t_lo, t_hi, max_step_rad=math.radians(2.0)):
        """Check the track covers [t_lo, t_hi] densely enough for row-exposure interpolation.

        The report is all-SI: times in seconds, angles in radians (matching "units").
        """
        problems = []
        if t_lo < self.times[0] - 1e-12:
            problems.append({
                "type": "coverage_start",
                "need_s": float(t_lo),
                "have_s": float(self.times[0]),
                "msg": "曝光窗口起点早于首个姿态样本，姿态插值不足",
            })
        if t_hi > self.times[-1] + 1e-12:
            problems.append({
                "type": "coverage_end",
                "need_s": float(t_hi),
                "have_s": float(self.times[-1]),
                "msg": "曝光窗口终点晚于末个姿态样本，姿态插值不足",
            })
        step = self.max_step_angle()
        if step > max_step_rad:
            problems.append({
                "type": "sparse",
                "max_step_rad": step,
                "limit_rad": max_step_rad,
                "msg": "相邻姿态样本转角过大，插值不可靠",
            })
        return {
            "ok": not problems,
            "problems": problems,
            "span_s": [float(self.times[0]), float(self.times[-1])],
            "exposure_window_s": [float(t_lo), float(t_hi)],
            "max_step_rad": step,
            "units": {"time": "s", "angle": "rad"},
        }
