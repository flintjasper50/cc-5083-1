"""Rolling-shutter readout model and scene-motion models.

ReadoutSpec maps a pixel to its exposure time on the pose clock.
Scene models map a reference-frame ray plus exposure time to input-image coordinates.
"""
from dataclasses import dataclass, field

import numpy as np

from . import lie

DIRECTIONS = ("top_to_bottom", "bottom_to_top", "left_to_right", "right_to_left")


@dataclass
class ReadoutSpec:
    """Line-exposure timing on the pose clock.

    `direction` says where readout line 0 sits in the image and which way the
    scan proceeds; every pixel of one line shares one exposure instant:

        top_to_bottom:  line 0 = top row    (v=0),     line index r = v
        bottom_to_top:  line 0 = bottom row (v=H-1),   r = (H-1) - v
        left_to_right:  line 0 = left column  (u=0),   r = u
        right_to_left:  line 0 = right column (u=W-1), r = (W-1) - u

    Line r is exposed at line_time(r) = t0 + t_offset + r * readout_time/(n_lines-1),
    so `readout_time` is the span from the first to the last line (not per-line).
    t0 is the first line's instant on the pose clock assuming both clocks agree;
    t_offset models a constant sensor-vs-pose clock skew. All times in seconds.
    """
    width: int
    height: int
    readout_time: float       # seconds from first to last line
    direction: str = "top_to_bottom"
    t0: float = 0.0           # pose-clock time of the first line before offset
    t_offset: float = 0.0     # extra clock offset between sensor and pose clocks (s)

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")

    @property
    def n_lines(self):
        return self.height if self.direction in ("top_to_bottom", "bottom_to_top") else self.width

    @property
    def line_dt(self):
        return self.readout_time / (self.n_lines - 1)

    def readout_index(self, u, v):
        """Line index (float ok) for pixel coordinate(s) u, v."""
        if self.direction == "top_to_bottom":
            return v
        if self.direction == "bottom_to_top":
            return (self.height - 1) - v
        if self.direction == "left_to_right":
            return u
        return (self.width - 1) - u

    def line_time(self, r):
        return self.t0 + self.t_offset + np.asarray(r, dtype=np.float64) * self.line_dt

    def time_range(self):
        lo = self.t0 + self.t_offset
        return lo, lo + self.readout_time


class RotationScene:
    """Pure-rotation camera: input pixel for a reference ray at time t."""

    def __init__(self, track, q_ref, K):
        self.track = track
        self.q_ref = np.asarray(q_ref, dtype=np.float64)
        self.K = K

    def to_input(self, rays_ref, t):
        q_t, outside = self.track.interp(t)
        world = lie.quat_rotate(lie.quat_conj(self.q_ref), rays_ref)
        cam = lie.quat_rotate(q_t, world)
        uv, z = self.K.project(cam)
        return uv, z, ~outside


class PlaneScene:
    """Known plane n^T X = d (reference frame) plus camera translation C(t) = center0 + velocity*(t - t_ref).

    normal/center0/velocity are given in the world frame of the pose track; the normal is
    converted internally so the plane equation stays n^T X_ref = d in the reference frame.
    """

    def __init__(self, track, q_ref, K, normal, dist, center0, velocity, t_ref):
        self.track = track
        self.q_ref = np.asarray(q_ref, dtype=np.float64)
        self.K = K
        n = np.asarray(normal, dtype=np.float64)
        self.n_ref = n / np.linalg.norm(n)
        self.n_world = lie.quat_rotate(lie.quat_conj(self.q_ref), self.n_ref)
        self.d = float(dist)
        self.c0 = np.asarray(center0, dtype=np.float64)
        self.v = np.asarray(velocity, dtype=np.float64)
        self.t_ref = float(t_ref)

    def center(self, t):
        return self.c0 + (np.asarray(t, dtype=np.float64)[..., None] - self.t_ref) * self.v

    def to_input(self, rays_ref, t):
        dir_w = lie.quat_rotate(lie.quat_conj(self.q_ref), rays_ref)
        den = dir_w @ self.n_world
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.where(den > 1e-12, self.d / den, np.nan)
        X = lam[..., None] * dir_w
        q_t, outside = self.track.interp(t)
        cam = lie.quat_rotate(q_t, X - self.center(t))
        uv, z = self.K.project(cam)
        ok = (~outside) & (den > 1e-12)
        return uv, z, ok
