"""Pinhole intrinsics. Pixel units, focal lengths in px."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def rays(self, width, height):
        """Camera-frame rays (z=1, not normalized) for every pixel center. Returns (H,W,3)."""
        u = np.arange(width, dtype=np.float64)
        v = np.arange(height, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
        x = (uu - self.cx) / self.fx
        y = (vv - self.cy) / self.fy
        return np.stack([x, y, np.ones_like(x)], axis=-1)

    def project(self, pts):
        """Project camera-frame points (...,3). Returns (uv (...,2), z (...))."""
        pts = np.asarray(pts, dtype=np.float64)
        z = pts[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * pts[..., 0] / z + self.cx
            v = self.fy * pts[..., 1] / z + self.cy
        return np.stack([u, v], axis=-1), z
