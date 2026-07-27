"""Camera math for the 3D frontend.

:class:`OrbitCamera` circles a target point at a fixed radius and elevation. The
follow and first-person behaviours are policies over it rather than separate
classes: the run loops move ``target`` (see ``app._follow_step``) or build a
look-at matrix directly (see ``app._lookat_mvp``).
"""

from __future__ import annotations

import math

import numpy as np
from pyrr import Matrix44, Vector3


class OrbitCamera:
    """Camera that orbits a target point at a fixed radius/elevation."""

    def __init__(self, target, radius: float, elevation_deg: float = 30.0,
                 fov_deg: float = 55.0, near: float = 1.0, far: float = 8000.0):
        self.target = Vector3(list(target), dtype="f4")
        self.radius = float(radius)
        self.elevation = math.radians(elevation_deg)
        self.azimuth = 0.0  # radians, advanced over time
        self.fov = math.radians(fov_deg)
        self.near = near
        self.far = far

    def eye(self) -> Vector3:
        r_horiz = self.radius * math.cos(self.elevation)
        x = self.target.x + r_horiz * math.sin(self.azimuth)
        z = self.target.z + r_horiz * math.cos(self.azimuth)
        y = self.target.y + self.radius * math.sin(self.elevation)
        return Vector3([x, y, z], dtype="f4")

    def view(self) -> Matrix44:
        return Matrix44.look_at(self.eye(), self.target, Vector3([0.0, 1.0, 0.0]), dtype="f4")

    def projection(self, aspect: float) -> Matrix44:
        return Matrix44.perspective_projection(
            math.degrees(self.fov), aspect, self.near, self.far, dtype="f4"
        )

    def mvp(self, aspect: float) -> np.ndarray:
        return (self.projection(aspect) * self.view()).astype("f4")
