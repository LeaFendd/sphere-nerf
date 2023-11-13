from dataclasses import dataclass
from typing import Any

import torch
from torch import nn, Tensor
from jaxtyping import Float, Int, Shaped, Bool

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle


@dataclass
class SphereSceneBox:
    """Data to represent the scene box."""

    center: Float[Tensor, "3"]
    radius: Float[Tensor, "1"]

    @staticmethod
    def build(cameras: Cameras, scale: float = 1.25, iter: int = 200):
        """Construct a sphere from the camera poses with optimization method."""

        # get rays from cameras' optical center to the principle point.
        cam_origins = cameras.camera_to_worlds[..., 3]
        cam_dir = cameras.camera_to_worlds[..., 2]

        # optimize the scene center.
        center = torch.zeros(3, dtype=torch.float, requires_grad=True)
        opt = torch.optim.SGD([center], lr=1e-3)

        # distance from a point(x, y, z) to ray(o, d) is:
        # $$dist=\cfrac{|(x-o)\times d|}{|d|}$$
        # optimize the distance to get the scene center.
        for _ in range(iter):
            opt.zero_grad()
            dist = torch.norm(
                torch.cross(center - cam_origins, cam_dir, dim=1), dim=1
            ) / torch.norm(cam_dir, dim=1)
            loss = torch.sum(dist**2)
            loss.backward()
            opt.step()

        # scene radius is the mean distance of cameras to center.
        radius = torch.mean(torch.sqrt(torch.sum((cam_origins - center) ** 2, dim=1))) * scale

        center = center.detach()
        radius = radius.detach()

        return SphereSceneBox(center=center, radius=radius)

    def within(self, pts: Float[Tensor, "*num_pts 3"]) -> Bool[Tensor, "*num_pts 1"]:
        dist = torch.norm((pts - self.center), dim=-1)

        return dist <= self.radius

    def polar_coords(self, pts: Float[Tensor, "*num_pts 3"]) -> Float[Tensor, "*num_pts 3"]:
        rel_coords = pts - self.center
        r = torch.norm((rel_coords), dim=-1)
        theta = torch.acos(rel_coords[..., 2] / r)
        phi = torch.atan2(rel_coords[..., 1], rel_coords[..., 0])
        polar_coords = torch.nan_to_num(torch.stack([r, theta, phi], dim=-1))

        return polar_coords
