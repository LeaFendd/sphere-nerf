from typing import Callable, Type
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn, Tensor
from jaxtyping import Float

from .sphere_scene_box import SphereSceneBox
from .interpolation import SphereInterpolation


class SphereScene(nn.Module):
    def __init__(
        self,
        field_dim: int,
        angular_resolution: float,
        scene_box: SphereSceneBox,
        interpolation: Type[SphereInterpolation],
        init_feat: Callable = partial(nn.init.trunc_normal_, std=0.02),
    ) -> None:
        super().__init__()
        self.field_dim = field_dim
        self.scene_box = scene_box
        self.angular_resolution = angular_resolution
        self.interpolation = interpolation
        # build features.
        self.register_buffer(
            "samples", torch.deg2rad_(create_meshgrid(360, 360, angular_resolution))
        )
        self.N = self.samples.shape[0]
        self.features = init_feat(nn.Parameter(torch.empty(self.N, self.N, field_dim)))

    def query(self, tgt: Float[Tensor, "*bs 3"]):
        return self.interpolation(tgt, self.features)

    @property
    def shape(self):
        return torch.Size([self.N, self.N])

    @property
    def dim(self):
        return self.field_dim


def create_meshgrid(h, w, step, dtype=torch.float) -> Float[Tensor, "N N 2"]:
    """Create meshgrid."""
    grid_xx, grid_yy = torch.meshgrid(
        torch.arange(0, h, step, dtype=dtype),
        torch.arange(0, w, step, dtype=dtype),
    )
    grid = torch.stack((grid_xx, grid_yy), dim=-1)
    return grid


if __name__ == "__main__":
    scene_box = SphereSceneBox(torch.Tensor([0, 0, 0]), torch.Tensor([1]))
    scene_feature = SphereScene(32, 0.5, scene_box)

    tgt = torch.randn(10, 3)
    print("tgt Shape:" + str(tgt.shape))
    feat = scene_feature.query(tgt)
    print("feat Shape:" + str(feat.shape))

    print("Scene Feature Shape:" + str(scene_feature.shape))
    print("Scene Feature Dim:" + str(scene_feature.dim))
