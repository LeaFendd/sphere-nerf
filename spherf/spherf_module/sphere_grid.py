from typing import Callable, Type
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn, Tensor
from jaxtyping import Float

from spherf.spherf_module.sphere_scene_box import SphereSceneBox
from spherf.spherf_module.misc import create_meshgrid


class SphereGrid(nn.Module):
    def __init__(
        self,
        field_dim: int,
        angular_resolution: float,
        scene_box: SphereSceneBox,
        interpolation: Callable,
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

    def query(self, tgt: Float[Tensor, "*bs 3"]) -> Float[Tensor, "*bs field_dim"]:
        return self.interpolation(tgt, self)

    @property
    def shape(self):
        return torch.Size([self.N, self.N])

    @property
    def dim(self):
        return self.field_dim


def angular_bilinear_interpolation(
    tgt: Float[Tensor, "*bs 3"], scene: SphereGrid
) -> Float[Tensor, "*bs field_dim"]:
    N = scene.N
    vec = scene.scene_box.polar_coords(tgt)[..., :2]  # [theta, phi]
    idx_lb = (vec / scene.angular_resolution).long()  # left bottom
    idx_lt = idx_lb + torch.Tensor([0, 1]).to(tgt.device).long()  # left top
    idx_rb = idx_lb + torch.Tensor([1, 0]).to(tgt.device).long()  # right bottom
    idx_rt = idx_lb + torch.Tensor([1, 1]).to(tgt.device).long()  # right top

    # query feats
    query_feats = lambda idx: scene.features[idx[..., 0] % N, idx[..., 1] % N]
    feat_lb = query_feats(idx_lb)
    feat_lt = query_feats(idx_lt)
    feat_rb = query_feats(idx_rb)
    feat_rt = query_feats(idx_rt)
    neighbors = torch.stack([feat_lb, feat_lt, feat_rb, feat_rt], dim=-2)

    # interpolate from neighbors.
    tgt = tgt - tgt.floor()
    x = tgt[..., 0]
    y = tgt[..., 1]
    weights = torch.stack(
        [
            (1 - x) * (1 - y),
            (1 - x) * y,
            x * (1 - y),
            x * y,
        ],
        dim=-1,
    ).unsqueeze(-1)
    interpolated = torch.sum(weights * neighbors, dim=-2)

    return interpolated


if __name__ == "__main__":
    scene_box = SphereSceneBox(torch.Tensor([0, 0, 0]), torch.Tensor([1]))
    scene_feature = SphereGrid(32, 0.5, scene_box, angular_bilinear_interpolation)

    tgt = torch.randn(10, 3)
    print("tgt Shape:" + str(tgt.shape))
    feat = scene_feature.query(tgt)
    print("feat Shape:" + str(feat.shape))

    print("Scene Feature Shape:" + str(scene_feature.shape))
    print("Scene Feature Dim:" + str(scene_feature.dim))
