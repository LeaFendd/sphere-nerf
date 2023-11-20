from abc import abstractmethod, ABCMeta
from typing import Any

import torch
from torch import nn, Tensor
from jaxtyping import Float

from spherf.spherf_module.sphere_grid import SphereGrid


class SphereInterpolation(metaclass=ABCMeta):
    @abstractmethod
    def search_neighbors(
        tgt: Float[Tensor, "*bs coord_dim"], scene: SphereGrid
    ) -> Float[Tensor, "*bs k field_dim"]:
        ...

    @abstractmethod
    def interpolate(
        tgt: Float[Tensor, "*bs coord_dim"],
        neighbors: Float[Tensor, "*bs k field_dim"],
    ) -> Float[Tensor, "*bs field_dim"]:
        ...

    def __call__(
        self, tgt: Float[Tensor, "*bs coord_dim"], scene: SphereGrid
    ) -> Float[Tensor, "*bs field_dim"]:
        neighbors = self.search_neighbors(tgt, scene)
        v = self.interpolate(tgt, neighbors)

        return v


class AngularBilinearInterpolation(SphereInterpolation):
    def interpolate(
        tgt: Float[Tensor, "*bs 3"], neighbors: Float[Tensor, "*bs 4 field_dim"]
    ) -> Float[Tensor, "*bs field_dim"]:
        tgt = tgt - tgt.floor()
        x = tgt[..., 0]
        y = tgt[..., 1]
        weights = torch.stack(
            [(1 - x) * (1 - y), (1 - x) * y, x * (1 - y), x * y], dim=-1
        ).unsqueeze(-1)
        interpolated = torch.sum(weights * neighbors, dim=-2)

        return interpolated

    def search_neighbors(
        tgt: Float[Tensor, "*bs 3"], scene: SphereGrid
    ) -> Float[Tensor, "*bs 4 field_dim"]:
        N = scene.N
        vec = scene.scene_box.polar_coords(tgt)[..., :2]  # [theta, phi]
        idx_lb = (vec / scene.angular_resolution).long()  # left bottom
        idx_lt = idx_lb + torch.Tensor([0, 1], device=tgt.device).long()  # left top
        idx_rb = idx_lb + torch.Tensor([1, 0], device=tgt.device).long()  # right bottom
        idx_rt = idx_lb + torch.Tensor([1, 1], device=tgt.device).long()  # right top
        # query feats
        query_feats = lambda idx: scene.features[idx[..., 0] % N, idx[..., 1] % N]
        feat_lb = query_feats(idx_lb)
        feat_lt = query_feats(idx_lt)
        feat_rb = query_feats(idx_rb)
        feat_rt = query_feats(idx_rt)

        return torch.stack([feat_lb, feat_lt, feat_rb, feat_rt], dim=-2)


class DistanceWeightedInterpolation(SphereInterpolation):
    def search_neighbors(self, tgt: Tensor) -> Tensor:
        raise NotImplementedError

    def interpolate(self, tgt: Tensor, neighbors: Tensor) -> Tensor:
        raise NotImplementedError


class SlerpInterpolation(SphereInterpolation):
    def search_neighbors(self, tgt: Tensor) -> Tensor:
        raise NotImplementedError

    def interpolate(self, tgt: Tensor, neighbors: Tensor) -> Tensor:
        raise NotImplementedError
