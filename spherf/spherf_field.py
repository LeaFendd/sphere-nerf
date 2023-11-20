from typing import Dict, Literal, Optional, Tuple, Callable
from functools import partial

import torch
from torch import Tensor, nn
from jaxtyping import Float, Shaped

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHeadNames,
    RGBFieldHead,
)

from spherf.spherf_module.sphere_scene_box import SphereSceneBox
from spherf.spherf_module.sphere_grid import SphereGrid, angular_bilinear_interpolation


class SpheRFField(Field):
    def __init__(
        self,
        # field
        field_dim: int,
        angular_resolution: float,
        sphere_scene_box: SphereSceneBox,
        init_feat: Callable = partial(nn.init.trunc_normal_, std=0.02),
        # mlp
        num_layers_density: int = 2,
        hidden_dim_density: int = 64,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        # encoding
        radius_enc_freq: int = 4,
        dir_shenc_level: int = 4,
        # backend
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()
        # Scene.
        self.scene = SphereGrid(
            field_dim,
            angular_resolution,
            sphere_scene_box,
            angular_bilinear_interpolation,
            init_feat,
        )

        # Direction Encoding.
        self.direction_encoding = SHEncoding(
            levels=dir_shenc_level,
            implementation=implementation,
        )

        # Radius Encoding.
        self.radius_encoding = NeRFEncoding(
            in_dim=1,
            num_frequencies=radius_enc_freq,
            min_freq_exp=0,
            max_freq_exp=radius_enc_freq - 1,
            implementation=implementation,
        )

        # Density MLP.
        self.dim_in_density = self.scene.dim + self.radius_encoding.get_out_dim()
        self.mlp_density = MLP(
            in_dim=self.dim_in_density,
            num_layers=num_layers_density,
            layer_width=hidden_dim_density,
            out_dim=hidden_dim_density,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        self.head_density = DensityFieldHead(self.mlp_density.get_out_dim())

        # RGB MLP.
        self.dim_in_color = (
            self.mlp_density.get_out_dim() + self.direction_encoding.get_out_dim() + self.scene.dim
        )
        self.mlp_rgb = MLP(
            in_dim=self.dim_in_color,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=hidden_dim_color,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        # FIXME use FieldHead will lead to dtype mismatching error.
        self.head_rgb = RGBFieldHead(self.mlp_rgb.get_out_dim())

    def get_density(
        self, ray_samples: RaySamples
    ) -> Tuple[Shaped[Tensor, "*batch 1"], Float[Tensor, "*batch num_features"]]:
        # convert to polar coordinates.
        positions: Float[Tensor, "bs 3"] = (
            ray_samples.frustums.get_positions().detach().reshape([-1, 3])
        )
        positions = self.scene.scene_box.polar_coords(positions)
        r: Float[Tensor, "bs 1"] = positions[..., -1:]

        # TODO: check query features.
        feature: Float[Tensor, "bs dim_f"] = self.scene.query(positions)
        encoded_r: Float[Tensor, "bs dim_r"] = self.radius_encoding(r)

        # passthrough mlp_density.
        mlp_density_out: Float[Tensor, "bs dim_o"] = self.mlp_density(
            torch.cat([feature, encoded_r], dim=-1)
        )
        density: Float[Tensor, "bs 1"] = self.head_density(mlp_density_out)

        # TODO: shotcut features?
        density_embedding = torch.cat([mlp_density_out, feature], -1)

        density: Float[Tensor, "*ray_sample 1"] = density.reshape([*ray_samples.shape, -1])

        return density, density_embedding

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}

        # encode dir.
        dir: Float[Tensor, "bs 3"] = ray_samples.frustums.directions.reshape([-1, 3])
        encoded_dir: Float[Tensor, "bs dim_d"] = self.direction_encoding(dir)

        # passthrough mlp_rgb.
        mlp_rgb_out: Tensor = self.mlp_rgb(torch.cat([encoded_dir, density_embedding], dim=-1))
        mlp_rgb_out = mlp_rgb_out.reshape([*ray_samples.shape, -1])
        outputs[FieldHeadNames.RGB] = self.head_rgb(mlp_rgb_out)

        return outputs
