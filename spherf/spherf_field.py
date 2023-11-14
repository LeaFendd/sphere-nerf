from typing import Dict, Literal, Optional, Tuple, Callable
from functools import partial
from dataclasses import dataclass

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

from .spherf_module.sphere_scene_box import SphereSceneBox
from .spherf_module.sphere_scene import SphereScene
from .spherf_module.interpolation import AngularBilinearInterpolation


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
        self.scene = SphereScene(
            field_dim,
            angular_resolution,
            sphere_scene_box,
            init_feat,
            AngularBilinearInterpolation,
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
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )
        self.head_rgb = RGBFieldHead(self.mlp_rgb.get_out_dim())

    def get_density(
        self, ray_samples: RaySamples
    ) -> Tuple[Shaped[Tensor, "*batch 1"], Float[Tensor, "*batch num_features"]]:
        positions = ray_samples.frustums.get_positions().detach()
        positions = self.scene.scene_box.polar_coords(positions)
        r: Float[Tensor, "*bs 1"] = positions[..., -1:]

        # TODO: check query features.
        feature: Float[Tensor, "*bs field_dim"] = self.scene.query(positions)
        encoded_r: Float[Tensor, "*bs r_enc_dim"] = self.radius_encoding(r)
        # mlp_density.
        mlp_density_out = self.mlp_density(torch.cat([feature, encoded_r], dim=-1))
        density = self.head_density(mlp_density_out)

        # TODO: shotcut features?
        density_embedding = torch.cat([mlp_density_out, feature], -1)

        return density, density_embedding

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}

        encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
        outputs[FieldHeadNames.RGB] = self.head_rgb(
            torch.cat([encoded_dir, density_embedding], dim=-1)
        )

        return outputs
