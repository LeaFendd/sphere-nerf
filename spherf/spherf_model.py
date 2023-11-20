from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Callable, Union, Optional
from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.base_model import Model
from nerfstudio.models.vanilla_nerf import VanillaModelConfig
from nerfstudio.model_components.scene_colliders import SphereCollider
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.losses import MSELoss, scale_gradients_by_distance_squared
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.utils import colormaps, misc

from spherf.spherf_field import SpheRFField
from spherf.spherf_module.sphere_scene_box import SphereSceneBox


@dataclass
class SpheRFModelConfig(VanillaModelConfig):
    _target: Type = field(default_factory=lambda: SpheRFModel)
    # field
    field_dim: int = 32
    angular_resolution: float = 0.2
    sphere_scene_box: Optional[SphereSceneBox] = None
    init_feat: Callable = partial(nn.init.trunc_normal_, std=0.02)
    # mlp
    num_layers_density: int = 2
    hidden_dim_density: int = 32
    num_layers_color: int = 3
    hidden_dim_color: int = 64
    # encoding
    radius_enc_freq: int = 4
    dir_shenc_level: int = 4
    # colider
    near_plane: float = 0.05
    # sampler
    coarse_samples: int = 64
    fine_samples: int = 128


class SpheRFModel(Model):
    """Implement SpheRF."""

    config: SpheRFModelConfig
    field: SpheRFField

    def __init__(self, config: SpheRFModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        super().populate_modules()

        self.field = SpheRFField(
            field_dim=self.config.field_dim,
            angular_resolution=self.config.angular_resolution,
            sphere_scene_box=self.config.sphere_scene_box,
            num_layers_density=self.config.num_layers_density,
            hidden_dim_density=self.config.hidden_dim_density,
            num_layers_color=self.config.num_layers_color,
            hidden_dim_color=self.config.hidden_dim_color,
            radius_enc_freq=self.config.radius_enc_freq,
            dir_shenc_level=self.config.dir_shenc_level,
        )

        # collider
        self.collider = SphereCollider(
            near_plane=self.config.near_plane,
            center=self.config.sphere_scene_box.center,
            radius=self.config.sphere_scene_box.radius,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(
            num_samples=self.config.num_importance_samples, include_original=False
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Union[Tensor, List]]:
        ray_bundle = self.collider.set_nears_and_fars(ray_bundle)

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform.generate_ray_samples(
            ray_bundle, self.config.coarse_samples
        )

        # First pass:
        field_outputs_coarse = self.field.forward(ray_samples_uniform)
        if self.config.use_gradient_scaling:
            field_outputs_coarse = scale_gradients_by_distance_squared(
                field_outputs_coarse, ray_samples_uniform
            )
        weights_coarse = ray_samples_uniform.get_weights(
            field_outputs_coarse[FieldHeadNames.DENSITY]
        )
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # Second pass:
        field_outputs_fine = self.field.forward(ray_samples_pdf)
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(
                field_outputs_fine, ray_samples_pdf
            )
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        pred_coarse, image_coarse = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_coarse"],
            pred_accumulation=outputs["accumulation_coarse"],
            gt_image=image,
        )
        pred_fine, image_fine = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )
        rgb_loss_coarse = self.rgb_loss(image_coarse, pred_coarse)
        rgb_loss_fine = self.rgb_loss(image_fine, pred_fine)
        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        assert (
            self.config.collider_params is not None
        ), "mip-NeRF requires collider parameters to be set."
        image = batch["image"].to(outputs["rgb_coarse"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])

        assert self.config.collider_params is not None
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
        rgb_coarse = torch.clip(rgb_coarse, min=0, max=1)
        rgb_fine = torch.clip(rgb_fine, min=0, max=1)

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)

        assert isinstance(fine_ssim, torch.Tensor)
        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr.item()),
            "fine_psnr": float(fine_psnr.item()),
            "fine_ssim": float(fine_ssim.item()),
            "fine_lpips": float(fine_lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())

        return param_groups
