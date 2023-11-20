from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.configs.base_config import ViewerConfig

from spherf.spherf_pipeline import SpheRFPipelineConfig
from spherf.spherf_model import SpheRFModelConfig


spherf = MethodSpecification(
    config=TrainerConfig(
        method_name="spherf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=SpheRFPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
            ),
            model=SpheRFModelConfig(
                field_dim=56,
                angular_resolution=0.2,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
        },
        viewer=ViewerConfig(),
        vis="viewer",
    ),
    description="Base config for SpheRF.",
)
