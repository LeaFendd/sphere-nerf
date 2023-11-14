import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type, Literal

from nerfstudio.models.base_model import Model, ModelConfig


@dataclass
class SphereNeRFModelConfig(ModelConfig):
    """Sphere NeRF Model Config"""

    _target: Type = field(default_factory=lambda: SphereNeRFModel)

    angular_resolution: float = 0.2
    """Angular resolution of sphere bones."""

    field_dim: int = 48
    """Feature dimension."""


class SphereNeRFModel(Model):
    """Sphere NeRF Model

    Args:
        config: Basic SphereNeRF configuration to instantiate model
    """

    config: SphereNeRFModelConfig

    def __init__(
        self,
        config: SphereNeRFModelConfig,
        **kwargs,
    ) -> None:
        self.angular_resolution = config.angular_resolution
        self.field_dim = config.field_dim
        self.filed_shape = [int(360 / self.angular_resolution)] * 2

        super().__init__(config=config, **kwargs)
