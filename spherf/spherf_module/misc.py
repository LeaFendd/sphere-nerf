import torch
from torch import Tensor
from jaxtyping import Float


def create_meshgrid(h, w, step, dtype=torch.float) -> Float[Tensor, "N N 2"]:
    """Create meshgrid."""
    grid_xx, grid_yy = torch.meshgrid(
        torch.arange(0, h, step, dtype=dtype),
        torch.arange(0, w, step, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_xx, grid_yy), dim=-1)
    return grid
