import torch
from typing import override
from torch import nn


class SimpleMLIP(nn.Module):
    """
    TLDR: SO(3) equivariance is cool and all, but let's try something dumb.

    Transformers already have permutation invariance built in which deals with lots
    of internal rotations and symmetries. The overall atomic frame can be rotated which
    may confuse the model, but let's try it anyway. It may be possible to include synthetic
    rotations in the dataset to help the model generalize over the rotaional space.

    Casting to spherical harmonics is a standard way to handle SO(3) equivariance, but
    i'm not sure what happens to the permutation invariance, and i think that'll come
    in handy.
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        # TODO: make inputs configurable
        self._embedding = nn.Linear(4, d_model)
        self._transformer = nn.Transformer(d_model=d_model)
        self._energy_head = nn.Linear(d_model, 1)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x represents the atomic positions with atomic charge as input. The shape
        is (N, 4) where N is the number of atoms.
        """
        x = self._embedding.forward(x)
        # TODO: being lazy for now, should probably switch to specifying the encoder/decoder layers
        # directly
        energy = self._energy_head.forward(self._transformer.forward(x, x))
        return energy.squeeze(-1).sum(dim=-1)
