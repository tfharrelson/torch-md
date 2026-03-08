from typing import Protocol
import torch


class Hamiltonian(Protocol):
    # TODO: come up with a better alternative to jaxtyping
    #  doesn't seem to work with ruff/based-pyright
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
