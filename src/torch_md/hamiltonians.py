from typing import final, override
from torch import nn
import torch


class AtomEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def state_size(self) -> int:
        # TODO: deal with this
        return 10

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

@final
class FullyConnectedHamiltonian(nn.Module):
    def __init__(self, atom_encoder: AtomEncoder, num_hidden: int, num_layers: int) -> None:
        super().__init__(num_hidden, num_layers)
        # TODO: add a method that converts an atomic system with an arbitrary number of coordinates
        #  to an equivariant system that does not depend the the atom number
        self.atom_encoder = atom_encoder
        layers: list[nn.Module] = [nn.Linear(atom_encoder.state_size, num_hidden), nn.SiLU()]
        for _ in range(num_layers):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(nn.SiLU())
        layers.extend([nn.Linear(num_hidden, atom_encoder.state_size), nn.SiLU()])
        self.layers = nn.Sequential(*layers)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.atom_encoder(x))



