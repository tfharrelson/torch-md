from typing import override
import torch
from torch import nn
from .types import Hamiltonian


class VelocityVerletIntegrator(nn.Module):
    def __init__(self, hamiltonian: Hamiltonian, masses: torch.Tensor, time_step: float) -> None:
        super().__init__()
        self.hamiltonian: Hamiltonian = hamiltonian
        self.masses: torch.Tensor
        self.time_step: float = time_step

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x must be a state tensor
        # TODO: figure out how to statically type a state tensor
        forces = self._calculate_forces(x)

        half_step_velocity = x[:, 3:] + 0.5 * forces / self.masses * self.time_step
        updated_positions = x[:, :3] + half_step_velocity * self.time_step
        updated_acceleration = self._calculate_forces(updated_positions) / self.masses

        updated_velocity = half_step_velocity + 0.5 * updated_acceleration * self.time_step
        x[:, :3] = updated_positions
        x[:, 3:] = updated_velocity
        return x

    def _calculate_forces(self, positions: torch.Tensor) -> torch.Tensor:
        forces = self.hamiltonian.forward(positions).backward()
        if not isinstance(forces, torch.Tensor):
            raise RuntimeError("gradient of hamiltonian is not a tensor")
        return forces

