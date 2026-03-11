from fairchem.core.datasets import AseDBDataset

from torch_md.data.config import SourceConfig
from torch_md.data.models import Calculation
from torch_md.data.ports import DatasetPort


def load_omol25(adapter: DatasetPort, src_config: SourceConfig):
    dataset = AseDBDataset({"src": src_config.data_path})
    calculation_batch: list[Calculation] = []
    for i in range(dataset.num_samples):
        atoms = dataset.get_atoms(i)
        calculation_batch.append(
            Calculation(
                id=i,
                formula=str(atoms.symbols),
                positions=atoms.positions,
                masses=atoms.get_masses(),
                energy=atoms.get_potential_energy(),
                forces=atoms.get_forces(),
            )
        )
        if len(calculation_batch) >= src_config.batch_size:
            # load the batch
            adapter.load(calculation_batch)

            # reset the batch
            calculation_batch = []
