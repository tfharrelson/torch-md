import patito as pt
import math
from tqdm import tqdm
from fairchem.core.datasets import AseDBDataset
import logging

from torch_md.data.config import SourceConfig
from torch_md.data.models import Calculation
from torch_md.data.ports import DataSink


logger = logging.getLogger(__name__)


def load_omol25(sink: DataSink, src_config: SourceConfig) -> None:
    logger.info(f"Loading OMOL25 data from {src_config.data_path.path}")
    print(f"Loading OMOL25 data from {src_config.data_path.path}")
    dataset = AseDBDataset({"src": src_config.data_path.path})

    ids: list[int] = []
    formulas: list[str] = []
    energies: list[float] = []
    forces_col: list[list[list[float]]] = []
    positions_col: list[list[list[float]]] = []
    masses_col: list[list[float]] = []

    with tqdm(total=math.ceil(len(dataset) / src_config.batch_size)) as progress_bar:
        for i in range(dataset.num_samples):
            atoms = dataset.get_atoms(i)

            ids.append(i)
            formulas.append(str(atoms.symbols))
            energies.append(float(atoms.get_potential_energy()))
            forces_col.append(atoms.get_forces().tolist())
            positions_col.append(atoms.positions.tolist())
            masses_col.append(atoms.get_masses().tolist())

            if len(ids) >= src_config.batch_size:
                batch = (
                    pt.DataFrame[Calculation](
                        {
                            "id": ids,
                            "formula": formulas,
                            "energy": energies,
                            "forces": forces_col,
                            "positions": positions_col,
                            "masses": masses_col,
                        }
                    )
                    .set_model(Calculation)
                    .validate()
                )
                sink.write(batch)
                progress_bar.update(1)

                ids = []
                formulas = []
                energies = []
                forces_col = []
                positions_col = []
                masses_col = []

        # flush remaining records
        if ids:
            batch = (
                pt.DataFrame[Calculation](
                    {
                        "id": ids,
                        "formula": formulas,
                        "energy": energies,
                        "forces": forces_col,
                        "positions": positions_col,
                        "masses": masses_col,
                    }
                )
                .set_model(Calculation)
                .validate()
            )
            sink.write(batch)
