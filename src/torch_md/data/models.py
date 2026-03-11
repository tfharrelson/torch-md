from typing import ClassVar
from pydantic import BaseModel, ConfigDict
import numpy as np


class Calculation(BaseModel):
    id: int
    formula: str
    energy: float
    forces: np.ndarray
    positions: np.ndarray
    masses: np.ndarray

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)
