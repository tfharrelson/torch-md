from typing import ClassVar
from pydantic import BaseModel, ConfigDict, field_validator
import numpy as np


class Calculation(BaseModel):
    id: int
    formula: str
    energy: float
    forces: np.ndarray
    positions: np.ndarray
    masses: np.ndarray

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('forces', 'positions', 'masses', mode='before')
    @classmethod
    def convert_list_to_array(cls, v):
        """Convert list (from DuckDB/JSON) to numpy array"""
        if isinstance(v, list):
            return np.array(v)
        return v
