import patito as pt


class Calculation(pt.Model):
    id: int
    formula: str
    energy: float
    forces: list[list[float]]
    positions: list[list[float]]
    masses: list[float]
