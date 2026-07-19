from dataclasses import dataclass
from enum import Enum


class CorrectionMode(Enum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"

@dataclass
class VariableConfig:
    mode: CorrectionMode
    lower_bound: float | None = None
    upper_bound: float | None = None


VARIABLE_REGISTRY: dict[str, VariableConfig] = {
    "tas":  VariableConfig(mode=CorrectionMode.ADDITIVE),
    "pr":   VariableConfig(mode=CorrectionMode.MULTIPLICATIVE, lower_bound=0.0),
    "mrso": VariableConfig(mode=CorrectionMode.MULTIPLICATIVE, lower_bound=0.0, upper_bound=1.0),
}
