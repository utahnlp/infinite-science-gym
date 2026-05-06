from dataclasses import dataclass
from typing import List

from .utils import default_field


@dataclass
class FileConfig:

    num_lines: List[int] = default_field([20, 300])

    # the error will be sampled from N(0, a * std), where std is the standard deviation of the distribution
    # this variable sets the value of a
    error_std_coeff: float = 0.1

    def __post_init__(self):
        assert len(self.num_lines) == 2
        assert all(type(x) == int for x in self.num_lines)
        assert all(x > 0 for x in self.num_lines)
        assert self.num_lines[0] < self.num_lines[1]
