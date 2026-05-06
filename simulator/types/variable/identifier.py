from dataclasses import dataclass

from . import FileVariable


@dataclass
class IdentifierFileVariable(FileVariable):
    shuffled: bool
