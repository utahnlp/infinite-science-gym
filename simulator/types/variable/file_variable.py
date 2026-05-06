from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class VariableRole(Enum):
    IDENTIFIER = 'identifier'
    INDEPENDENT = 'independent'
    DEPENDENT = 'dependent'
    DATETIME = 'datetime'

class VariableType(Enum):
    IDENTIFIER = 'identifier'
    CATEGORICAL = 'categorical'
    CONTINUOUS = 'continuous'
    INTEGER = 'integer'
    DATETIME = 'datetime'

default_variable_type_map = {
    VariableType.IDENTIFIER: 'str',
    VariableType.CATEGORICAL: 'str',
    VariableType.CONTINUOUS: 'float',
    VariableType.INTEGER: 'int',
    VariableType.DATETIME: 'datetime'}


@dataclass
class FileVariable:
    name: str
    short_name: str
    description: str
    role: VariableRole
    var_type: VariableType
    type: str

    def to_type(self, v: str) -> Any:
        if self.type == 'float':
            return float(v)
        elif self.type == 'int':
            if '.' in str(v):
                v = str(v).split('.')[0]
            return int(v)
        elif self.type == 'bool':
            return bool(v)
        elif self.type == 'datetime':
            return datetime.fromisoformat(v)
        else:
            return str(v) 
