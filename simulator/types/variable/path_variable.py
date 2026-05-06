from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict


PLACEHOLDERS = {
    # identifiers and metadata
    "seq_number": {"name": "Sequential Number", "cat": "increment", "type": "int", "description": "A sequential trial or run number"},
    "researcher": {"name": "Researcher", "cat": "generate", "type": "str", "description": "Researcher name or identifier"},
    # time and date placeholders
    "date": {"name": "Date", "cat": "date", "type": "datetime.date", "description": "Date. Includes day, month and optional year"},
    # variables corresponding to specific conditions
    "var": {"name": "Custom Variable", "cat": "var", "type": "str", "description": "A custom variable specific to the experimental context (can be reused)"},
}

PATH_CHOICES = {
    "file_extension": [".csv", ".json", ".jsonl", ".xlsx", ".txt", ".log"],
    "var_expansion": ["{name}={val}", "{name}_{val}", "{name}-{val}", "{name}{val}", "{val}"],
    "date_expansion": [
        "%Y%m%d", "%d%m%Y", "%m%d%Y", "%m%d", "%d%m",
        "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%m-%d", "%d-%m",
        "%Y_%m_%d", "%d_%m_%Y", "%m_%d_%Y", "%m_%d", "%d_%m"],
    "datetime_abbreviation": ["full", "short"],
    "researcher_format": ["username", "first_last", "firstname", "first-last", "last"],
}


class PathVariableLocation(Enum):
    DIRNAME = 'dir_name'
    FILENAME = 'file_name'


@dataclass
class PathVariable:
    name: str
    short_name: str
    description: str
    placeholder: str
    type: str
    category: str
    location: PathVariableLocation
    values: list = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        pass
    
    @classmethod
    def from_json(cls, obj):
        values = obj['values']
        metadata = obj['metadata']
        if obj['category'] == 'date':
            values = [date(*t) for t in values]
            metadata = {k: date(*t) for k, t in metadata.items()}

        return cls(
            name=obj['name'],
            short_name=obj['short_name'],
            description=obj['description'],
            placeholder=obj['placeholder'],
            type=obj['type'],
            category=obj['category'],
            location=PathVariableLocation[obj['location']],
            values=values,
            metadata=metadata,
        )

    def to_json(self) -> Dict[str, Any]:
        values = self.values
        metadata = self.metadata
        if self.category == 'date':
            values = [[d.year, d.month, d.day] for d in values]
            for k, v in self.metadata.items():
                if type(v) == date:
                    metadata[k] = (v.year, v.month, v.day)

        return {
            'name': self.name,
            'short_name': self.short_name,
            'description': self.description,
            'placeholder': self.placeholder,
            'type': self.type,
            'category': self.category,
            'location': self.location.name,
            'values': values,
            'metadata': metadata,
        }
    
    def reevaluate_type(self):
        try:
            all(int(v) for v in self.values)
            self.type = 'int'
            return
        except Exception:
            pass
        try:
            all(float(v) for v in self.values)
            self.type = 'float'
            return
        except Exception:
            pass
        if all(str(v) in ['True', 'False'] for v in self.values):
            self.type = 'bool'
        else:
            self.type = 'str'

    def to_type(self, v: str) -> Any:
        if self.type == 'float':
            return float(v)
        elif self.type == 'int':
            if '.' in str(v):
                v = str(v).split('.')[0]
            return int(v)
        elif self.type == 'bool':
            return bool(v)
        else:
            return str(v)
        
    def val_to_str(self, value: Any, path_choices: Dict[str, str]) -> str:
        if self.category == 'date':
            fmt = path_choices["date_expansion"]
            value = value.strftime(fmt)
        elif self.category == 'time':
            raise NotImplementedError
        return str(value)
    
    def vals_to_str(self, path_choices: Dict[str, str]) -> str:
        if self.category == 'date':
            min_date_str = self.metadata['min_date'].strftime("%Y-%m-%d")
            max_date_str = self.metadata['max_date'].strftime("%Y-%m-%d")
            fmt = path_choices["date_expansion"]
            values_str = f'({min_date_str} to {max_date_str}, encoded as "{fmt}")'
        elif self.category == 'time':
            raise NotImplementedError
        else:
            values_str = '[' + ', '.join([self.val_to_str(x, path_choices) for x in self.values]) + ']'
        return values_str
    
    def populate(self, s: str, value: Any, path_choices: Dict[str, str]):
        value = self.val_to_str(value, path_choices)
        if self.category == 'var':
            exp = path_choices["var_expansion"]
            value = exp.replace('{val}', str(value)).replace('{name}', str(self.short_name))
        return s.replace(self.placeholder, str(value))
