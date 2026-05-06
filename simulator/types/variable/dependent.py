from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import FileVariable


@dataclass
class DependentFileVariable(FileVariable):
    fn_name: str
    fn_str: str
    fn: Callable = None
    depends_on: List[str] = field(default_factory=list) # short_name values of dependencies

    def __post_init__(self):
        pass

    def compile_fn(self, raise_error_on_failure: bool = False) -> bool:
        try:
            vars = {}
            exec(self.fn_str, globals(), vars)
            self.fn = vars[self.fn_name]
            return True
        except Exception as e:
            if raise_error_on_failure:
                raise e
        return False

    def apply_fn(
            self, 
            path_variables: Dict[str, str], 
            independent_variables: Dict[str, Any], 
            error: float = 0.0) -> Any:

        return self.fn(
            path_variables | independent_variables, 
            error)
    
    def format_fn(self) -> str:
        return '\n'.join([f'{i+1:6d}  {l}' for i, l in enumerate(self.fn_str.split('\n'))])


@dataclass
class DependentCategoricalVariable(DependentFileVariable):
    values: List[str] = field(default_factory=dict)
    probabilities: List[float] = field(default_factory=dict)

    @classmethod
    def from_file_variable(
            cls, 
            fv: FileVariable, 
            fn_name: str, 
            fn_str: str, 
            depends_on: List[str], 
            values: List[str] = [],
            probabilities: List[float] = [],
            fn: Optional[Callable] = None):
        
        return cls(
            name=fv.name, 
            short_name=fv.short_name, 
            description=fv.description, 
            role=fv.role, 
            var_type=fv.var_type, 
            type='str',
            fn_name=fn_name, 
            fn_str=fn_str,
            fn=fn,
            depends_on=depends_on,
            values=values,
            probabilities=probabilities)

    def __post_init__(self):
        assert len(self.values) == len(self.probabilities), f'{len(self.values)=} needs to be same as {len(self.probabilities)=}'
        
        # normalize if not normalized
        if self.probabilities and sum(self.probabilities) != 1:
            sum_probs = sum(self.probabilities)
            self.probabilities = [x / sum_probs for x in self.probabilities]

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


@dataclass
class DependentContinuousVariable(DependentFileVariable):
    std: float = 0.0
    deciles: List[float] = field(default_factory=dict)

    @classmethod
    def from_file_variable(
            cls, 
            fv: FileVariable, 
            fn_name: str, 
            fn_str: str, 
            depends_on: List[str], 
            std: float = 0.0,
            deciles: List[float] = [],
            fn: Optional[Callable] = None):
        
        return cls(
            name=fv.name, 
            short_name=fv.short_name, 
            description=fv.description, 
            role=fv.role, 
            var_type=fv.var_type, 
            type='float',
            fn_name=fn_name, 
            fn_str=fn_str,
            fn=fn,
            depends_on=depends_on,
            std=std,
            deciles=deciles)

    def __post_init__(self):
        pass

    def get_low_high_quantiles(self, low=0.1, high=0.9) -> Tuple[float, float]:
        assert 0 < low and low < 1
        assert 0 < high and high < 1
        assert low in np.arange(1.1, step=0.1)
        assert high in np.arange(1.1, step=0.1)

        return self.deciles[int(low * 10)], self.deciles[int(high * 10)]


@dataclass
class DependentIntegerVariable(DependentFileVariable):
    std: float = 0.0
    deciles: List[int] = field(default_factory=dict)

    @classmethod
    def from_file_variable(
            cls, 
            fv: FileVariable, 
            fn_name: str, 
            fn_str: str, 
            depends_on: List[str], 
            std: float = 0.0,
            deciles: List[float] = [],
            fn: Optional[Callable] = None):
        
        return cls(
            name=fv.name, 
            short_name=fv.short_name, 
            description=fv.description, 
            role=fv.role, 
            var_type=fv.var_type, 
            type='int',
            fn_name=fn_name, 
            fn_str=fn_str,
            fn=fn,
            depends_on=depends_on,
            std=std,
            deciles=deciles)

    def __post_init__(self):
        pass

    def get_low_high_quantiles(self, low=0.1, high=0.9) -> Tuple[float, float]:
        assert 0 < low and low < 1
        assert 0 < high and high < 1
        assert low in np.arange(1.1, step=0.1)
        assert high in np.arange(1.1, step=0.1)

        return self.deciles[int(low * 10)], self.deciles[int(high * 10)]
