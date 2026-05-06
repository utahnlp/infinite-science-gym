from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

import pandas as pd


class Comparator:
    pass

class CategoricalComparator(Comparator, Enum):
    IS = 'is'
    IN_SET = 'is one of'

class ContinuousComparator(Comparator, Enum):
    GREATER_THAN = 'is greater than'
    LESS_THAN = 'is less than'
    AT_LEAST = 'is at least'
    AT_MOST = 'is at most'
    IN_RANGE = 'is in the range'

class IntegerComparator(Comparator, Enum):
    EQUALS = 'is'
    NOT_EQUALS = 'is not'
    GREATER_THAN = 'is greater than'
    LESS_THAN = 'is less than'
    AT_LEAST = 'is at least'
    AT_MOST = 'is at most'
    IN_SET = 'is one of'
    IN_RANGE = 'is in the range'


@dataclass
class Condition:
    name: str
    short_name: str
    value: Any
    comparator: Comparator

    def __post_init__(self):
        pass

    def __str__(self):
        v = self.value

        val = f'"{v}"' if self.comparator == CategoricalComparator.IS else str(v)
        if self.comparator == CategoricalComparator.IN_SET:
            val = f'"{v}"'
            if len(v) == 2:
                val = f'"{v[0]}" or "{v[1]}"'
            else:
                val = ''
                for vi in v[:-1]:
                    val += f'"{vi}", '
                val += f'or "{v[-1]}"'
        elif self.comparator == IntegerComparator.IN_SET:
            if len(v) == 2:
                val = f'{v[0]} or {v[1]}'
            else:
                val = ''
                for vi in v[:-1]:
                    val += f'{vi}, '
                val += f'or {v[-1]}'
        elif self.comparator in [ContinuousComparator.IN_RANGE, IntegerComparator.IN_RANGE]:
            val = f'{v[0]}-{v[1]}'

        return f'"{self.short_name}" {self.comparator.value} {val}'
    

    def to_json(self):
        if isinstance(self.comparator, CategoricalComparator):
            comparator_class = 'categorical'
        elif isinstance(self.comparator, ContinuousComparator):
            comparator_class = 'continuous'
        elif isinstance(self.comparator, IntegerComparator):
            comparator_class = 'integer'
        
        return {
            'name': self.name,
            'short_name': self.short_name,
            'value': self.value,
            'comparator_class': comparator_class,
            'comparator': self.comparator.name}


    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.comparator in [CategoricalComparator.IS, IntegerComparator.EQUALS]:
            df = df[df[self.short_name] == self.value]
        elif self.comparator in [IntegerComparator.NOT_EQUALS]:
            df = df[df[self.short_name] != self.value]
        elif self.comparator in [ContinuousComparator.GREATER_THAN, IntegerComparator.GREATER_THAN]:
            df = df[df[self.short_name] > self.value]
        elif self.comparator in [ContinuousComparator.LESS_THAN, IntegerComparator.LESS_THAN]:
            df = df[df[self.short_name] < self.value]
        elif self.comparator in [ContinuousComparator.AT_LEAST, IntegerComparator.AT_LEAST]:
            df = df[df[self.short_name] >= self.value]
        elif self.comparator in [ContinuousComparator.AT_MOST, IntegerComparator.AT_MOST]:
            df = df[df[self.short_name] <= self.value]
        elif self.comparator in [CategoricalComparator.IN_SET, IntegerComparator.IN_SET]:
            df = df[df[self.short_name].isin(self.value)]
        elif self.comparator in [ContinuousComparator.IN_RANGE, IntegerComparator.IN_RANGE]:
            df = df[(df[self.short_name] >= self.value[0]) & (df[self.short_name] <= self.value[1])]
        else:
            raise ValueError(f'unknown comparator: {self.comparator}')

        return df


def condition_from_json(d: Dict[str, Any]) -> Condition:
    if d['comparator_class'] == 'categorical':
        ComparatorClass = CategoricalComparator
    elif d['comparator_class'] == 'continuous':
        ComparatorClass = ContinuousComparator
    elif d['comparator_class'] == 'integer':
        ComparatorClass = IntegerComparator
    
    comparator = ComparatorClass[d['comparator']]

    return Condition(
        name=d['name'],
        short_name=d['short_name'],
        value=d['value'],
        comparator=comparator)
