from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Tuple

from scipy import stats

from . import FileVariable


class ContinuousDistribution(Enum):
    BETA = ['alpha', 'beta']
    EXPONENTIAL = ['beta']
    NORMAL = ['mean', 'std']
    UNIFORM = ['low', 'high']

class IntegerDistribution(Enum):
    BERNOULLI = ['p']
    BINOMIAL = ['n', 'p']
    GEOMETRIC = ['p']
    NEGATIVE_BINOMIAL = ['p', 'r']
    POISSON = ['lambda']


@dataclass
class IndependentFileVariable(FileVariable):
    pass

@dataclass
class IndependentCategoricalVariable(IndependentFileVariable):
    values: List[str]
    probabilities: List[float]

    @classmethod
    def from_file_variable(cls, v: FileVariable, values: List[str], probabilities: List[float]):
        return cls(
            name=v.name, 
            short_name=v.short_name, 
            description=v.description, 
            role=v.role, 
            var_type=v.var_type, 
            type='str',
            values=values, 
            probabilities=probabilities)

    def __post_init__(self):
        assert len(self.values) == len(self.probabilities), f'{len(self.values)=} needs to be same as {len(self.probabilities)=}'
        
        # normalize if not normalized
        if sum(self.probabilities) != 1:
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
class IndependentContinuousVariable(IndependentFileVariable):
    distribution: ContinuousDistribution
    params: Dict[str, float | int]

    @classmethod
    def from_file_variable(cls, v: FileVariable, distribution: ContinuousDistribution, params: Dict[str, float | int]):
        return cls(
            name=v.name, 
            short_name=v.short_name, 
            description=v.description, 
            role=v.role, 
            var_type=v.var_type, 
            type='float',
            distribution=distribution, 
            params=params)

    def __post_init__(self):
        for k, v in self.params.items():
            assert k in self.distribution.value, f'{k} not in {self.distribution.value}'
            assert type(v) in [int, float], f'{k}: {v} range values need to be of type float'
        for k in self.distribution.value:
            assert k in self.params, f'{k} not in {self.params=}'

    def get_low_high_quantiles(self, low=0.1, high=0.9) -> Tuple[float, float]:
        assert 0 < low and low < 1
        assert 0 < high and high < 1

        if self.distribution == ContinuousDistribution.BETA:
            a, b = self.params['alpha'], self.params['beta']
            return stats.beta.ppf([low, high], a=a, b=b)
        elif self.distribution == ContinuousDistribution.EXPONENTIAL:
            b = self.params['beta']
            return stats.expon.ppf([low, high], scale=b)
        elif self.distribution == ContinuousDistribution.NORMAL:
            m, s = self.params['mean'], self.params['std']
            return stats.norm.ppf([low, high], loc=m, scale=s)
        elif self.distribution == ContinuousDistribution.UNIFORM:
            l = self.params['low']
            r = self.params['high'] - l
            return stats.uniform.ppf([low, high], loc=l, scale=r)
        
        raise ValueError(f'{self.distribution=} must be of type ContinuousDistribution')


@dataclass
class IndependentIntegerVariable(IndependentFileVariable):
    distribution: IntegerDistribution
    params: Dict[str, float | int]

    @classmethod
    def from_file_variable(cls, v: FileVariable, distribution: IntegerDistribution, params: Dict[str, float | int]):
        return cls(
            name=v.name, 
            short_name=v.short_name, 
            description=v.description, 
            role=v.role, 
            var_type=v.var_type, 
            type='int',
            distribution=distribution, 
            params=params)

    def __post_init__(self):
        for k, v in self.params.items():
            assert k in self.distribution.value, f'{k} not in {self.distribution.value}'
            assert type(v) in [int, float], f'{k}: {v} range values need to be of type float'
        for k in self.distribution.value:
            assert k in self.params, f'{k} not in {self.params=}'

        if self.distribution == IntegerDistribution.BINOMIAL:
            self.params['n'] = int(self.params['n'])
            

    def get_low_high_quantiles(self, low=0.1, high=0.9) -> Tuple[float, float]:
        assert 0 < low and low < 1
        assert 0 < high and high < 1

        if self.distribution == IntegerDistribution.BERNOULLI:
            p = self.params['p']
            quantiles = stats.bernoulli.ppf([low, high], p=p)
        elif self.distribution == IntegerDistribution.BINOMIAL:
            n, p = self.params['n'], self.params['p']
            quantiles = stats.binom.ppf([low, high], n=n, p=p)
        elif self.distribution == IntegerDistribution.GEOMETRIC:
            p = self.params['p']
            quantiles = stats.geom.ppf([low, high], p=p)
        elif self.distribution == IntegerDistribution.NEGATIVE_BINOMIAL:
            p, r = self.params['p'], self.params['r']
            quantiles = stats.nbinom.ppf([low, high], n=r, p=p)
        elif self.distribution == IntegerDistribution.POISSON:
            l = self.params['lambda']
            quantiles = stats.poisson.ppf([low, high], mu=l)
        else:
            raise ValueError(f'{self.distribution=} must be of type IntegerDistribution')
        return [int(x) for x in quantiles]


@dataclass
class IndependentDatetimeVariable(IndependentFileVariable):
    """Base class for independent datetime variables."""
    start_time: datetime
    end_time: datetime
    fmt: str

    @classmethod
    def from_file_variable(cls, v: FileVariable, start_time: datetime, end_time: datetime, fmt: str):
        return cls(
            name=v.name,
            short_name=v.short_name,
            description=v.description,
            role=v.role,
            var_type=v.var_type,
            type='datetime',
            start_time=start_time,
            end_time=end_time,
            fmt=fmt)

    def __post_init__(self):
        assert isinstance(self.start_time, datetime), f'start_time must be a datetime object'
        assert isinstance(self.end_time, datetime), f'end_time must be a datetime object'
        assert self.start_time < self.end_time, f'start_time must be before end_time'
        assert isinstance(self.fmt, str), f'fmt must be a string'
        # Validate format string by attempting to format the start_time
        try:
            self.start_time.strftime(self.fmt)
        except Exception as e:
            raise ValueError(f'Invalid format string {self.fmt!r}: {e}')

    def to_type(self, v: str) -> datetime:
        """Convert a formatted string to datetime using the variable's format."""
        return datetime.strptime(v, self.fmt)


@dataclass
class IndependentIntervalDatetimeVariable(IndependentDatetimeVariable):
    """Represents sequential datetimes at a known cadence from start to end time."""
    interval: timedelta

    @classmethod
    def from_file_variable(cls, v: FileVariable, start_time: datetime, end_time: datetime, interval: timedelta, fmt: str):
        return cls(
            name=v.name,
            short_name=v.short_name,
            description=v.description,
            role=v.role,
            var_type=v.var_type,
            type='datetime',
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            fmt=fmt)

    def __post_init__(self):
        assert isinstance(self.interval, timedelta), f'interval must be a timedelta object'
        assert self.interval.total_seconds() > 0, f'interval must be positive'
        super().__post_init__()

    def get_datetimes(self) -> List[datetime]:
        """Generate all datetime values in the interval."""
        datetimes = []
        current = self.start_time
        while current <= self.end_time:
            datetimes.append(current)
            current += self.interval
        return datetimes


@dataclass
class IndependentSampledDatetimeVariable(IndependentDatetimeVariable):
    """Represents datetimes sampled uniformly between start and end time."""

    @classmethod
    def from_file_variable(cls, v: FileVariable, start_time: datetime, end_time: datetime, fmt: str):
        return cls(
            name=v.name,
            short_name=v.short_name,
            description=v.description,
            role=v.role,
            var_type=v.var_type,
            type='datetime',
            start_time=start_time,
            end_time=end_time,
            fmt=fmt)
