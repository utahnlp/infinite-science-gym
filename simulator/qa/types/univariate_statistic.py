from collections import Counter
from enum import Enum
from typing import Any, List

import numpy as np


class UnivariateStatistic:
    pass

class UnivariateCategoricalStatistic(UnivariateStatistic, Enum):
    MOST_COMMON = 'most common value'
    # LEAST_COMMON = 'least common value'

class UnivariateContinuousStatistic(UnivariateStatistic, Enum):
    MIN = 'minimum value'
    MAX = 'maximum value'
    MEDIAN = 'median value'
    MEAN = 'average value'
    STD_DEV = 'standard deviation'
    VARIANCE = 'variance'

# these are the same for now
UnivariateIntegerStatistic = UnivariateContinuousStatistic


def calculate_univariate_statistic(vals: List[Any], statistic: UnivariateStatistic) -> Any:
    if statistic == UnivariateCategoricalStatistic.MOST_COMMON:
        return Counter(vals).most_common()[0][0]
    # elif statistic == CategoricalUnivariateStatistic.LEAST_COMMON:
    #     return Counter(vals).most_common()[0][0]
    elif statistic in [UnivariateContinuousStatistic.MIN, UnivariateIntegerStatistic.MIN]:
        return min(vals)
    elif statistic in [UnivariateContinuousStatistic.MAX, UnivariateIntegerStatistic.MAX]:
        return max(vals)
    elif statistic in [UnivariateContinuousStatistic.MEDIAN, UnivariateIntegerStatistic.MEDIAN]:
        return np.median(vals)
    elif statistic in [UnivariateContinuousStatistic.MEAN, UnivariateIntegerStatistic.MEAN]:
        return np.mean(vals)
    elif statistic in [UnivariateContinuousStatistic.STD_DEV, UnivariateIntegerStatistic.STD_DEV]:
        return np.std(vals)
    elif statistic in [UnivariateContinuousStatistic.VARIANCE, UnivariateIntegerStatistic.VARIANCE]:
        return np.var(vals)
    else:
        raise ValueError(f'{statistic=} must be of type UnivariateStatistic')
