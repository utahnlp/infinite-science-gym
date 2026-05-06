from enum import Enum
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chisquare, pearsonr, spearmanrho


class BivariateStatistic:
    pass

class BivariateCategoricalStatistic(BivariateStatistic, Enum):
    CHI_SQUARE_TEST = ['chi-square', 'chi-squared test']

class BivariateContinuousStatistic(BivariateStatistic, Enum):
    PEARSON_R = ['Pearson correlation coefficient', 'Pearson correlation', 'Pearson R']
    SPEARMAN_RHO = [
        'Spearman rho correlation coefficient', 'Spearman r correlation coefficient', 
        "Spearman's rho", "Spearman's rank order correlation coefficient"]

class BivariateIntegerStatistic(BivariateStatistic, Enum):
    pass

BivariateIntegerStatistic = BivariateContinuousStatistic


null_hypothesis_map = {
    BivariateCategoricalStatistic.CHI_SQUARE_TEST: 'relationship',
    BivariateContinuousStatistic.PEARSON_R: 'linear relationship',
    BivariateContinuousStatistic.SPEARMAN_RHO: 'monotonic relationship',
}



def calculate_bivariate_statistic(vals: List[Any], statistic: BivariateStatistic) -> Tuple[float, float]:
    assert len(vals) == 2, len(vals)
    assert len(vals[0]) == len(vals[1]), (len(vals[0]), len(vals[1]))

    if statistic == BivariateCategoricalStatistic.CHI_SQUARE_TEST:
        df = pd.DataFrame({'A': vals[0].astype(str), 'B': vals[1].astype(str)})
        all_values = sorted(list(set(df['A']).union(set(df['B']))))
        cooc_matrix = pd.DataFrame(0, index=all_values, columns=all_values)
        for index, row in df.iterrows():
            item_a = row['A']
            item_b = row['B']
            cooc_matrix.loc[item_a, item_b] += 1
            # The matrix is often symmetrical, so you might also increment the inverse cell
            if item_a != item_b:
                cooc_matrix.loc[item_b, item_a] += 1
        result = chisquare(cooc_matrix, axis=None)
        return float(result.statistic), float(result.pvalue)
    elif statistic == BivariateContinuousStatistic.PEARSON_R:
        vals = np.array(vals).astype(float)
        result = pearsonr(vals[0], vals[1])
        return float(result.statistic), float(result.pvalue)
    elif statistic in[BivariateContinuousStatistic.SPEARMAN_RHO]:
        vals = np.array(vals).astype(float)
        result = spearmanrho(vals[0].astype(float), vals[1].astype(float))
        return float(result.statistic), float(result.pvalue)
