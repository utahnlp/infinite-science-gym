import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .config import FileConfig
from .types import (
    ContinuousDistribution, 
    DependentFileVariable,
    File, 
    FileSystem, 
    IdentifierFileVariable,
    IndependentFileVariable,
    IndependentIntervalDatetimeVariable,
    IndependentSampledDatetimeVariable,
    IntegerDistribution,
    VariableRole, 
    VariableType, 
)


class FileGenerator:
    logger = logging.getLogger("FileGenerator")

    def __init__(self, file_config: FileConfig):
        self.conf = file_config

    def populate_file(self, fs: FileSystem, path: str, rng: np.random.Generator, num_lines: int = None) -> File:
        
        path_var_dict = self.parse_path(path=path, fs=fs)

        num_lines = num_lines if num_lines else int(rng.integers(*self.conf.num_lines))

        # first populate identifier and independent variables, fill dependent with temporary placeholder
        df = self.get_nondependent_df(fs=fs, num_lines=num_lines, rng=rng)

        # populate dependent variables
        independent_col_names = [fv.short_name for fv in fs.file_variables if fv.role == VariableRole.INDEPENDENT]
        independent_df = df[independent_col_names]
        for fv in fs.file_variables:
            if fv.role != VariableRole.DEPENDENT:
                continue
            df[fv.short_name] = self.populate_dependent(
                fv=fv, 
                path_var_dict=path_var_dict,
                independent_df=independent_df.copy(deep=True),
                rng=rng,
                fillna_on_error=True)
            
        if fs.sort_key is not None and fs.sort_key in df.columns:
            df.sort_values(fs.sort_key, inplace=True)
            
        return File(data=df, extension=fs.path_choices['file_extension'])


    def parse_path(self, path: str, fs: FileSystem) -> Dict[str, Any]:

        def parse_part(part: str, sep: str, num_phs: int, idx: int) -> Tuple[Dict[str, Any], int]:
            part_vars = {}
            for _ in range(num_phs):
                pv = fs.path_variables[idx]
                for v in pv.values:
                    pop_ph = pv.populate(pv.placeholder, v, ch)
                    if pop_ph == part[:len(pop_ph)]:
                        part = part[len(pop_ph)+len(sep):]
                        break
                if pv.short_name in ['date', 'researcher']:
                    v = pop_ph
                part_vars[pv.short_name] = pv.to_type(v)
                idx += 1
            
            return part_vars, idx
            
        ch = fs.path_choices

        *dir_parts, file_str = path.split('/')
        file_root = file_str.split('.')[0]

        path_vars = {}
        idx = 0
        for part, sep, num_phs in zip(dir_parts, ch['dir_inner_separators'], ch['ph_per_level']):            
            part_vars, idx = parse_part(part, sep, num_phs, idx)
            path_vars |= part_vars
        part_vars, _ = parse_part(file_root, ch['file_separator'], ch['file_phs'], idx)
        path_vars |= part_vars

        return path_vars


    def get_nondependent_df(
            self, 
            fs: FileSystem, 
            num_lines: int, 
            rng: np.random.Generator, 
            only_independent_cols: bool = False) -> pd.DataFrame:
        
        # first populate identifier and independent variables, fill dependent with temporary placeholder
        df = pd.DataFrame()
        independent_col_names = []
        for fv in fs.file_variables:
            col_name = fv.short_name
            if fv.role == VariableRole.IDENTIFIER:
                col = self.populate_identifier(fv=fv, num_lines=num_lines)
            elif fv.role == VariableRole.INDEPENDENT:
                col = self.populate_independent(fv=fv, num_lines=num_lines, rng=rng)
                independent_col_names.append(col_name)
            elif fv.role == VariableRole.DEPENDENT:
                col = np.zeros(num_lines)
            df[col_name] = col

        if only_independent_cols:
            return df[independent_col_names]
        return df


    def populate_identifier(self, fv: IdentifierFileVariable, num_lines: int) -> pd.Series:
        raise NotImplementedError

    
    def populate_independent(self, fv: IndependentFileVariable, num_lines: int, rng: np.random.Generator) -> pd.Series:
        if fv.var_type == VariableType.CATEGORICAL:
            values = fv.values
            probs = np.array(fv.probabilities)
            col = rng.choice(values, size=num_lines, p=probs)

        elif fv.var_type == VariableType.CONTINUOUS:
            if fv.distribution == ContinuousDistribution.BETA:
                a, b = fv.params['alpha'], fv.params['beta']
                col = rng.beta(a=a, b=b, size=num_lines)
            elif fv.distribution == ContinuousDistribution.EXPONENTIAL:
                beta = fv.params['beta']
                col = rng.exponential(scale=beta, size=num_lines)
            elif fv.distribution == ContinuousDistribution.NORMAL:
                mean, std = fv.params['mean'], fv.params['std']
                col = rng.normal(loc=mean, scale=std, size=num_lines)
            elif fv.distribution == ContinuousDistribution.UNIFORM:
                low, high = fv.params['low'], fv.params['high']
                col = rng.uniform(low=low, high=high, size=num_lines)
            
        elif fv.var_type == VariableType.INTEGER:
            if fv.distribution == IntegerDistribution.BERNOULLI:
                p = fv.params['p']
                col = rng.binomial(n=1, p=p, size=num_lines)
            elif fv.distribution == IntegerDistribution.BINOMIAL:
                n, p = fv.params['n'], fv.params['p']
                col = rng.binomial(n=n, p=p, size=num_lines)
            elif fv.distribution == IntegerDistribution.GEOMETRIC:
                p = fv.params['p']
                col = rng.geometric(p=p, size=num_lines)
            elif fv.distribution == IntegerDistribution.NEGATIVE_BINOMIAL:
                p, r = fv.params['p'], fv.params['r']
                col = rng.negative_binomial(n=r, p=p, size=num_lines)
            elif fv.distribution == IntegerDistribution.POISSON:
                lam = fv.params['lambda']
                col = rng.poisson(lam=lam, size=num_lines)

        elif fv.var_type == VariableType.DATETIME:
            if isinstance(fv, IndependentIntervalDatetimeVariable):
                # For interval datetimes, generate sequential values from start to end
                datetimes = fv.get_datetimes()
                # Take only the first num_lines values
                datetimes = datetimes[:num_lines]
                # If we have fewer datetimes than num_lines, repeat to fill
                if len(datetimes) < num_lines:
                    # Repeat the pattern if necessary
                    full_cycles = num_lines // len(datetimes)
                    remainder = num_lines % len(datetimes)
                    datetimes = list(datetimes) * full_cycles + list(datetimes[:remainder])
                col = [dt.strftime(fv.fmt) for dt in datetimes[:num_lines]]
            else:  # IndependentSampledDatetimeVariable
                # For sampled datetimes, uniformly sample between start and end
                start_ts = fv.start_time.timestamp()
                end_ts = fv.end_time.timestamp()
                # Generate random timestamps
                random_ts = rng.uniform(start_ts, end_ts, size=num_lines)
                # Convert back to datetimes and format
                col = [datetime.fromtimestamp(ts).strftime(fv.fmt) for ts in random_ts]

        return pd.Series(col)

    def populate_dependent(
            self, 
            fv: DependentFileVariable, 
            path_var_dict: Dict[str, Any],
            independent_df: pd.DataFrame,
            rng: np.random.Generator,
            add_error: bool = True,
            fillna_on_error: bool = False) -> pd.Series:
        
        def _apply(row: pd.Series):
            try:
                val = fv.apply_fn(
                    path_variables=path_var_dict,
                    independent_variables=row.to_dict(),
                    error=row['simulator_error'])
                return val
            except Exception as e:
                if fillna_on_error:
                    return None
                raise e

        if add_error:
            std = 0
            if fv.var_type in [VariableType.CONTINUOUS, VariableType.INTEGER]:
                std = fv.std
            
            independent_df['simulator_error'] = rng.normal(
                loc=0, 
                scale=self.conf.error_std_coeff * std, 
                size=len(independent_df))
        else:
            independent_df['simulator_error'] = np.zeros(len(independent_df))
        
        try:
            new_col = independent_df.apply(_apply, axis=1)
        except Exception as e:
            raise e
        independent_df = independent_df.drop('simulator_error', axis=1)

        return new_col
