from collections import Counter
import logging
import traceback
from typing import Any, List
import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, NearConstantInputWarning

from .config import QAConfig
from .types import (
    Answer,
    BivariateCategoricalStatistic,
    BivariateContinuousStatistic,
    BivariateIntegerStatistic,
    calculate_bivariate_statistic,
    calculate_univariate_statistic,
    CategoricalComparator, 
    Condition, 
    ContinuousComparator, 
    IntegerComparator,
    null_hypothesis_map,
    Question,
    QuestionAnswerPair, 
    QAScope,
    UnivariateCategoricalStatistic,
    UnivariateContinuousStatistic,
    UnivariateIntegerStatistic)
from .utils import signif, is_none_or_nan
from .. import Simulator
from ..types import FileSystem, FileVariable, VariableType, VariableRole

warnings.filterwarnings("error")


class QAGenerator:
    def __init__(self, qa_config: QAConfig, sim: Simulator, fs: FileSystem):
        self.conf = qa_config
        self.sim = sim
        self.fs = fs
        self.logger = logging.getLogger("QAGenerator")


    ########## METADATA ##########


    def has_readme(self) -> QuestionAnswerPair:
        answer = 'yes' if self.fs.tree.readme_path is not None else 'no'
        return QuestionAnswerPair(
            question=Question('Yes or no, does this repository have a README file?'),
            answer=Answer(answer=answer, has_answer=True, _type='str'),
            scope=QAScope.METADATA)
    

    def get_title(self) -> QuestionAnswerPair:
        has_answer = bool(self.fs.readme_visibility.get('title'))
        answer = None
        answer_type = None
        if has_answer:
            answer = self.fs.story.title
            answer_type = 'str'
        return QuestionAnswerPair(
            question=Question('Is the project title in the README file? If yes, what is it?'),
            answer=Answer(answer=answer, has_answer=has_answer, _type=answer_type),
            scope=QAScope.METADATA)
    

    def get_abstract(self) -> QuestionAnswerPair:
        has_answer = bool(self.fs.readme_visibility.get('abstract'))
        answer = None
        answer_type = None
        if has_answer:
            answer = self.fs.story.abstract
            answer_type = 'str'
        return QuestionAnswerPair(
            question=Question('Is the project abstract in the README file? If yes, what is it?'),
            answer=Answer(answer=answer, has_answer=has_answer, _type=answer_type),
            scope=QAScope.METADATA)
    

    ########## DIRECTORY ##########

    
    def file_extension(self) -> QuestionAnswerPair:
        return QuestionAnswerPair(
            question=Question('What is the file extension for the data in this repository?'),
            answer=Answer(answer=self.fs.path_choices['file_extension'], has_answer=True, _type='str'),
            scope=QAScope.DIRECTORY)


    def count_files_prefix(self, seed: int = 0, prefix: str = None) -> QuestionAnswerPair:
        if prefix is None:
            rng = np.random.default_rng(seed=seed)
            path = rng.choice(self.fs.tree.get_paths(exclude_readme=True))
            prefix = path[:rng.integers(len(path))] + '*'
        question = 'How many files in this repository have the prefix: "{prefix}"?'
        return QuestionAnswerPair(
            question=Question(question=question, variables={'{prefix}': prefix}),
            answer=Answer(answer=len(self.fs.tree.get_paths(prefix=prefix)), has_answer=True, _type='int'),
            scope=QAScope.DIRECTORY)


    def count_files_conditions(self, seed: int = 0, n_conditions: int = None) -> QuestionAnswerPair:
        rng = np.random.default_rng(seed)
        
        if n_conditions is None:
            n_conditions = rng.integers(self.conf.max_path_conditions + 1)
        assert n_conditions >= 0, f'{n_conditions=} must be a non-negative integer'
        if n_conditions > len(self.fs.path_variables):
            self.logger.warning(
                f'{n_conditions=} is bigger than available path variables, '
                f'setting to {len(self.fs.path_variables)=}')
            n_conditions = len(self.fs.path_variables)
        
        question = 'How many files are in this repository'
        paths = self.fs.tree.get_paths(exclude_readme=False)
        conditions = []
        if n_conditions > 0:
            conditions = self._get_path_conditions(n_conditions, rng)
            paths = self._filter_paths(conditions, exclude_readme=False)
            question += ' ' + self._filter_clause(conditions)
        question += '?'

        return QuestionAnswerPair(
            question=Question(question),
            answer=Answer(answer=len(paths), has_answer=True, _type='int'),
            scope=QAScope.DIRECTORY,
            path_conditions=conditions)

    
    ########## FILE ##########


    def count_rows(self, seed: int = 0, path: str = None) -> QuestionAnswerPair:
        if path is None:
            rng = np.random.default_rng(seed=seed)
            path = rng.choice(self.fs.tree.get_paths(exclude_readme=True))

        file, status = self.sim.read_file(seed=self.fs.seed, path=path, fs=self.fs)
        question = 'How many rows of data (excluding headers) are in the file: "{path}"?'
        return QuestionAnswerPair(
            question=Question(question=question, variables={'{path}': path}),
            answer=Answer(answer=len(file.data), has_answer=True, _type='int'),
            scope=QAScope.SINGLE_FILE)


    def univariate_statistic_single_file(self, seed: int = 0, path: str = None, n_file_conds: int = None) -> QuestionAnswerPair:
        rng = np.random.default_rng(seed)
        sig_figs = rng.integers(*self.conf.sig_figs)

        if path is None:
            path = rng.choice(self.fs.tree.get_paths(exclude_readme=True))

        if n_file_conds is None:
            n_file_conds = rng.integers(self.conf.max_file_conditions + 1)
        assert n_file_conds >= 0, f'{n_file_conds=} must be a non-negative integer'
        if n_file_conds > len(self.fs.file_variables) - 1:
            self.logger.warning(
                f'{n_file_conds=} is bigger than available file variables, '
                f'setting to {len(self.fs.file_variables)=}')
            n_file_conds = len(self.fs.file_variables) - 1

        file_clause = None
        file_conditions = []

        # get file conditions and filter rows using conditions
        eligible_fvs = [x for x in self.fs.file_variables if x.role != VariableRole.IDENTIFIER]
        univariate_fv = rng.choice(eligible_fvs)
        if n_file_conds > 0:
            file_conditions = self._get_file_conditions(
                num_conditions=n_file_conds, 
                rng=rng, 
                eligible=eligible_fvs)
            vals = self._filter_rows([path], [univariate_fv], file_conditions)[0]
            file_clause = self._filter_clause(file_conditions)
        else:
            vals = self._filter_rows([path], [univariate_fv], [])[0]

        # determine univariate statistic
        if univariate_fv.var_type == VariableType.CATEGORICAL:
            statistic = rng.choice(UnivariateCategoricalStatistic)
            answer_type = 'str'
        elif univariate_fv.var_type == VariableType.CONTINUOUS:
            statistic = rng.choice(UnivariateContinuousStatistic)
            answer_type = 'float'
        elif univariate_fv.var_type == VariableType.INTEGER:
            statistic = rng.choice(UnivariateIntegerStatistic)
            ISS = UnivariateIntegerStatistic
            answer_type = 'float'
            if statistic in [ISS.MIN, ISS.MAX, ISS.MEDIAN]:
                answer_type = 'int'
        
        # calculate answer value
        if len(vals):
            has_answer = True
            answer = calculate_univariate_statistic(vals, statistic)
            if is_none_or_nan(answer):
                has_answer = False
                answer = None
            else:
                if answer_type in ['int', 'float']:
                    answer = signif(answer, sig_figs)
                answer = self._to_type(answer, answer_type)
        else:
            has_answer = False
            answer = None

        # construct natural language question
        question = 'In the file "{path}", '
        if n_file_conds > 0:
            question += f'only considering rows {file_clause}, '
        question += f'what is the {statistic.value} of the "{univariate_fv.short_name}" variable? '
        if answer_type in ['int', 'float']:
            question += f'If your answer is numeric, report {sig_figs} significant figures. '

        return QuestionAnswerPair(
            question=Question(question=question, variables={'{path}': path}),
            answer=Answer(answer=answer, has_answer=has_answer, _type=(answer_type if has_answer else None)),
            scope=QAScope.SINGLE_FILE,
            file_conditions=file_conditions)
        

    def univariate_statistic_conditions(self, seed: int = 0, n_path_conds: int = None, n_file_conds: int = None) -> QuestionAnswerPair:
        rng = np.random.default_rng(seed)
        sig_figs = rng.integers(*self.conf.sig_figs)

        if n_path_conds is None:
            n_path_conds = rng.integers(self.conf.max_path_conditions + 1)
        assert n_path_conds >= 0, f'{n_path_conds=} must be a non-negative integer'
        if n_path_conds > len(self.fs.path_variables):
            self.logger.warning(
                f'{n_path_conds=} is bigger than available path variables, '
                f'setting to {len(self.fs.path_variables)=}')
            n_path_conds = len(self.fs.path_variables)

        if n_file_conds is None:
            n_file_conds = rng.integers(self.conf.max_file_conditions + 1)
        assert n_file_conds >= 0, f'{n_file_conds=} must be a non-negative integer'
        if n_file_conds > len(self.fs.file_variables) - 1:
            self.logger.warning(
                f'{n_file_conds=} is bigger than available file variables, '
                f'setting to {len(self.fs.file_variables)=}')
            n_file_conds = len(self.fs.file_variables) - 1

        path_clause, file_clause = None, None
        path_conditions, file_conditions = [], []

        # get path conditions and filter tree using condititions
        paths = self.fs.tree.get_paths(exclude_readme=True)
        qa_scope = QAScope.MULTIPLE_FILES
        if n_path_conds > 0:
            path_conditions = self._get_path_conditions(n_path_conds, rng)
            paths = self._filter_paths(path_conditions, exclude_readme=True)
            path_clause = self._filter_clause(path_conditions)
            if len(paths) == 0:
                qa_scope = QAScope.NO_FILES
            elif len(paths) == 1:
                qa_scope = QAScope.SINGLE_FILE
            
        # get file conditions and filter rows using conditions
        eligible_fvs = [x for x in self.fs.file_variables if x.role != VariableRole.IDENTIFIER]
        univariate_fv = rng.choice(eligible_fvs)
        if n_file_conds > 0:
            file_conditions = self._get_file_conditions(
                num_conditions=n_file_conds, 
                rng=rng, 
                eligible=eligible_fvs)
            vals = self._filter_rows(paths, [univariate_fv], file_conditions)[0]
            file_clause = self._filter_clause(file_conditions)
        else:
            vals = self._filter_rows(paths, [univariate_fv], [])[0]

        # determine univariate statistic
        if univariate_fv.var_type == VariableType.CATEGORICAL:
            statistic = rng.choice(UnivariateCategoricalStatistic)
            answer_type = 'str'
        elif univariate_fv.var_type == VariableType.CONTINUOUS:
            statistic = rng.choice(UnivariateContinuousStatistic)
            answer_type = 'float'
        elif univariate_fv.var_type == VariableType.INTEGER:
            statistic = rng.choice(UnivariateIntegerStatistic)
            ISS = UnivariateIntegerStatistic
            answer_type = 'float'
            if statistic in [ISS.MIN, ISS.MAX, ISS.MEDIAN]:
                answer_type = 'int'
        
        # calculate answer value
        if len(vals):
            has_answer = True
            answer = calculate_univariate_statistic(vals, statistic)
            if is_none_or_nan(answer):
                has_answer = False
                answer = None
            else:
                if answer_type in ['int', 'float']:
                    answer = signif(answer, sig_figs)
                answer = self._to_type(answer, answer_type)
        else:
            has_answer = False
            answer = None

        # construct natural language question
        if n_path_conds == 0:
            question = 'Across all files, '
        else:
            question = f'Only considering files {path_clause}, '
        if n_file_conds > 0:
            if n_path_conds > 0:
                question +='and '
            question += f'only considering rows {file_clause}, '
        question += f'what is the {statistic.value} of the "{univariate_fv.short_name}" variable? '
        if answer_type in ['int', 'float']:
            question += f'If your answer is numeric, report {sig_figs} significant figures. '

        return QuestionAnswerPair(
            question=Question(question),
            answer=Answer(answer=answer, has_answer=has_answer, _type=(answer_type if has_answer else None)),
            scope=qa_scope,
            path_conditions=path_conditions,
            file_conditions=file_conditions)


    def bivariate(
            self,
            bivariate_qn_type: str, 
            seed: int = 0, 
            path: str = None, 
            n_file_conds: int = None) -> QuestionAnswerPair:
        
        assert bivariate_qn_type in ['statistic value', 'hypothesis test']
        
        rng = np.random.default_rng(seed)
        sig_figs = rng.integers(*self.conf.sig_figs)

        if path is None:
            path = rng.choice(self.fs.tree.get_paths(exclude_readme=True))

        if n_file_conds is None:
            n_file_conds = rng.integers(self.conf.max_file_conditions + 1)
        assert n_file_conds >= 0, f'{n_file_conds=} must be a non-negative integer'
        if n_file_conds > len(self.fs.file_variables) - 2:
            self.logger.warning(
                f'{n_file_conds=} is bigger than available file variables, '
                f'setting to {len(self.fs.file_variables)=}')
            n_file_conds = len(self.fs.file_variables) - 2
            
        # choose statistic and file variables
        c = Counter([v.var_type for v in self.fs.file_variables])
        type_choices = [t for t, count in c.items() if count >= 2]
        chosen_type = rng.choice(type_choices)
        if chosen_type == VariableType.CATEGORICAL:
            statistic = rng.choice(BivariateCategoricalStatistic)
        elif chosen_type == VariableType.CONTINUOUS:
            statistic = rng.choice(BivariateContinuousStatistic)
        else: # VariableType.INTEGER
            statistic = rng.choice(BivariateIntegerStatistic)
        statistic_str = rng.choice(statistic.value)
        eligible_fvs = [fv for fv in self.fs.file_variables if fv.var_type == chosen_type]
        bivariate_fvs = rng.choice(eligible_fvs, size=2, replace=False)

        # get file conditions and filter rows using conditions
        eligible_fvs = [
            x for x in self.fs.file_variables 
            if (x.role != VariableRole.IDENTIFIER) and (x not in bivariate_fvs)]
        if n_file_conds > 0:
            file_conditions = self._get_file_conditions(
                num_conditions=n_file_conds, 
                rng=rng, 
                eligible=eligible_fvs)
            vals = self._filter_rows([path], bivariate_fvs, file_conditions)
            file_clause = self._filter_clause(file_conditions)
        else:
            vals = self._filter_rows([path], bivariate_fvs, [])

        # calculate answer value
        if len(vals) and len(vals[0]) > 1:
            has_answer = True
            try:
                value, pvalue = calculate_bivariate_statistic(vals, statistic)
                if is_none_or_nan(value) or is_none_or_nan(pvalue):
                    has_answer = False
                    value = None
                    pvalue = None
            except (ConstantInputWarning, NearConstantInputWarning):
                has_answer = False
                value = None
                pvalue = None
            except Exception as e:
                self.logger.warning(f'skipping providing answer to bivariate statistic due to exception:\n{traceback.format_exc()}')
                has_answer = False
                value = None
                pvalue = None
        else:
            has_answer = False
            value = None
            pvalue = None
        if bivariate_qn_type == 'statistic value':
            answer = None 
            if has_answer:
                answer = signif(value, sig_figs)
            answer_type = 'float'
        else: # hypothesis test
            significance_level = rng.choice([0.01, 0.05])
            answer = None
            if has_answer:
                answer = 'yes' if pvalue < significance_level else 'no'
            answer_type = 'str'

        # construct natural language question
        bfv1, bfv2 = bivariate_fvs
        question = 'In the file "{path}", '
        if n_file_conds > 0:
            question += f'only considering rows {file_clause}, '
        if bivariate_qn_type == 'statistic value':
            question += f'what is the "{statistic_str}" value between the "{bfv1.short_name}" variable and the "{bfv2.short_name}" variable?'
            question += f' If your answer is numeric, report {sig_figs} significant figures.'
        else: # hypothesis test
            relationship = null_hypothesis_map[statistic]
            question += (
                f'using "{statistic_str}" and a p-value of {significance_level}, can you reject the null hypothesis (yes/no) that '
                f'there is no {relationship} between the "{bfv1.short_name}" and "{bfv2.short_name}" variables?')            
    
        return QuestionAnswerPair(
            question=Question(question=question, variables={'{path}': path}),
            answer=Answer(answer=answer, has_answer=has_answer, _type=(answer_type if has_answer else None)),
            scope=QAScope.SINGLE_FILE)
    

    def bivariate_statistic(self, seed: int = 0, path: str = None, n_file_conds: int = None):
        return self.bivariate(bivariate_qn_type='statistic value', seed=seed, path=path, n_file_conds=n_file_conds)


    def bivariate_hypothesis(self, seed: int = 0, path: str = None, n_file_conds: int = None):
        return self.bivariate(bivariate_qn_type='hypothesis test', seed=seed, path=path, n_file_conds=n_file_conds)


    ########## HELPER FUNCTIONS ##########


    def _get_path_conditions(self, num_conditions: int, rng: np.random.Generator) -> List[Condition]:
        conds = []
        cond_pvs = rng.choice(self.fs.path_variables, size=num_conditions, replace=False)
        for pv in cond_pvs:
            comparator = rng.choice(CategoricalComparator)
            if comparator == CategoricalComparator.IS:
                val = str(rng.choice(pv.values))
            elif comparator == CategoricalComparator.IN_SET:
                size = rng.choice(list(range(min(1, len(pv.values)), max(2, len(pv.values)))))
                if size == 1:
                    val = str(pv.values[0])
                    comparator = CategoricalComparator.IS
                else:
                    val = sorted([str(x) for x in rng.choice(pv.values, size=size, replace=False)])
            else:
                raise ValueError(comparator)
            conds.append(Condition(name=pv.name, short_name=pv.short_name, value=val, comparator=comparator))
        return conds


    def _filter_paths(self, conditions: List[Condition], exclude_readme: bool = False) -> List[str]:
        assert all((type(c.comparator) == CategoricalComparator) for c in conditions), (
            'all conditions must use CategoricalComparator')
    
        paths = self.fs.tree.get_paths(exclude_readme=exclude_readme)

        filtered = [x for x in paths]
        for c in conditions:
            if c.comparator == CategoricalComparator.IS:
                filtered = [x for x in paths if c.value in x]
            elif c.comparator == CategoricalComparator.IN_SET:
                filtered = []
                for x in paths:
                    for v in c.value:
                        if v in x:
                            filtered.append(x)
            else:
                raise ValueError(c)
        return filtered
    

    def _get_file_conditions(self, num_conditions: int, rng: np.random.Generator, eligible: List[FileVariable] = None) -> List[Condition]:
        
        sig_figs = rng.integers(*self.conf.sig_figs)
        quantiles = self.conf.dist_sample_quants
        
        conds = []
        fvs = eligible if eligible else self.fs.file_variables
        cond_fvs = rng.choice(fvs, size=num_conditions, replace=False)
        for fv in cond_fvs:
            
            if fv.var_type == VariableType.CATEGORICAL:
                if len(fv.values) == 0:
                    continue

                comparator = rng.choice(CategoricalComparator)
                if comparator == CategoricalComparator.IS:
                    val = str(rng.choice(fv.values))
                elif comparator == CategoricalComparator.IN_SET:
                    size = rng.choice(list(range(min(1, len(fv.values)), max(2, len(fv.values)))))
                    if size == 1:
                        val = str(fv.values[0])
                        comparator = CategoricalComparator.IS
                    else:
                        val = sorted([str(x) for x in rng.choice(fv.values, size=size, replace=False)])
                else:
                    raise ValueError(comparator)

            elif fv.var_type == VariableType.CONTINUOUS:
                comparator = rng.choice(ContinuousComparator)
                low, high = fv.get_low_high_quantiles(*quantiles)
                if comparator == ContinuousComparator.IN_RANGE:
                    val = sorted(rng.uniform(low, high, size=2))
                    val = [signif(float(x), sig_figs) for x in val]
                    if val[0] == val[1]:
                        comparator = ContinuousComparator.AT_MOST
                        val = val[0]
                else:
                    val = signif(float(rng.uniform(low, high)), sig_figs)
                
            elif fv.var_type == VariableType.INTEGER:
                comparator = rng.choice(IntegerComparator)
                low, high = fv.get_low_high_quantiles(*quantiles)

                if comparator == IntegerComparator.IN_SET:
                    if high-low+1 < 3:
                        high = low + 2
                    size = rng.choice(list(range(2, high-low+1)))
                    val = sorted(rng.choice(list(range(low, high+1)), size=size, replace=False))
                    val = [signif(int(x), sig_figs) for x in val]
                elif comparator == IntegerComparator.IN_RANGE:
                    if high-low+1 == 1:
                        high += 1
                    val = sorted(rng.choice(list(range(low, high+1)), size=2, replace=False))
                    val = [signif(int(x), sig_figs) for x in val]
                else:
                    val = signif(int(rng.choice(list(range(low, high+1)))), sig_figs)
            
            else:
                raise ValueError(
                    f'{type(fv)=} must be one of IndependentCategoricalVariable, '
                    'IndependentContinuousVariable, or IndependentIntegerVariable')

            conds.append(Condition(name=fv.name, short_name=fv.short_name, value=val, comparator=comparator))
        
        return conds
            

    def _filter_rows(self, paths: List[str], cols: List[FileVariable], conditions: List[Condition]) -> List[np.ndarray]:
        
        vals = [np.empty((1)) for _ in cols]
        for path in paths:
            file, status = self.sim.read_file(seed=self.fs.seed, path=path, fs=self.fs)
            df = file.data
            for c in conditions:
                df = c.filter_df(df)
            for i, col in enumerate(cols):
                vals[i] = np.concatenate((vals[i], df[col.short_name].to_numpy()))

        return vals
    

    def _filter_clause(self, conditions: List[Condition]) -> str:
        clause = f'where the {conditions[0]}'
        if len(conditions) == 2:
            clause += f' and the {conditions[1]}'
        elif len(conditions) > 2:
            for c in conditions[1:-1]:
                clause += f', the {c}'
            clause += f', and the {conditions[-1]}'
        return clause
    

    def _to_type(self, answer: Any, _type: str):        
        if _type == 'int':
            return int(answer)
        elif _type == 'float':
            return float(answer)
        elif _type == 'str':
            return str(answer)
        elif _type == 'bool':
            return bool(answer)
        raise ValueError(_type)
