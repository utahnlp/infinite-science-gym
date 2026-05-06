from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import product
import logging
import math
import re
import traceback
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
# from transformers import GenerationConfig

from .file import FileGenerator
from .llm import LLM, IncorrectFormatLLMOutputException
from .prompts import FileSystemConfigPrompt, PlaceholderPrompt, VariablePrompt
from .utils import set_global_seed, signif
from .config import FileConfig, FileSystemConfig
from .types import (
    ContinuousDistribution,
    DependentFileVariable,
    DependentCategoricalVariable,
    DependentContinuousVariable,
    DependentIntegerVariable,
    DirectoryTree, 
    PATH_CHOICES,
    FileSystem, 
    FileVariable, 
    IdentifierFileVariable,
    IndependentCategoricalVariable,
    IndependentContinuousVariable,
    IndependentDatetimeVariable,
    IndependentFileVariable,
    IndependentIntervalDatetimeVariable,
    IndependentIntegerVariable,
    IndependentSampledDatetimeVariable,
    IntegerDistribution,
    PathVariable,
    PathVariableLocation,
    PLACEHOLDERS,
    Story,
    VariableRole,
    VariableType,
    default_variable_type_map)


class FileSystemGenerator:
    """
    Generates a single file system config using a Python dataclass schema, with LLM and random seed.
    """
    logger = logging.getLogger("FileSystemGenerator")

    def __init__(self, fs_config: FileSystemConfig, file_config: FileConfig, llm: LLM):
        self.conf = fs_config
        self.llm = llm
        self.file_generator = FileGenerator(file_config=file_config)

    def generate_config(
            self, 
            story: Story, 
            seed: int, 
            num_files: int = None, 
            num_file_columns: int = None) -> FileSystem:
        
        set_global_seed(seed)
        rng = np.random.default_rng(seed)

        # optionally populate README.md
        readme_visibility, readme_path, readme_str = self._populate_readme(story=story, rng=rng)
        metadata_visibility = self._determine_metadata_visibility(rng=rng)

        # generate the directory and file templates
        self.logger.info('  Generate templates')
        path_choices = {k: str(rng.choice(v)) for k, v in PATH_CHOICES.items()}
        dir_template, file_template, sep_choices, var_names = self._generate_templates_with_llm(
            placeholders=PLACEHOLDERS, 
            story=story, 
            chosen_ext=path_choices["file_extension"], 
            rng=rng)
        path_choices = path_choices | sep_choices
        self.logger.debug(f'{path_choices=}\n{dir_template=}\n{file_template=}\n{var_names=}')

        # extract and parse path variables from templates 
        path_variables = []
        i = 0
        for template, loc in [(dir_template, 'dir_name'), (file_template, 'file_name')]:
            for ph in re.findall(r"\{([^}]+)\}", template):
                if 'var' in ph:
                    ph_dir = deepcopy(PLACEHOLDERS['var'])
                    ph_dir['name'] = var_names[i]
                    i += 1
                else:
                    ph_dir = PLACEHOLDERS[ph]
                path_variable = PathVariable(
                    name=ph_dir['name'], 
                    short_name=ph, 
                    description=ph_dir['description'],
                    placeholder=f'{{{ph}}}',
                    category=ph_dir['cat'],
                    type=ph_dir['type'],
                    location=PathVariableLocation(loc))
                path_variables.append(path_variable)
        self.logger.debug(f'{path_variables=}')
        
        # Populate directory and filename placeholders
        self.logger.info('  Populate placeholders')
        path_variables = self._populate_placeholders(
            path_variables=path_variables,
            story=story,
            path_choices=path_choices,
            rng=rng)
        self.logger.debug(f'{path_variables=}')
        
        # check for size of path variable cartesian cross product
        num_possible_files = math.prod([len(ph.values) for ph in path_variables])
        max_files = min(num_possible_files, self.conf.max_num_files)
        min_files = self.conf.min_num_files
        if not num_files:
            if max_files > min_files:
                if self.conf.num_files_dist is not None:
                    dist, p = self.conf.num_files_dist
                    if dist == 'beta':
                        num_files = min_files + int(rng.beta(**p) * (max_files - min_files))
                    else:
                        raise NotImplementedError(f'{self.conf.num_files_dist}')
                else:
                    num_files = int(rng.integers(min_files, max_files))
            else:
                self.logger.critical(f'NEED TO FIX WHAT HAPPENS IF LESS THAN {self.conf.min_num_files=} POSSIBLE FILES')
                num_files = max_files  
        if num_possible_files < num_files:
            self.logger.critical(f'NEED TO FIX WHAT HAPPENS IF LESS THAN {num_files} POSSIBLE FILES')
            num_files = max_files
        self.logger.debug(f'{num_possible_files=}  {min_files=}  {max_files=}  {num_files=}')

        # sample from all possible paths to get num_files paths
        self.logger.info(f'  Sample cartesian cross product tree ({num_files:,d} out of {num_possible_files:,d} total possible)')
        tree = self._cartesian_cross_product_tree(
            dir_template=dir_template, 
            file_template=file_template,
            path_variables=path_variables,
            path_choices=path_choices,
            rng=rng,
            num_files=num_files)
        if readme_path:
            tree.readme_path = readme_path
            tree.readme_str = readme_str
            tree.add_paths([readme_path])
        self.logger.debug('\n' + "\n".join(tree.get_paths()[:10]))

        fs = FileSystem(
            seed=seed,
            readme_visibility=readme_visibility,
            metadata_visibility=metadata_visibility,
            directory_template=dir_template,
            filename_template=file_template,
            path_choices=path_choices,
            story=story,
            tree=tree,
            path_variables=path_variables,
            file_variables=[])
        
        # determine file variables
        self.logger.info('  Determine file variables')
        if not num_file_columns:
            num_file_columns = int(rng.integers(*self.conf.num_file_columns))
        self._determine_file_variables(num_file_columns=num_file_columns, fs=fs, rng=rng)
        self.logger.debug(f'{fs.file_variables=}')

        return fs


    def _populate_readme(self, story: Story, rng: np.random.Generator) -> Tuple[Dict[str, bool], str, str]:
        visibility = {}
        readme_path = None
        readme_str = None
        if rng.random() <= self.conf.readme_prob:
            readme_path = 'README.md'
            readme_str = f'# {story.root_dir_name}\n'
            if rng.random() <= self.conf.readme_title_prob:
                visibility['title'] = True
                readme_str += f'\n**"{story.title}"**\n'
                if rng.random() <= self.conf.readme_abstract_prob:
                    visibility['abstract'] = True
                    readme_str += f'\n{story.abstract}\n'
        return visibility, readme_path, readme_str
    

    def _determine_metadata_visibility(self, rng: np.random.Generator) -> Dict[str, bool]:
        visibility = {}
        if rng.random() <= self.conf.metadata_tax_prob:
            visibility['taxonomy'] = True
            if rng.random() <= self.conf.metadata_title_prob:
                visibility['title'] = True
                if rng.random() <= self.conf.metadata_abstract_prob:
                    visibility['abstract'] = True
        return visibility


    def _generate_templates_with_llm(
            self, 
            placeholders: Dict[str, dict], 
            story: Story, 
            chosen_ext: str, 
            rng: np.random.Generator) -> Tuple[str, str, Dict[str, str], str]:
        """
        Build directory and file templates by prompting the LLM for one placeholder at a time.
        Uses config for separators and structure hyperparameters.
        Each directory level can have a different number of placeholders and separator.
        """
        sep_choices = {'dir_inner_separators': [], 'ph_per_level': []}

        
        available_placeholders = list(placeholders.keys())
        placeholder_descriptions = {k: v['description'] for k, v in placeholders.items()}

        num_levels = int(rng.integers(*self.conf.dir_levels))
        sep_choices['num_levels'] = num_levels
        
        used_placeholders = []
        dir_template = ""
        var_names = []
        for level in range(num_levels):            
            
            # Sample number of placeholders for this level
            num_per_level = int(rng.integers(*self.conf.per_level_dir))
            sep_choices['ph_per_level'].append(num_per_level)
            
            # Sample separator for this level
            dir_inner_separator = str(rng.choice(self.conf.dir_inner_separators))
            sep_choices['dir_inner_separators'].append(dir_inner_separator)
            
            level_placeholders = []
            for i in range(num_per_level):
                if not available_placeholders:
                    break
                if i != 0:
                    dir_template += dir_inner_separator

                result = self.llm.generate_dict(
                    prompt=FileSystemConfigPrompt.prompt(
                        story=story,
                        placeholders=[(ph, placeholder_descriptions[ph]) for ph in available_placeholders],
                        current_dir_template=f'data/{dir_template}',
                        current_file_template="",
                        template_type="directory",
                        var_names=var_names), 
                    required_keys=["next_placeholder"],
                )
                
                ph_match = re.search(r"\{([^}]+)\}", str(result["next_placeholder"]))
                if ph_match:
                    next_ph = ph_match.group(1)
                else:
                    next_ph = str(result["next_placeholder"]).strip()

                # Strictly enforce only unused placeholders
                if next_ph not in available_placeholders:
                    next_ph = rng.choice(available_placeholders)

                # Remove from available immediately
                if next_ph != 'var':
                    available_placeholders.remove(next_ph)
                else:
                    var_names.append(result['name'].replace(' ', '_'))
                level_placeholders.append(next_ph)
                used_placeholders.append(next_ph)

                dir_template += f"{{{next_ph}}}"
            dir_template += '/'
            
        self.logger.debug(f'{dir_template=}')

        # File template: use remaining placeholders, sample count and separator
        file_separator = str(rng.choice(self.conf.file_separators))
        sep_choices['file_separator'] = file_separator
        file_placeholders = []
        num_file_placeholders = min(
            len(available_placeholders), 
            int(rng.integers(*self.conf.file_placeholders)))
        sep_choices['file_phs'] = num_file_placeholders
        file_template = ""
        for i in range(num_file_placeholders):
            if not available_placeholders:
                break
            if i != 0:
                file_template += file_separator
            
            result = self.llm.generate_dict(
                prompt=FileSystemConfigPrompt.prompt(
                    story=story,
                    placeholders=[(ph, placeholder_descriptions[ph]) for ph in available_placeholders],
                    current_dir_template=f'data/{dir_template}',
                    current_file_template=file_template,
                    template_type="file",
                    var_names=var_names), 
                required_keys=["next_placeholder"],
            )

            ph_match = re.search(r"\{([^}]+)\}", str(result["next_placeholder"]))
            if ph_match:
                next_ph = ph_match.group(1)
            else:
                next_ph = str(result["next_placeholder"]).strip()

            # Strictly enforce only unused placeholders, use experiment variable otherwise
            if next_ph not in available_placeholders:
                next_ph = 'var'

            # Remove from available immediately
            if next_ph != 'var':
                available_placeholders.remove(next_ph)
            else:
                var_names.append(result['name'])
            file_placeholders.append(next_ph)
            used_placeholders.append(next_ph)

            file_template += f"{{{next_ph}}}"
        file_template += chosen_ext


        def enumerate_vars(template, idx: int = 1):
            new_template = ''
            i = 0
            while i < len(template):
                if template[i:i+3] == 'var':
                    new_template += f'var{idx}'
                    idx += 1
                    i += 3
                else:
                    new_template += template[i]
                    i += 1
            return new_template, idx
        
        dir_template, idx = enumerate_vars(dir_template)
        file_template, _ = enumerate_vars(file_template, idx)
        return dir_template, file_template, sep_choices, var_names
    

    def _populate_placeholders(
        self, 
        path_variables: List[PathVariable], 
        story: Story, 
        path_choices: Dict[str, Any],
        rng: np.random.Generator) -> List[PathVariable]:

        variables = []

        for v in path_variables:
            
            if v.short_name in ['seq_number']:
                v.values = list(range(
                    rng.choice([0, 1]), 
                    rng.integers(*self.conf.max_run_number)))

            elif v.short_name == 'researcher':
                result = self.llm.generate_dict(
                    prompt=PlaceholderPrompt.researchers(
                        num_names=rng.integers(*self.conf.num_researchers), 
                        format=path_choices['researcher_format']),
                    required_keys=["names", "dir_names"])
                v.values = clean_values(result["dir_names"])
                v.metadata['names'] = result["names"]

            elif v.short_name == 'date':
                all_valid_dates = get_dates_between(*self.conf.date_range)
                start = rng.choice(all_valid_dates[:-10])
                end = rng.choice(get_dates_between(start, all_valid_dates[-1]))
                all_chosen_dates = get_dates_between(start, end)
                size = min(len(all_chosen_dates), rng.integers(*self.conf.num_dates))
                chosen_dates = sorted(rng.choice(all_chosen_dates, size=size, replace=False))
                v.values = chosen_dates
                v.metadata['min_date'] = min(chosen_dates)
                v.metadata['max_date'] = max(chosen_dates)
            
            elif v.category == 'var':
                d = self.llm.generate_dict(
                    prompt=VariablePrompt.path_variable(
                        story=story,
                        var_name=v.name,
                        preceding_variables=variables, 
                        path_choices=path_choices,
                        var_format=path_choices['var_expansion']),
                    required_keys=['name', 'short_name', 'values', 'description'])
                v.name = d['name']
                v.short_name = d['short_name']
                v.description = d['description']
                v.values = clean_values(d['values'])
                variables.append(v)

            v.reevaluate_type()

        return path_variables
    

    def _cartesian_cross_product_tree(
        self,
        dir_template: str,
        file_template: str,
        path_variables: List[PathVariable],
        num_files: int,
        path_choices: Dict[str, str],
        rng: np.random.Generator) -> DirectoryTree:

        # collect lists of values and sizes
        all_populated_values = [pv.values for pv in path_variables]
        sizes = [len(v) for v in all_populated_values]

        # quick guard
        if any(s == 0 for s in sizes):
            return DirectoryTree.from_strs([])  # or raise depending on desired behavior

        # compute total number of combinations (Python int; may be huge)
        total_combinations = 1
        for s in sizes:
            total_combinations *= s

        if num_files >= total_combinations:
            # want all combos: generate directly (safe only if total not enormous)
            # If total is huge and user requested all, this will still blow memory — but that's expected.
            all_combinations = list(product(*all_populated_values))
            # shuffle and compute
            indices = np.arange(len(all_combinations))
            rng.shuffle(indices)
            chosen_idxs = np.sort(indices[:num_files])
            paths = []
            for idx in chosen_idxs:
                dir_name = dir_template
                file_name = file_template
                for pv, val in zip(path_variables, all_combinations[idx]):
                    if pv.location == PathVariableLocation.DIRNAME:
                        dir_name = pv.populate(dir_name, val, path_choices)
                    elif pv.location == PathVariableLocation.FILENAME:
                        file_name = pv.populate(file_name, val, path_choices)
                paths.append(f'{dir_name}{file_name}')
            return DirectoryTree.from_strs(sorted(paths))

        # Sample k distinct indices from 0..total_combinations-1
        chosen_indices = _sample_k_indices_floyd(total_combinations, num_files, rng)

        # Convert indices into actual value tuples and build paths
        paths = []
        for idx in chosen_indices:
            vals = _index_to_combination(idx, sizes, all_populated_values)
            dir_name = dir_template
            file_name = file_template
            for pv, val in zip(path_variables, vals):
                if pv.location == PathVariableLocation.DIRNAME:
                    dir_name = pv.populate(dir_name, val, path_choices)
                elif pv.location == PathVariableLocation.FILENAME:
                    file_name = pv.populate(file_name, val, path_choices)
            paths.append(f'{dir_name}{file_name}')

        return DirectoryTree.from_strs(sorted(paths))


    def _determine_file_variables(self, num_file_columns: int, fs: FileSystem, rng: np.random.Generator):
        
        # get variable names and descriptions
        dict_variables = self.llm.generate_list_of_dict(
            prompt=VariablePrompt.file_variables(
                num_names=num_file_columns,
                story=fs.story,
                path_variables=fs.path_variables,
                path_choices=fs.path_choices),
            required_keys=['name', 'short_name', 'description', 'role', 'var_type'],
            length=num_file_columns)

        # resolve Identifier and Independent variables
        id_variables = {}
        ind_variables = {}
        dep_variables = {}
        for i, dv in enumerate(dict_variables):
            try:
                var_role = VariableRole(dv['role'].replace('VariableRole.', '').lower())
            except:
                self.logger.warning(f'ignoring ineligible var role "{var_role}" and using INDEPENDENT instead')
                var_role = VariableRole.INDEPENDENT
            try:
                var_type = VariableType(dv['var_type'].lower())
            except:
                self.logger.warning(f'ignoring ineligible var type "{var_type}" and using CATEGORICAL instead')
                var_type = VariableType.CATEGORICAL
            if var_role in [VariableRole.IDENTIFIER, VariableRole.INDEPENDENT, VariableRole.DATETIME]:
                ind_variables[i] = self._create_independent_file_variable(
                    fv=FileVariable(
                        name=dv['name'],
                        short_name=dv['short_name'].lower(),
                        description=dv['description'],
                        role=VariableRole.INDEPENDENT,
                        var_type=(var_type if var_type != VariableType.IDENTIFIER else VariableType.CATEGORICAL),
                        type=default_variable_type_map[var_type]),
                    story=fs.story,
                    path_variables=fs.path_variables,
                    rng=rng)
            elif var_role == VariableRole.DEPENDENT:
                dep_variables[i] = FileVariable(
                    name=dv['name'],
                    short_name=dv['short_name'].lower(),
                    description=dv['description'],
                    role=var_role,
                    var_type=var_type,
                    type=default_variable_type_map[var_type])
            else:
                raise ValueError(var_role)
        self.logger.debug(f'{id_variables=}\n{ind_variables=}\n{dep_variables=}')

        # determine sort key
        fs.sort_key = self.llm.generate_dict(
            prompt=VariablePrompt.sorting_variable(
                story=fs.story,
                ind_file_variables=list(ind_variables.values())),
            required_keys=['sort_key'])['sort_key']
        
        # reconstruct partial variables from separated dictionaries for test df below
        variables = []
        for i in range(len(dict_variables)):
            if i in id_variables:
                variables.append(id_variables[i])
            elif i in ind_variables:
                variables.append(ind_variables[i])
        fs.file_variables = variables

        # resolve Dependent variables
        test_paths = _generate_comprehensive_test_paths(
            dir_template=fs.directory_template,
            file_template=fs.filename_template,
            path_variables=fs.path_variables,
            path_choices=fs.path_choices)
        self.logger.debug(f'test_paths\n' + '\n'.join(test_paths))
        test_path_var_dicts = [self.file_generator.parse_path(path=path, fs=fs) for path in test_paths]
        for i, fv in dep_variables.items():
            self.logger.info(f'    Determine dependent file variable: {fv.name}')
            dep_variables[i] = self._create_dependent_file_variable(
                fv=fv, 
                fs=fs,
                ind_file_variables=list(ind_variables.values()),
                test_path_var_dicts=test_path_var_dicts,
                rng=rng)
            
        # reconstruct variables from separated dictionaries
        variables = []
        for i in range(len(dict_variables)):
            if i in id_variables:
                variables.append(id_variables[i])
            elif i in ind_variables:
                variables.append(ind_variables[i])
            else:
                variables.append(dep_variables[i])
        fs.file_variables = variables

        # sample dependent variable fns to estimate distribution
        sample_paths = rng.choice(
            fs.tree.get_paths(exclude_readme=True), 
            size=min(fs.tree.num_files, self.conf.stoch_est_num_paths))
        samples = {}
        for path in sample_paths:
            file = self.file_generator.populate_file(
                fs=fs, path=str(path), rng=rng, num_lines=self.conf.stoch_est_num_rows_per_path)
            for i, dv in dep_variables.items():
                new_samples = file.data[dv.short_name].dropna().to_list()
                if len(new_samples) == 0:
                    self.logger.warning(f'{dv.short_name} has all NaNs in path: {path}')
                samples[i] = samples.get(i, []) + new_samples
        for i, sample in samples.items():
            if fs.file_variables[i].var_type == VariableType.CATEGORICAL:
                c = Counter(samples[i])
                fs.file_variables[i].values = list(c.keys())
                fs.file_variables[i].probabilities = (np.array(list(c.values())) / len(samples[i])).tolist()
                fs.file_variables[i].reevaluate_type()
            elif fs.file_variables[i].var_type == VariableType.CONTINUOUS:
                fs.file_variables[i].deciles = np.percentile(sample, np.arange(101, step=10)).tolist()
                fs.file_variables[i].std = float(np.std(sample))
            elif fs.file_variables[i].var_type == VariableType.INTEGER:
                fs.file_variables[i].deciles = np.percentile(sample, np.arange(101, step=10)).round().astype(int).tolist()
                fs.file_variables[i].std = float(np.std(sample))
            else:
                raise NotImplementedError(f'DEPENDENT - {fs.file_variables[i].var_type}')

 
    def _create_independent_file_variable(
            self, 
            fv: FileVariable, 
            story: Story,
            path_variables: List[PathVariable],
            rng: np.random.Generator) -> IndependentFileVariable:
        
        prompt = VariablePrompt.independent_variable_distribution(
            fv=fv, story=story, path_variables=path_variables)
        
        if fv.var_type == VariableType.CATEGORICAL:
            out = self.llm.generate_dict(prompt=prompt)
            ifv = IndependentCategoricalVariable.from_file_variable(
                v=fv, values=list(out.keys()), probabilities=list(out.values()))
            ifv.reevaluate_type()

        elif (fv.var_type == VariableType.CONTINUOUS) or (fv.var_type == VariableType.INTEGER):
            out = self.llm.generate_dict(prompt=prompt, required_keys=['distribution', 'param_range'])
            dist_name = out['distribution'].upper()
            params = {
                k: signif(rng.uniform(min(l), max(l)), self.conf.sig_figs) 
                for k, l in out['param_range'].items()}
            
            if fv.var_type == VariableType.CONTINUOUS:
                ifv = IndependentContinuousVariable.from_file_variable(
                    v=fv, distribution=ContinuousDistribution[dist_name], params=params)
            else:
                ifv = IndependentIntegerVariable.from_file_variable(
                    v=fv, distribution=IntegerDistribution[dist_name], params=params)

        elif fv.var_type == VariableType.DATETIME:
            out = self.llm.generate_dict(prompt=prompt, required_keys=['type', 'start_time', 'end_time', 'fmt'])
            
            # Parse datetime strings to datetime objects
            start_time = datetime.fromisoformat(out['start_time'])
            end_time = datetime.fromisoformat(out['end_time'])
            fmt = out['fmt']
            
            if out['type'].lower() == 'interval':
                # Reconstruct timedelta from interval dict
                interval_data = out['interval']
                unit = interval_data['unit'].lower()
                value = interval_data['value']
                
                # Map unit names to timedelta parameter names
                unit_to_param = {
                    'days': 'days',
                    'seconds': 'seconds',
                    'microseconds': 'microseconds',
                    'milliseconds': 'milliseconds',
                    'minutes': 'minutes',
                    'hours': 'hours',
                    'weeks': 'weeks'
                }
                
                interval = timedelta(**{unit_to_param[unit]: value})
                ifv = IndependentIntervalDatetimeVariable.from_file_variable(
                    v=fv, start_time=start_time, end_time=end_time, interval=interval, fmt=fmt)
            else:  # sampled
                ifv = IndependentSampledDatetimeVariable.from_file_variable(
                    v=fv, start_time=start_time, end_time=end_time, fmt=fmt)
                
        else:
            raise ValueError(fv.var_type)
        
        return ifv


    def _create_dependent_file_variable(
            self, 
            fv: FileVariable, 
            fs: FileSystem, 
            ind_file_variables: List[FileVariable],
            test_path_var_dicts: Dict[str, str],
            rng: np.random.Generator) -> DependentFileVariable:

        fn_name = 'f_' + re.sub(r'[^A-Za-z0-9_]', '', fv.short_name)

        for attempt in range(self.conf.max_generation_retries):

            try:
                prompt = VariablePrompt.dependent_variable_function(
                    fv=fv,
                    fn_name=fn_name,
                    story=fs.story,
                    path_variables=fs.path_variables,
                    ind_file_variables=ind_file_variables,
                    path_choices=fs.path_choices,
                    include_assistant_response_prefix=self.conf.include_assistant_response_prefix)
                fn_var_assignment_prefix = prompt[-1]['content']
                fn_str = self.llm.generate_markdown_code(
                    prompt=prompt,
                    continue_final_message=True)
                fn_str = sanitize_math_functions(fn_str)
                depends_on = [
                    v.short_name for v in fs.path_variables + ind_file_variables 
                    if v.short_name in fn_str[len(fn_var_assignment_prefix):]]

                if fv.var_type == VariableType.CATEGORICAL:
                    DepVarClass = DependentCategoricalVariable
                elif fv.var_type == VariableType.CONTINUOUS:
                    DepVarClass = DependentContinuousVariable
                else:
                    DepVarClass = DependentIntegerVariable
                dfv = DepVarClass.from_file_variable(fv=fv, fn_name=fn_name, fn_str=fn_str, depends_on=depends_on)
                
                # compile
                try:
                    dfv.compile_fn(raise_error_on_failure=True)
                except Exception as e:
                    raise IncorrectFormatLLMOutputException(f"Dependent File Variable fn `{dfv.fn_name}` doesn't compile: {e}")
                
                # forward pass
                try:
                    for d in test_path_var_dicts:
                        # Generate fresh independent data for each test path to catch edge cases
                        path_test_df = self.file_generator.get_nondependent_df(
                            fs=fs, 
                            num_lines=self.conf.stoch_est_num_rows_per_path, 
                            rng=rng, 
                            only_independent_cols=True)
                        out = self.file_generator.populate_dependent(
                            fv=dfv, path_var_dict=d, independent_df=path_test_df, rng=rng)
                    return dfv
                except Exception as e:
                    error_trace = ''.join(traceback.format_exception(e)[-2:]).strip()
                    self.logger.info(f"Didn't finish forward pass, producing the following error:\n{error_trace}")

                # debug loop
                for debug_attempt in range(self.conf.max_debug_retries):
                    self.logger.warning(f"Attempt {attempt+1}/{self.conf.max_generation_retries}, Debug Attempt {debug_attempt+1}/{self.conf.max_debug_retries}")
                    new_fn_str = self.llm.generate_markdown_code(
                        prompt=VariablePrompt.debug_dependent_variable(
                            dfv=dfv,
                            stack_trace=error_trace,
                            path_variables=fs.path_variables,
                            ind_file_variables=ind_file_variables,
                            path_choices=fs.path_choices,
                            include_assistant_response_prefix=self.conf.include_assistant_response_prefix),
                        continue_final_message=True)
                    new_fn_str = sanitize_math_functions(new_fn_str)
                    dfv.fn_str = new_fn_str
                    try:
                        dfv.compile_fn(raise_error_on_failure=True)
                    except:
                        raise IncorrectFormatLLMOutputException(f"Dependent File Variable fn `{dfv.fn_name}` doesn't compile")
                    try:
                        for d in test_path_var_dicts:
                            # Generate fresh independent data for each test path to catch edge cases
                            path_test_df = self.file_generator.get_nondependent_df(
                                fs=fs, num_lines=self.conf.stoch_est_num_rows_per_path, rng=rng, only_independent_cols=True)
                            out = self.file_generator.populate_dependent(
                                fv=dfv, path_var_dict=d, independent_df=path_test_df, rng=rng)
                        return dfv
                    except Exception as e:
                        error_trace = ''.join(traceback.format_exception(e)[-2:]).strip()
                        self.logger.info(f"Didn't finish forward pass, producing the following error:\n{error_trace}")
                        

            except IncorrectFormatLLMOutputException as e:
                self.logger.warning(f"Attempt {attempt+1}/{self.conf.max_generation_retries}: {e}")

        raise IncorrectFormatLLMOutputException(f"Failed to generate valid structured output after {self.conf.max_generation_retries} attempts. Failing fn:\n{dfv.format_fn()}")


def _generate_comprehensive_test_paths(
        dir_template: str,
        file_template: str,
        path_variables: List[PathVariable],
        path_choices: Dict[str, str]) -> List[str]:
    """
    Generate minimal test paths that comprehensively cover all distinct values of path variables.
    
    Strategy: Create N paths where N = max(len(pv.values) for all pv).
    For path i, use the i-th value of each variable (or first value if variable has fewer values).
    
    This guarantees:
    - Every value of every path variable appears in at least one test path
    - Minimal number of paths (only as many as the most-valued variable)
    
    Args:
        dir_template: Directory template with placeholders
        file_template: File template with placeholders
        path_variables: List of path variables with populated values
        path_choices: Path formatting choices
        
    Returns:
        List of test paths covering all variable values
    """
    # Find the maximum number of values across all variables
    max_values = max((len(pv.values) for pv in path_variables), default=1)
    
    test_paths = []
    
    # Create one path per index
    for idx in range(max_values):
        dir_name = dir_template
        file_name = file_template
        
        for pv in path_variables:
            # Use the idx-th value if available, otherwise use the first value
            if idx < len(pv.values):
                actual_val = pv.values[idx]
            else:
                actual_val = pv.values[0] if pv.values else ''
            
            if pv.location == PathVariableLocation.DIRNAME:
                dir_name = pv.populate(dir_name, actual_val, path_choices)
            elif pv.location == PathVariableLocation.FILENAME:
                file_name = pv.populate(file_name, actual_val, path_choices)
        
        test_paths.append(f'{dir_name}{file_name}')
    
    return sorted(list(set(test_paths)))  # Remove duplicates and sort



def sanitize_math_functions(fn_str: str) -> str:
    """
    Wraps math function arguments with domain-safe checks.
    For example, math.sqrt(x) becomes math.sqrt(max(0, x)).
    Properly handles nested parentheses to avoid breaking the code.
    
    Args:
        fn_str: Python function code containing math library calls
        
    Returns:
        Modified function code with domain-safe wrappers
    """
    # Map math functions to their safe wrappers
    safe_wrappers = {
        'sqrt': 'max(0, {})',           # sqrt requires x >= 0
        'log': 'max(1e-10, {})',        # log requires x > 0
        'log10': 'max(1e-10, {})',      # log10 requires x > 0
        'log1p': 'max(-1 + 1e-10, {})', # log1p requires x > -1
        'log2': 'max(1e-10, {})',       # log2 requires x > 0
        'asin': 'max(-1, min(1, {}))',  # asin requires -1 <= x <= 1
        'acos': 'max(-1, min(1, {}))',  # acos requires -1 <= x <= 1
        'atanh': 'max(-0.9999, min(0.9999, {}))',  # atanh requires -1 < x < 1
    }
    
    result = []
    i = 0
    while i < len(fn_str):
        # Look for math.function_name(
        if fn_str[i:i+5] == 'math.':
            # Extract function name
            j = i + 5
            while j < len(fn_str) and (fn_str[j].isalnum() or fn_str[j] == '_'):
                j += 1
            
            func_name = fn_str[i+5:j]
            
            # Check if this is one of our unsafe functions
            if func_name in safe_wrappers and j < len(fn_str) and fn_str[j] == '(':
                # We found an unsafe function call, extract its argument
                paren_start = j
                paren_count = 0
                arg_start = j + 1
                arg_end = arg_start
                
                # Find matching closing parenthesis
                for k in range(paren_start, len(fn_str)):
                    if fn_str[k] == '(':
                        paren_count += 1
                    elif fn_str[k] == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            arg_end = k
                            break
                
                # Extract the argument(s)
                arg = fn_str[arg_start:arg_end]
                
                # Build the safe call
                wrapper = safe_wrappers[func_name]
                safe_call = f'math.{func_name}({wrapper.format(arg)})'
                
                result.append(safe_call)
                i = arg_end + 1
            else:
                result.append(fn_str[i])
                i += 1
        else:
            result.append(fn_str[i])
            i += 1
    
    return ''.join(result)


def clean_values(values: list) -> list:
    return [str(x).replace('/', '') for x in values]


def contiguous_range(items: List[int | float], rng: np.random.Generator):
    start = rng.choice(items)
    end = rng.choice([x for x in items if x >= start])
    return sorted([x for x in items if x >= start and x <= end])


def get_dates_between(start_date, end_date):
    """
    Generates a list of all dates between start_date and end_date (inclusive).

    Args:
        start_date (date): The starting date.
        end_date (date): The ending date.

    Returns:
        list: A list of date objects.
    """
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")

    dates_list = []
    current_date = start_date
    while current_date <= end_date:
        dates_list.append(current_date)
        current_date += timedelta(days=1)
    return dates_list


def _sample_k_indices_floyd(N: int, k: int, rng: np.random.Generator):
    """
    Sample k unique integers from range(0, N) using Floyd's algorithm.
    Returns a sorted list of indices.
    """
    if k >= N:
        return list(range(N))
    s = set()
    # iterate j from N-k to N-1 inclusive
    for j in range(N - k, N):
        # rng.integers is [low, high) so pass j+1 for inclusive 0..j
        t = int(rng.integers(0, j + 1))
        if t in s:
            s.add(j)
        else:
            s.add(t)
    return sorted(s)


def _index_to_combination(idx: int, sizes: List[int], all_values: List[List]):
    """
    Convert index in [0, prod(sizes)-1] to a tuple of selected values.
    `sizes` and `all_values` are in the same order as path_variables (first->last).
    """
    # Work from last dimension (fastest changing) back to first
    vals = [None] * len(sizes)
    for i in range(len(sizes) - 1, -1, -1):
        n = sizes[i]
        if n == 0:
            raise ValueError("One of the path variable value lists is empty")
        rem = idx % n
        vals[i] = all_values[i][rem]
        idx //= n
    return vals
