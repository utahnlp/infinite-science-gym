import builtins
import keyword
import re
from string import ascii_lowercase
from typing import Dict, List, Tuple

from ..types import (
    FileVariable, 
    PathVariable,
    VariableType
)


def format_prompt_for_llm(user_msg: str, system_msg: str = None, assistant_response_prefix: str = None) -> List[Dict[str, str]]:
    messages = []
    if system_msg:
        messages.append({'role': 'system', 'content': system_msg})
    messages.append({'role': 'user', 'content': user_msg})
    if assistant_response_prefix:
        messages.append({'role': 'assistant', 'content': assistant_response_prefix})
    return messages


def path_variables_to_md_list(path_variables: List[PathVariable], path_choices: Dict[str, str]) -> str:
    md_list = ''
    for pv in path_variables:
        val_str = pv.vals_to_str(path_choices)
        md_list += f'* full name: "{pv.name}", short name: "{pv.short_name}", type: {pv.type}, values: {val_str}, description: {pv.description}\n'
    return md_list


def file_variables_to_md_list(file_variables: List[FileVariable]) -> str:
    md_list = ''
    for fv in file_variables:
        if fv.var_type == VariableType.CATEGORICAL:
            val_str = '[' + ', '.join([str(x) for x in fv.values]) + ']'
            md_list += f'* full name: "{fv.name}", short name: "{fv.short_name}", type: {fv.type}, values: {val_str}, description: {fv.description}\n'
        elif fv.var_type in [VariableType.CONTINUOUS, VariableType.INTEGER]:
            param_str = ', '.join([f'{k}={v}' for k, v in fv.params.items()])
            dist_str = f'{fv.distribution.name.lower()}({param_str})'
            md_list += f'* full name: "{fv.name}", short name: "{fv.short_name}", type: {fv.type}, dist: {dist_str}, description: {fv.description}\n'
    return md_list    


def variables_to_python_def_and_prefix(
        fn_name: str, 
        _type: str, 
        path_variables: List[PathVariable], 
        file_variables: List[FileVariable], 
        include_assistant_response_prefix: bool,
        libraries: List[str] = ['datetime', 'math']) -> Tuple[str, str, str]:

    fn_var_assignment_prefix = ''
    i = 0
    for v in (path_variables + file_variables):
        var_name = v.short_name
        var_name = re.sub(r'[^a-zA-Z0-9_]', '', var_name)
        if keyword.iskeyword(var_name) or (var_name in dir(builtins)):
            var_name = f'_{var_name}'
        if var_name == '':
            var_name = ascii_lowercase[i]
            i += 1
        fn_var_assignment_prefix += f'    {var_name} = {v.type}(ind_variables["{v.short_name}"])\n'
    
    fn_def = f'def {fn_name}(ind_variables: dict, error: float) -> {_type}:'
    import_str = '\n'.join([f'    import {x}' for x in sorted(libraries)])
    assistant_response_prefix = f'```python\n{fn_def}\n\n{import_str}\n\n{fn_var_assignment_prefix}\n    #'

    if include_assistant_response_prefix:
        return fn_def, fn_var_assignment_prefix, assistant_response_prefix
    return fn_def, None, None

