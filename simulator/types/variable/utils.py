from datetime import datetime, timedelta
from typing import Any, Dict

from . import (
    ContinuousDistribution,
    DependentFileVariable, 
    DependentCategoricalVariable,
    DependentContinuousVariable,
    DependentIntegerVariable,
    FileVariable, 
    IdentifierFileVariable, 
    IndependentCategoricalVariable, 
    IndependentContinuousVariable, 
    IndependentDatetimeVariable,
    IndependentIntervalDatetimeVariable,
    IndependentIntegerVariable,
    IndependentSampledDatetimeVariable,
    IntegerDistribution,
    VariableRole, 
    VariableType)


def file_variable_to_json(fv: FileVariable) -> Dict[str, Any]:
    if fv.role == VariableRole.IDENTIFIER:
        if fv.var_type == VariableType.IDENTIFIER:
            subclass_data = {'shuffled': fv.shuffled}
    elif fv.role == VariableRole.INDEPENDENT:
        if fv.var_type == VariableType.CATEGORICAL:
            subclass_data = {'values': fv.values, 'probabilities': fv.probabilities}
        elif fv.var_type == VariableType.CONTINUOUS and not hasattr(fv, 'start_time'):
            # IndependentContinuousVariable
            subclass_data = {'distribution': fv.distribution.name, 'params': fv.params}
        elif fv.var_type == VariableType.INTEGER:
            subclass_data = {'distribution': fv.distribution.name, 'params': fv.params}
        elif fv.var_type == VariableType.DATETIME and hasattr(fv, 'interval'):
            # IndependentIntervalDatetimeVariable
            subclass_data = {
                'start_time': fv.start_time.isoformat(),
                'end_time': fv.end_time.isoformat(),
                'interval': fv.interval.total_seconds(),
                'fmt': fv.fmt}
        elif fv.var_type == VariableType.DATETIME and not hasattr(fv, 'interval'):
            # IndependentSampledDatetimeVariable
            subclass_data = {
                'start_time': fv.start_time.isoformat(),
                'end_time': fv.end_time.isoformat(),
                'fmt': fv.fmt}
    elif fv.role == VariableRole.DEPENDENT:
        subclass_data = {
            'fn_name': fv.fn_name, 
            'fn_str': fv.fn_str, 
            'depends_on': fv.depends_on}
        if fv.var_type == VariableType.CATEGORICAL:
            subclass_data = subclass_data | {'values': fv.values, 'probabilities': fv.probabilities}
        elif fv.var_type == VariableType.CONTINUOUS:
            subclass_data = subclass_data | {'deciles': fv.deciles, 'std': fv.std}
        elif fv.var_type == VariableType.INTEGER:
            subclass_data = subclass_data | {'deciles': fv.deciles, 'std': fv.std}

    return {
            'name': fv.name,
            'short_name': fv.short_name,
            'description': fv.description,
            'role': fv.role.name,
            'var_type': fv.var_type.name,
            'type': fv.type,
            'subclass_data': subclass_data}
        

def file_variable_from_json(obj: Dict[str, Any], compile_fns: bool = False) -> FileVariable:
    role = VariableRole[obj['role']]
    var_type = VariableType[obj['var_type']]

    shared_data = {
        'name': obj['name'],
        'short_name': obj['short_name'],
        'description': obj['description'],
        'role': VariableRole[obj['role']],
        'var_type': var_type,
        'type': obj['type']}

    sc_data = obj['subclass_data']

    if role == VariableRole.IDENTIFIER:
        if var_type == VariableType.IDENTIFIER:
            fv = IdentifierFileVariable(
                shuffled=sc_data['shuffled'], 
                **shared_data)
    elif role == VariableRole.INDEPENDENT:
        if var_type == VariableType.CATEGORICAL:
            fv = IndependentCategoricalVariable(
                values=sc_data['values'], 
                probabilities=sc_data['probabilities'], 
                **shared_data)
        elif var_type == VariableType.CONTINUOUS:
            # IndependentContinuousVariable
            fv = IndependentContinuousVariable(
                distribution=ContinuousDistribution[sc_data['distribution']], 
                params=sc_data['params'], 
                **shared_data)
        elif var_type == VariableType.DATETIME and 'interval' in sc_data:
            # IndependentIntervalDatetimeVariable
            fv = IndependentIntervalDatetimeVariable(
                start_time=datetime.fromisoformat(sc_data['start_time']),
                end_time=datetime.fromisoformat(sc_data['end_time']),
                interval=timedelta(seconds=sc_data['interval']),
                fmt=sc_data['fmt'],
                **shared_data)
        elif var_type == VariableType.DATETIME:
            # IndependentSampledDatetimeVariable
            fv = IndependentSampledDatetimeVariable(
                start_time=datetime.fromisoformat(sc_data['start_time']),
                end_time=datetime.fromisoformat(sc_data['end_time']),
                fmt=sc_data['fmt'],
                **shared_data)
        elif var_type == VariableType.INTEGER:
            fv = IndependentIntegerVariable(
                distribution=IntegerDistribution[sc_data['distribution']], 
                params=sc_data['params'], 
                **shared_data)
    elif role == VariableRole.DEPENDENT:
        if var_type == VariableType.CATEGORICAL:
            fv = DependentCategoricalVariable(
                fn_name=sc_data['fn_name'], 
                fn_str=sc_data['fn_str'], 
                depends_on=sc_data['depends_on'],
                values=sc_data['values'], 
                probabilities=sc_data['probabilities'],
                **shared_data)
        elif var_type == VariableType.CONTINUOUS:
            fv = DependentContinuousVariable(
                fn_name=sc_data['fn_name'], 
                fn_str=sc_data['fn_str'], 
                depends_on=sc_data['depends_on'],
                std=sc_data['std'],
                deciles=sc_data['deciles'],
                **shared_data)
        elif var_type == VariableType.INTEGER:
            fv = DependentContinuousVariable(
                fn_name=sc_data['fn_name'], 
                fn_str=sc_data['fn_str'], 
                depends_on=sc_data['depends_on'],
                std=sc_data['std'],
                deciles=sc_data['deciles'],
                **shared_data)
        
        if compile_fns:
            fv.compile_fn()

    return fv
