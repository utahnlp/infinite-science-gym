from typing import Any, Dict, List, Tuple

from .format import (
    format_prompt_for_llm,
    path_variables_to_md_list,
    file_variables_to_md_list,
    variables_to_python_def_and_prefix
)
from ..types import (
    ContinuousDistribution,
    DependentFileVariable,
    FileVariable, 
    IndependentFileVariable,
    IntegerDistribution,
    PathVariable,
    Story, 
    VariableRole,
    VariableType
)


class VariablePrompt:
    system_msg = "You are a helpful assistant for aiding in science experiments."

    @staticmethod
    def path_variable(
            story: Story, 
            var_name: str,
            preceding_variables: List[PathVariable], 
            path_choices: Dict[str, str],
            var_format: str) -> List[Dict[str, Any]]:
        
        preceding_prompt = ''
        if preceding_variables:
            used_vars_str = path_variables_to_md_list(path_variables=preceding_variables, path_choices=path_choices)
            preceding_prompt = f"For context, the variables (and associated values) below have already been used. Don't repeat variables by the experiment.\n\n{used_vars_str}\n"

        user_msg = (
            "Given the following scientific project context:\n"
            f"project title: {story.title}\n"
            f"project description: {story.description}\n\n" + '-' * 80 + '\n\n'
            f'Consider an independent variable in this project called "{var_name}".\n\n'
            "Given the project information and the variable name, generate a plausible context for this variable. "
            "Think about how the variable relates to the project's hypotheses. "
            "Create two variants of the variable name: the full descriptive name in natural language, and a short name that's appropriate for including in a directory path. "
            "It might even be abbreviated if appropriate for the scientific field. "
            "Also generate a list of the values that variable could take on for this project, again in a format appropriate for including in a directory name. "
            f'When coming up with short names and values, keep in mind that variable will be used in a directory name with the following format: "{var_format}". '
            "Please keep the short name and values short, definitely less than 15 characters each, and ideally less than 8 characters if possible. "
            "**Stick to using ASCII characters and characters that are common in directory and file paths. Do not use uncommon UTF-8 characters that a person wouldn't use when naming a directory or file!** "
            "Finally, generate a brief, one sentence description of what the variable represents. "
            "Make sure to specify the variable's units if relevant.\n\n"
            + preceding_prompt +
            "Return the output as a JSON object with keys: name, short_name, values, description.\n"
            'The "name" and "short_name" keys should map to a string, the "values" key should map to list of string values, and the "description" key should map to a string description. '
            'Think out loud before responding.'
        )

        return format_prompt_for_llm(user_msg, VariablePrompt.system_msg)
    
    @staticmethod
    def file_variables(
            num_names: int, 
            story: Story, 
            path_variables: List[PathVariable], 
            path_choices: Dict[str, str]) -> List[Dict[str, Any]]:

        used_vars_str = path_variables_to_md_list(path_variables=path_variables, path_choices=path_choices)

        user_msg = (
            "Given the following scientific project context:\n"
            f"project title: {story.title}\n"
            f"project description: {story.description}\n\n" + '-' * 80 + '\n\n'
            "Using the description above, your task is to decide what experimental results (independent, and dependent variables) might go in a results file. "
            f"Please generate {num_names} plausible variables that haven't been used in the path name.\n\n"
            "For context, these variables (and associated values) have already been used: \n\n"
            f"{used_vars_str}\n"
            'Please follow these guidelines:\n'
            "1. Make sure you include at least a few independent variables and a few dependent variables.\n"
            "2. Don't repeat any variables listed above that have already been used.\n"
            '3. If your variable is of type float or integer, but it only takes discrete values from a known set of values, its type should be "categorical".\n'
            "4. Your chosen variables should, together, be enough to provide support for or against the project's hypotheses.\n"
            '5. If your variable role is "identifier", then your variable type must also be "identifier". '
            'Conversely, if your variable role is not "identifier", then your variable type must not be "identifier".\n'
            '6. If your variable role is "datetime", then your variable type must be "datetime". '
            'Conversely, if your variable role is not "datetime", then your variable type must not be "datetime".\n\n'
            f"Generate the {num_names} variable names as a list containing {num_names} dictionaries, one for each variable. "
            "Each dictionary should contain the following keys/values:\n"
            f"* `idx` (int): A counter that increments from 1 to help you make sure you have exactly {num_names} items.\n"
            "* `name` (str): The name of the variable in plain English.\n"
            "* `short_name` (str): The name of the variable how you might see it as a column name in a results file.\n"
            "* `description` (str): A brief one sentence description of the variable. It should specify the unit of the variable if relevant.\n"
            f'* `role` (str): The variable\'s role in the experiment, one of `{[x.value for x in VariableRole]}`.\n'
            f'* `var_type` (str): The variable\'s type, one of `{[x.value for x in VariableType]}`. Note: `datetime` variables represent temporal data and may be appropriate if the experiment involves time-series measurements or temporal dynamics.\n\n'
            'Return your result as a JSON list of dictionaries, like this:\n\n```\n[\n  {\n    "idx": 1,\n    ...\n  },\n  {\n    "idx": 2, \n    ...\n  },\n  ...\n]\n```\n\n'
            'Think out loud for a bit before responding with your decision.'
        )

        return format_prompt_for_llm(user_msg, VariablePrompt.system_msg)

    @staticmethod
    def independent_variable_distribution(fv: FileVariable, story: Story, path_variables: List[PathVariable]) -> List[Dict[str, Any]]:
        
        if fv.var_type == VariableType.IDENTIFIER:
          raise NotImplementedError  
        elif fv.var_type == VariableType.CATEGORICAL:
            dist_prompt = (
                "Since the variable is categorical, please generate all possible values that the variable can take, as well as the corresponding probabilities. "
                "Return a JSON object containing a dictionary mapping each possible value (str) to its probability (float). "
                "The probabilities should sum to 1. "
                'Here is an example of what the output should look like:\n\n```\n{\n  "value1": 0.25,\n  "value2": 0.15,\n    ...\n}\n```\n\n'
                "Think out loud for a bit before responding with your decision."
            )
        elif fv.var_type == VariableType.CONTINUOUS:
            dist_prompt = (
                "Since the variable is continuous, please generate a plausible distribution that might describe it. "
                "The possible distributions are:\n"
                "* `beta` parameterized by `alpha` and `beta`.\n"
                "* `exponential` parameterized by `beta` (equivalent to 1/lambda).\n"
                "* `normal` parameterized by `mean` and `std`.\n"
                "* `uniform` parameterized by `low` and `high`.\n\n"
                'Return a JSON object containing a dictionary with the key "distribution" mapping to the string of the chosen distribution above, and a key called "param_range". '
                'The "param_range" key maps to a dictionary, where each entry maps the chosen distribution\'s parameters to a list of length two containing lower- and upper-bounds of a plausible range for that parameter. '
                "For example, if you were to pick an `abcd` distribution with parameters `e` and `f`, you might return the object `{'distribution': 'abcd', 'param_range': {'e': [1, 2], 'f': [0.1, 0.2]}}` if the ranges for `e` and `f` are plausible. "
                "Think out loud for a bit before responding with your decision."
            )
        elif fv.var_type == VariableType.INTEGER:
            dist_prompt = (
                "Since the variable is integer, please generate a plausible distribution that might describe it. "
                "The possible distributions are:\n"
                "* `bernoulli` parameterized by `p` (probability of 1).\n"
                "* `binomial` parameterized by `n` (number of trials), and `p` (success probability of each trial).\n"
                "* `geometric` parameterized by `p` (success probability of each trial).\n"
                "* `negative_binomial` parameterized by `p` (success probability of each trial) and `r` (number of successes).\n"
                "* `poisson` parameterized by `lambda` (events in interval).\n\n"
                'Return a JSON object containing a dictionary with the key "distribution" mapping to the string of the chosen distribution above, and a key called "param_range". '
                'The "param_range" key maps to a dictionary, where each entry maps the chosen distribution\'s parameters to a list of length two containing lower- and upper-bounds of a plausible range for that parameter. '
                "For example, if you were to pick an `abcd` distribution with parameters `e` and `f`, you might return the object `{'distribution': 'abcd', 'param_range': {'e': [1, 2], 'f': [0.1, 0.2]}}` if the ranges for `e` and `f` are plausible. "
                "Think out loud for a bit before responding with your decision."
            )
        elif fv.var_type == VariableType.DATETIME:
            # Build path variable info
            datetime_path_var = next((pv for pv in path_variables if pv.category == 'date'), None)
            path_datetime_info = ""
            if datetime_path_var:
                min_date_str = datetime_path_var.metadata['min_date'].strftime("%Y-%m-%d")
                max_date_str = datetime_path_var.metadata['max_date'].strftime("%Y-%m-%d")
                path_datetime_info = (
                    f"\n\n**NOTE:** A date variable is already used in the directory path:\n"
                    f"  - `{datetime_path_var.short_name}`: {datetime_path_var.description} (from {min_date_str} to {max_date_str})\n\n"
                    f"It's recommended to coordinate your datetime variable's start and end times with this date range. "
                    f"For example, you might choose start and end times that align with or encompass the date range from {min_date_str} to {max_date_str}."
                )
            
            dist_prompt = (
                "Since the variable is datetime, please choose how to represent the temporal data. "
                "There are two options:\n"
                "1. **Interval Datetime**: Select this if the datetime values occur at regular, known intervals (e.g., daily measurements, hourly samples). "
                "You'll need to specify the start time, end time, and the interval between measurements.\n"
                "2. **Sampled Datetime**: Select this if the datetime values are sampled uniformly at random between a start and end time, with no specific regular pattern.\n\n"
                'Return a JSON object with the following structure:\n'
                '- If interval: `{"type": "interval", "start_time": "YYYY-MM-DD HH:MM:SS", "end_time": "YYYY-MM-DD HH:MM:SS", "interval": {"value": <number>, "unit": "<unit>"}, "fmt": "strftime_format"}` '
                'where "interval" is a dictionary with "value" (a positive number) and "unit" (one of: days, seconds, microseconds, milliseconds, minutes, hours, weeks). '
                'For example, "interval": {"value": 1, "unit": "days"} or "interval": {"value": 30, "unit": "minutes"}.\n'
                '"fmt" is a Python strftime format string for how the datetime should be rendered.\n'
                '- If sampled: `{"type": "sampled", "start_time": "YYYY-MM-DD HH:MM:SS", "end_time": "YYYY-MM-DD HH:MM:SS", "fmt": "strftime_format"}` '
                'where "fmt" is a Python strftime format string.\n\n'
                'Example strftime formats: "%Y-%m-%d" (2026-02-18), "%Y/%m/%d %H:%M" (2026/02/18 14:30), "%d-%b-%Y" (18-Feb-2026).\n'
                'Think out loud for a bit before responding with your decision.'
                + path_datetime_info
            )
        
        user_msg = (
            "Given the following scientific project context:\n"
            f"project title: {story.title}\n"
            f"project description: {story.description}\n\n" + '-' * 80 + '\n\n'
            "Your task is to determine a plausible distribution a variable might take. "
            f'The variable in question is called "{fv.name}". It\'s a {fv.var_type.value}, {fv.role.value} variable, and it has the following description: {fv.description}\n\n'
            + dist_prompt
        )

        return format_prompt_for_llm(user_msg, VariablePrompt.system_msg)

    @staticmethod
    def dependent_variable_function(
            fv: FileVariable, 
            fn_name: str,
            story: Story, 
            path_variables: List[PathVariable], 
            ind_file_variables: List[IndependentFileVariable], 
            path_choices: Dict[str, str],
            include_assistant_response_prefix: bool) -> List[Dict[str, Any]]:
        
        existing_var_prompt = path_variables_to_md_list(path_variables=path_variables, path_choices=path_choices)
        existing_var_prompt += file_variables_to_md_list(file_variables=ind_file_variables)
        fn_def, fn_var_assignment_prefix, assistant_response_prefix = variables_to_python_def_and_prefix(
            fn_name=fn_name,
            _type=fv.type,
            path_variables=path_variables,
            file_variables=ind_file_variables,
            include_assistant_response_prefix=include_assistant_response_prefix)

        user_msg = (
            "Given the following scientific project context:\n"
            f"project title: {story.title}\n"
            f"project description: {story.description}\n\n" + '-' * 80 + '\n\n'
            "Your task is to decide how a new dependent variable depends on the experiment's independent variables. "
            f'The variable in question is called "{fv.name}". '
            f"It's a {fv.var_type.value} variable, and it has the following description: {fv.description}\n\n"
            "Here is a list of the existing independent variables you can use:\n\n"
            f'{existing_var_prompt}\n' 
            f'To determine the function that calculates "{fv.name}" from the existing variables, please write the relationship as a python function with the following signature:\n\n'
            f"```python\n{fn_def}\n```\n\n"
            "Please follow these guidelines:\n\n"
            "1. The first argument to the function, `ind_variables` is a dictionary mapping containing the independent variables above. "
            "It maps each variable's short name (str) to its value (see list above for each variable's corresponding type).\n"
            "2. The second argument to the function, `error`, is a float variable that we'll use to capture natural noise in the relationship. "
            "Please make sure to add the `error` term to your formula where appropriate. "
            "If you perform some clipping to ensure the function output fits within some range (e.g. `x = min(max(x, lower_bound), upper_bound))`, make sure to add the `error` term before so the clipped result stays within the intended range. "
            "It should look something like: `x = min(max(x + error, lower_bound), upper_bound))`. "
            "You're not obligated to clip your final value's minimum, maximum, or minimum and maximum value if that's not appropriate for the variable's range."
            f"3. Since this variable is a {fv.var_type.name.lower()} variable, the type of the value returned by `{fn_name}()` should be {fv.type}.\n"
            "4. If you believe there may be a non-linear relationship, you can use any functions from the python `math` package if needed, but you're not obligated to use them. "
            "If you use a function from the `math` library, ensure that the input to the function is in the function's domain. "
            "If you need to use any common python packages (e.g. `math`, `datetime`, etc.) please import them **inside of the function body.** "
            "You can use `numpy` if you need it, but **DO NOT use the `numpy.random` module.** "
            "All randomness should come from the `error` function parameter.\n"
            "5. You can set any hard-coded coefficients and variables you need.\n"
            "6. You should use at least one of the available variables, but you don't need to use all of them if you don't think there's a relationship to the dependent variable.\n"
            + ("7. Respond by starting the function, and leave any thinking you need to do as python comments inside the function.\n\n" if include_assistant_response_prefix else '\n') +
            "**The primary goal is to come up with a function that realistically characterizes the dependent variable's distribution.**\n\n"
            "Please return your solution as a markdown python code block containing the function code."
        )
        
        return format_prompt_for_llm(user_msg, VariablePrompt.system_msg, assistant_response_prefix)

    @staticmethod
    def debug_dependent_variable(
            dfv: DependentFileVariable, 
            stack_trace: str,
            path_variables: List[PathVariable], 
            ind_file_variables: List[IndependentFileVariable], 
            path_choices: Dict[str, str],
            include_assistant_response_prefix: bool) -> List[Dict[str, Any]]:
        
        existing_var_prompt = path_variables_to_md_list(path_variables=path_variables, path_choices=path_choices)
        existing_var_prompt += file_variables_to_md_list(file_variables=ind_file_variables)        
        _, _, assistant_response_prefix = variables_to_python_def_and_prefix(
            fn_name=dfv.fn_name,
            _type=dfv.type,
            path_variables=path_variables,
            file_variables=ind_file_variables,
            include_assistant_response_prefix=include_assistant_response_prefix)

        user_msg = (
            "Please help me fix a python function that's throwing an error when run.\n\n"
            f'For context, the function is called `{dfv.fn_name}()`, and it represents a plausible hypothesis for the distribution of an experimental variable called "{dfv.name}" with the following description: {dfv.description}. '
            "Here are the contents of the function, with line numbers on the left:\n\n"
            f"```python\n{dfv.format_fn()}\n```\n\n"
            "When run, the function produces the following error:\n\n"
            f"```\n{stack_trace}\n```\n\n"
            "Please reproduce the function, but fix the error in a straightforward way so the function works as intended. "
            "Keep these guidelines in mind:\n\n"
            "1. The first argument to the function, `ind_variables` is a dictionary mapping containing the variables described above above. "
            "It maps each variable's short name (str) to its value (see list above for each variable's corresponding type).\n"
            "2. The second argument to the function, `error`, is a float variable that will capture natural noise in the relationship. "
            "Please make sure to add the `error` term to your formula where appropriate. "
            "If you perform some clipping to ensure the function output fits within some range (e.g. `x = min(max(x, lower_bound), upper_bound))`, make sure to add the `error` term before so the clipped result stays within the intended range. "
            "It should look something like: `x = min(max(x + error, lower_bound), upper_bound))`. "
            "You're not obligated to clip your final value's minimum, maximum, or minimum and maximum value if that's not appropriate for the variable's range."
            f"3. Since this variable is a {dfv.var_type.name.lower()} variable, the type of the value returned by `{dfv.fn_name}()` should be {dfv.type}.\n"
            "4. If you believe there may be a non-linear relationship, you can use any functions from the python `math` package if needed, but you're not obligated to use them. "
            "If you use a function from the `math` library, ensure that the input to the function is in the function's domain. "
            "If you need to use any common python packages (e.g. `math`, `numpy`, etc.) please import them **inside of the function body.**\n\n"
            "5. You can set any hard-coded coefficients and variables you need.\n"
            "6. You should use at least one of the available variables, but you don't need to use all of them if you don't think there's a relationship to the dependent variable.\n"
            + ("7. Respond by starting the function, and leave any thinking you need to do as python comments inside the function.\n\n" if include_assistant_response_prefix else '\n') +
            "**The primary goal is to fix the broken function, while ensuring the resulting code realistically characterizes the dependent variable's distribution.**\n\n"
            "Please return your solution as a markdown python code block containing the fixed function code."
        )

        return format_prompt_for_llm(user_msg, VariablePrompt.system_msg, assistant_response_prefix)
    

    @staticmethod
    def sorting_variable(story: Story, ind_file_variables: List[IndependentFileVariable]) -> List[Dict[str, Any]]:

        existing_var_prompt = file_variables_to_md_list(file_variables=ind_file_variables)
        user_msg = (
            "Given the following scientific project context:\n"
            f"project title: {story.title}\n"
            f"project description: {story.description}\n\n" + '-' * 80 + '\n\n'
            "Please help me decide which variable makes the most sense for sorting the files by. "
            "For context, these are the columns in each file:\n\n"
            f"{existing_var_prompt}\n"
            'The goal is to replicate how scientific data in this domain might typically look. '
            "If it's the most realistic to not sort the data and leave it in random order, respond with `null`. "
            'Provide your answer as a JSON object with a single key: "sort_key" mapping to the short name of the variable you think would be best to sort by (or `null` if none make sense).'
        )

        return format_prompt_for_llm(user_msg, VariablePrompt.system_msg)
