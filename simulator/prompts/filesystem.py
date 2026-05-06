import re
from typing import Any, Dict, List, Tuple

from .format import format_prompt_for_llm
from ..types import Story


class FileSystemConfigPrompt:
    system_msg = "You are a helpful assistant for aiding in science experiments."

    @staticmethod
    def prompt(
        story: Story,
        placeholders: List[Tuple[str, str]],
        current_dir_template: str = "",
        current_file_template: str = "",
        template_type: str = "directory",
        var_names: List[str] = []) -> List[Dict[str, Any]]:

        placeholders_str = "\n".join([f"* `{{{name}}}`: {desc}" for name, desc in placeholders])
        var_names_str = ''
        if var_names:
            var_names_str = 'For context, these are the variables chosen for the previous {var} placeholders above:\n\n'
            var_names_str += '\n'.join([f'* {x}' for x in var_names]) + '\n\n'
        user_msg = (
            f'Consider a scientific research project in the domain "{story.domain}" and subdomain "{story.subdomain}", with the title "{story.title}".\n'
            f"The project's description is: {story.description}\n\n" + '-' * 80 + '\n\n'
            "Given the following available placeholder variables (with descriptions):\n\n"
            f"{placeholders_str}\n\n"
            f'...the current directory template structure: "{current_dir_template}"\n'
            + (f'...and the current file template structure: "{current_file_template}"\n\n' if template_type == 'file' else '\n') + 
            f"...your job is to decide which placeholder would make the most sense to come next in the {template_type} template. "
            "Please follow these guidelines:\n\n"
            "1. Make the template's structure specific to how research might be done in that scientific domain. "
            "If the project seems like it could be collaborative, consider adding the {researcher} placeholder. "
            "If you think the project might be time dependent, adding a {date} placeholder might be a good choice. "
            "If the project can be broken down into multiple experiments, consider using the {experiment_name} placeholder. "
            "Be creative!\n"
            "2. Only choose from the available placeholders above.\n"
            "3. Do not repeat any non-var placeholder already used in the directory template when building the file template.\n"
            "4. If you think a researcher in that domain would use a project-specific independent variable next in the template, the {var} placeholder may be applicable. "
            "It can be reused many times, each time representing a different independent variable. "
            "It can also be used as an experiment name, a benchmark name, or any other project-specific experimental condition. "
            "If you want to choose a variable next, specify {var} as your placeholder, and also provide the variable's `name`, mapping to a descriptive, natural language name of the variable. "
            'Note that a variable\'s `name` should not include the possible values it takes, but rather a descriptor of the variable itself (i.e. "color" instead of "red" or "blue").\n'
            f"5. If the project has a time-dependent component that requires a date or time, consider whether a researcher might put that in the {template_type} template or as a column in the file itself. "
            "Only dates (i.e. no times) are allowed in the directory/file templates. "
            "More granular measurements, like hours, minutes, seconds, etc. should go in files. "
            "Using date in a directory/file template and then using time in a file column is allowed.\n"
            "6. Stick to using ASCII characters and characters that are common in directory and file paths.\n\n"
            + var_names_str + 
            "Return the output as a JSON object with the key: `next_placeholder`, and the optional `name` key if applicable. "
            "Think out loud for a bit before responding with your decision."
        )
        return format_prompt_for_llm(user_msg, FileSystemConfigPrompt.system_msg)


class PlaceholderPrompt:
    system_msg = "You are a helpful assistant for aiding in science experiments."

    @staticmethod
    def researchers(num_names: int, format: str) -> List[Dict[str, Any]]:
        if format == 'username':
            format_prompt = "The directory names should be usernames like you might see in a shared computing environment.\n"
        else:
            format_prompt = f"The directory names should be in the following format: {format}. Note that they're always lower-case.\n"

        user_msg = (
            f"Please generate {num_names} names for hypothetical people.\n"
            "Please also generate variants of the names that could be used as directory names in a file system.\n"
            + format_prompt +
            "Return the output as a JSON object with keys: names, dir_names.\n"
            f"Both keys should map to lists of strings. Both lists should be of length: {num_names}"
        )

        return format_prompt_for_llm(user_msg, PlaceholderPrompt.system_msg)
    
    @staticmethod
    def benchmark_names(num_names: int, story: Story) -> List[Dict[str, Any]]:
        user_msg = (
            "Given the following experiment context:\n"
            f"project title: {story.title}\n"
            f"project description: {story.description}\n\n" + '-' * 80 + '\n\n'
            f"Generate {num_names} plausible evaluation benchmark names for experiments that could be a part of the project. "
            "These can be real benchmarks if there are known benchmarks that qualify, or they can be made up. "
            "Please also generate the corresponding directory names for each benchmark. "
            "Please keep the directory names, definitely less than 20 characters each, and ideally less than 15 characters if possible. "
            'Do not use the word "benchmark" in the directory names. '
            "Return the output as a JSON object with keys: names, dir_names.\n"
            f"Both keys should map to lists of strings. Both lists should be of length: {num_names}"
        )

        return format_prompt_for_llm(user_msg, PlaceholderPrompt.system_msg)
