from typing import Dict, List

ROLE_INSTRUCTIONS = (
    "You are a scientific research assistant. "
    "Answer questions using the data in the available scientific data repository."
)

def get_system_prompt(add_role_instructions: bool = True, tool_instructions: str = None) -> str:
    p = ''
    if add_role_instructions:
        p = ROLE_INSTRUCTIONS
    if tool_instructions:
        p = f'{p}\n\n{tool_instructions}' if p else tool_instructions
    return p


def prepare_messages(user_msg: str, system_msg: str = None, assistant_response_prefix: str = None) -> List[Dict[str, str]]:
    messages = []
    if system_msg:
        messages.append({'role': 'system', 'content': system_msg})
    messages.append({'role': 'user', 'content': user_msg})
    if assistant_response_prefix:
        messages.append({'role': 'assistant', 'content': assistant_response_prefix})
    return messages


def format_prompt(seed: int, prompt: str) -> str:
    formatted_prompt = (
        f'This question is about filesystem #{seed}. '
        f'Any calls to the scientific data repository should specify `id={seed}`.\n\n'
        f'{prompt.strip()}\n\n'
        'When you have your answer ready, return a JSON object with a single key, "answer", mapping to your answer. '
        'If the question doesn\'t have an answer, reply "not possible" as your answer, like this: `{"answer": "not possible"}`.'
    )

    return formatted_prompt


def get_tool_instructions(simulator_cfg: dict, interpreter_cfg: dict) -> str:
    instructions = (
        'Below are the available tools and their descriptions.\n\n'
        '# Tools\n\n'
        'There are two sets of tools for you to use.\n\n'
    )

    for i, cfg in enumerate([simulator_cfg, interpreter_cfg]):
        instructions += (
            f'## {i+1}. {cfg["encoded_name"]}\n\n'
            f'{cfg["instructions"]}\n'
            + '\n'.join([f'### `{n}`\n\n{d}' for n, d in cfg["tool_descriptions"].items()]) +
            '\n'
        )

    return instructions.strip()
