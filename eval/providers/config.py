MAX_OUTPUT_TOKENS = 10000
MAX_THINKING_TOKENS = 8000

TEMPERATURE = 1.0
EFFORT = 'medium'
SEED = 0

ANTHROPIC_ARGS = {
    'temperature': TEMPERATURE,
    'max_tokens': MAX_OUTPUT_TOKENS,
    'output_config':{'effort': EFFORT},
    'tool_choice': {'type': 'auto', 'disable_parallel_tool_use': True},
    'betas': ["mcp-client-2025-11-20"],
}

GOOGLE_ARGS = {
    'temperature': TEMPERATURE,
    'maxOutputTokens': MAX_OUTPUT_TOKENS,
    'seed': SEED,
}

OPENAI_ARGS = {
    'temperature': TEMPERATURE,
    'max_output_tokens': MAX_OUTPUT_TOKENS,
    'reasoning': {'effort': EFFORT},
    'parallel_tool_calls': False
}
