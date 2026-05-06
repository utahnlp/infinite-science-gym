"""
Main evaluation orchestration module.
Handles loading provider-specific evaluation modules and executing evaluations.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

import yaml

from .prompts import get_tool_instructions
from .providers import (
    AnthropicProvider, 
    GoogleProvider, 
    LocalProvider, 
    OpenaiProvider, 
    BaseProvider,
)


logger = logging.getLogger(__name__)


# Mapping of provider names to module names
PROVIDER_CLASSES = {
    'anthropic': AnthropicProvider,
    'google': GoogleProvider,
    'local': LocalProvider,
    'openai': OpenaiProvider,
}


class EvaluationHarness:
    """Extensible harness for running LLM evaluations across multiple providers."""
    
    def __init__(self, provider: str, model_name: str, device_map: str):

        self.provider = provider
        self.model_name = model_name
        
        if provider not in PROVIDER_CLASSES:
            raise ValueError(f"Unsupported provider: {provider}. Must be one of {list(PROVIDER_CLASSES.keys())}")
                
        logger.info(f"Loading tool instructions")
        self.tool_instructions = load_tool_instructions()

        # Initialize the provider
        logger.info(f"Initializing {provider} provider with model: {model_name}")
        ProviderClass: BaseProvider = PROVIDER_CLASSES[provider]
        self.provider_obj = ProviderClass(
            model_name=model_name, 
            tool_instructions=self.tool_instructions, 
            tool_cfg=load_tool_cfg(), 
            device_map=device_map)

    
    def evaluate(self, fs_seed: int, question: str, fail_on_error: bool = False) -> Dict[str, Any]:    
        start_time = datetime.now()
        try:
            response = self.provider_obj.run_prompt(fs_seed=fs_seed, question=question)
            duration = datetime.now() - start_time
            
            result = {
                'status': 'success',
                'response': response,
                'duration_seconds': duration.total_seconds(),
                'timestamp': start_time.isoformat(),
                'model': self.model_name,
                'provider': self.provider
            }
            return result
        except Exception as e:
            if fail_on_error:
                raise e
            duration = datetime.now() - start_time
            logger.error(f"Error during evaluation: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'duration_seconds': duration.total_seconds(),
                'timestamp': start_time.isoformat(),
                'model': self.model_name,
                'provider': self.provider
            }

def load_tool_cfg(path: str = 'eval/tools.yaml') -> Dict[str, Any]:

    with open(path) as f:
        tool_cfg = yaml.safe_load(f)
    return tool_cfg

def load_tool_instructions(
        api_path: str = 'api/cfg/default.yaml', 
        interpreter_path: str = 'python_interpreter/cfg/default.yaml') -> str:
    
    with open(api_path) as f:
        simulator_cfg = yaml.safe_load(f)['mcp']
    with open(interpreter_path) as f:
        interpreter_cfg = yaml.safe_load(f)['mcp']
    return get_tool_instructions(simulator_cfg, interpreter_cfg)
