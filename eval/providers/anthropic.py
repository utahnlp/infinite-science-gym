import logging
import os
from typing import Any, Dict

import anthropic

from .config import ANTHROPIC_ARGS
from .provider import BaseProvider
from ..prompts import format_prompt, prepare_messages, get_system_prompt


logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):

    provider: str = 'anthropic'

    def __init__(self, model_name: str, tool_instructions: str, tool_cfg: Dict[str, Any], **kwargs):
        self.model_name = model_name
        self.instructions = get_system_prompt(
            add_role_instructions=True, 
            tool_instructions=tool_instructions)

        self.mcp_servers = [
            {"type": "url", "url": d['url'], "name": name} 
            for name, d in tool_cfg.items()]
        self.tools = [
            {'type': 'mcp_toolset', 'mcp_server_name': name} 
            for name in tool_cfg.keys()]
        
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic()

        logger.info(f"Initialized {self.provider} provider with model: {model_name}")
        
        
    def run_prompt(self, fs_seed: int, question: str) -> Dict:
        prompt = format_prompt(seed=fs_seed, prompt=question)
        messages = prepare_messages(user_msg=prompt)
        with self.client.beta.messages.stream(
            messages=messages,
            model=self.model_name,
            system=self.instructions,
            mcp_servers=self.mcp_servers,
            tools=self.tools,
            **ANTHROPIC_ARGS
        )  as stream:
            response = stream.get_final_message()
        return response.to_dict()
