import logging
import os
from typing import Any, Dict
import warnings

import openai
from openai import OpenAI

from .config import OPENAI_ARGS
from .provider import BaseProvider
from ..prompts import format_prompt, prepare_messages, get_system_prompt


logger = logging.getLogger(__name__)


class OpenaiProvider(BaseProvider):

    provider: str = 'openai'
    
    def __init__(self, model_name: str, tool_instructions: str, tool_cfg: Dict[str, Any], **kwargs):
        
        self.model_name = model_name
        self.instructions = get_system_prompt(
            add_role_instructions=True, 
            tool_instructions=tool_instructions)

        self.tools = [
            {
                "type": "mcp",
                "server_label": name,
                "server_description": d['description'],
                "server_url": d['url'],
                "require_approval": "never"
            } for name, d in tool_cfg.items()]
        
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI()

        warnings.filterwarnings("ignore", category=UserWarning, message="Pydantic serializer warnings:")
        logger.info(f"Initialized {self.provider} provider with model: {model_name}")


    def run_prompt(self, fs_seed: int, question: str) -> Dict:
        prompt = format_prompt(seed=fs_seed, prompt=question)
        messages = prepare_messages(user_msg=prompt)
        response = self.client.responses.create(
            input=messages,
            instructions=self.instructions,
            model=self.model_name,
            tools=self.tools,
            **OPENAI_ARGS)
        return response.model_dump()
