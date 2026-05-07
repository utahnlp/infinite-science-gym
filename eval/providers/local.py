import logging
import os
from typing import Any, Dict

from smolagents import (
    MCPClient, 
    PythonInterpreterTool,
    ToolCallingAgent, 
    TransformersModel, 
    VLLMModel, 
)

from .provider import BaseProvider
from ..prompts import format_prompt


logger = logging.getLogger(__name__)


def list_directory(id: int, prefix: str = '/*', depth: int = 1) -> dict:
    import requests
    api_url = "https://localhost:8000/mcp/mcp"
    url = f'{api_url}/api/directory/{id}'
    params = {'prefix': prefix, 'depth': depth}
    response = requests.get(url=url, params=params)
    return response.json()

def read_text_file(id: int, path: str, head: int | None = None, tail: int | None = None) -> dict:
    import requests
    api_url = "https://localhost:8000/mcp/mcp"
    url = f'{api_url}/api/text_file/{id}'
    params = {'path': path, 'head': head, 'tail': tail}
    return requests.get(url=url, params=params).json()

def read_binary_file(id: int, path: str) -> dict:
    import requests
    api_url = "https://localhost:8000/mcp/mcp"
    url = f'{api_url}/api/binary_file/{id}'
    params = {'path': path}
    return requests.get(url=url, params=params).json()

additional_functions = {
    'list_directory': list_directory,
    'read_text_file': read_text_file,
    'read_binary_file': read_binary_file
}

authorized_imports = ["json", "numpy", "pandas", "sklearn", "scipy", "matplotlib", "seaborn", "sympy", "requests"]

instructions = """
The three functions for accessing data, `list_directory`, `read_text_file`, and `read_binary_file`, are also available from the `python_interpreter` tool as python functions. 
These python variants return the exact same data in the exact same format as the tools you can call directly. 
They may be useful if you want to operate directly on the data using code rather than copying files' contents manually. 
Here are the function definitions:
- `def list_directory(id: int, prefix: str = '/*', depth: int = 1) -> dict:`
- `def read_text_file(id: int, path: str, head: int | None = None, tail: int | None = None) -> dict:`
- `def read_binary_file(id: int, path: str) -> dict:`

Again, you can use these anywhere in the code you submit to the `python_interpreter` tool if you believe they'd help you answer the question.

**Your tool calls must be the JSON format (containing the tool name and the arguments as key-value pairs) specified above! Do not use XML or some other format to invoke a tool!**
"""

class LocalProvider(BaseProvider):

    provider: str = 'local'
    
    def __init__(self, model_name: str, tool_instructions: str, tool_cfg: Dict[str, Any], **kwargs):
        
        self.model_name = model_name
        self.mcp_server_params = [{'url': tool_cfg['scientific_data_repo']['url'], "transport": "streamable-http"}]

        self.mcp_client = MCPClient(self.mcp_server_params, structured_output=True)
        self.tools = self.mcp_client.get_tools()

        if os.environ.get("USE_LOCAL_HUGGINGFACE") == 'yes':
            logger.info('Use TransformersModel')
            self.model = TransformersModel(
                model_id=model_name, 
                device_map='auto', 
                trust_remote_code=True,
                model_kwargs={'cache_dir': os.environ['HF_HOME']})
        else:
            logger.info('Use VLLMModel')
            self.model = VLLMModel(model_id=model_name)

        pit = PythonInterpreterTool(
            authorized_imports=authorized_imports,
            timeout_seconds=300,
        )
        pit.base_python_tools = {**pit.base_python_tools.copy(), **additional_functions}


        self.agent = ToolCallingAgent(
            tools=self.tools + [pit],
            model=self.model, 
            max_tool_threads=False,
            add_base_tools=True,
            instructions=instructions,
            verbosity_level=-1
        )

        logger.info(f"Initialized local provider with model: {model_name}")

    
    def run_prompt(self, fs_seed: int, question: str) -> Dict:
        prompt = format_prompt(seed=fs_seed, prompt=question)         
        response = self.agent.run(prompt, reset=True, return_full_result=True)
        d = response.dict()
        return d

    def shutdown(self):
        self.mcp_client.disconnect()
