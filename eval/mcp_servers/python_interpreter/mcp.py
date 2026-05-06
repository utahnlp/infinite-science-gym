import os
import pathlib

from fastmcp import FastMCP

from .interpreter import run_code
from .model import GetVersionMetadataResponse, RunPythonCodeResponse
from .utils import load_config, load_python_packages


# Load config and packages
config_path = os.getenv('PYTHON_INTERPRETER_CONFIG_PATH')
cfg = load_config(config_path) if config_path else load_config()
python_packages = load_python_packages()

mcp = FastMCP(
    name=cfg['mcp']['name'],
    instructions=cfg['mcp']['instructions'],
    version=cfg['mcp']['version'])

@mcp.tool(
    'get_version_metadata',
    description=cfg['mcp']['tool_descriptions']['get_version_metadata'],
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
def version_metadata() -> GetVersionMetadataResponse:
    return GetVersionMetadataResponse(
        status='success', 
        python_version=cfg['sandbox']['python_version'],
        python_packages=python_packages)

@mcp.tool(
    'run_python_code',
    description=cfg['mcp']['tool_descriptions']['run_python_code'],
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True})
def run_python_code(code: str) -> RunPythonCodeResponse:
    return run_code(code)
