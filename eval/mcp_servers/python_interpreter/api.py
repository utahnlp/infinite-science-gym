import importlib.metadata
import os
import pathlib
import sys

from fastapi import FastAPI, Body
import yaml

from .interpreter import run_code
from .mcp import mcp
from .model import StatusResponse, GetVersionMetadataResponse, RunPythonCodeResponse
from .utils import load_config, load_python_packages


# Load config and packages
config_path = os.getenv('PYTHON_INTERPRETER_CONFIG_PATH')
cfg = load_config(config_path) if config_path else load_config()
python_packages = load_python_packages()

mcp_app = mcp.http_app()
app = FastAPI(
    title=cfg['api']['name'],
    description=cfg['api']['description'],
    version=cfg['api']['version'],
    lifespan=mcp_app.router.lifespan_context)
app.mount("/mcp", mcp_app)


@app.get("/")
async def root() -> StatusResponse:
    return StatusResponse(service=cfg['api']['name'], version=cfg['api']['version'])

@app.get("/api/version_metadata")
async def version_metadata() -> GetVersionMetadataResponse:
    return GetVersionMetadataResponse(
        status='success', 
        python_version=cfg['sandbox']['python_version'],
        python_packages=python_packages)
    
@app.post("/api/run_python_code")
def run_python_code(code: str = Body(..., embed=False)) -> RunPythonCodeResponse:
    return run_code(code)
