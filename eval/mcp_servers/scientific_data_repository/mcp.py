import os
import pathlib
import sys

from fastmcp import FastMCP

project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eval.mcp_servers.scientific_data_repository.simulator import simulator, SuccessStatus
from eval.mcp_servers.scientific_data_repository.mime import EXTENSION_TO_MIME_TYPE
from eval.mcp_servers.scientific_data_repository.model import (
    ListDirectoryResponse,
    ReadTextFileResponse,
    ReadBinaryFileResponse)
from eval.mcp_servers.scientific_data_repository.utils import load_config


cfg = load_config(os.environ["CONFIG_PATH"], os.environ["DEFAULT_CONFIG_PATH"])['mcp']


mcp = FastMCP(
    name=cfg['name'],
    instructions=cfg['instructions'],
    version=cfg['version'])

@mcp.tool(
    'list_directory',
    description=cfg['tool_descriptions']['list_directory'],
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
def list_directory(id: int, prefix: str = '/*', depth: int = 1) -> ListDirectoryResponse:
    fs, status = simulator.get_filesystem(seed=id)
    
    paths = None
    if fs is not None:
        paths = fs.tree.get_paths(prefix=prefix, depth=depth)

    return ListDirectoryResponse(status=status, paths=paths)

@mcp.tool(
    'read_text_file',
    description=cfg['tool_descriptions']['read_text_file'],
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
def read_text_file(id: int, path: str, head: int | None = None, tail: int | None = None) -> ReadTextFileResponse:    
    if head and tail:
        return ReadTextFileResponse(status=SuccessStatus.HEAD_AND_TAIL.value, file_content=None)

    file, status = simulator.read_file(seed=id, path=path)
    
    content = None
    if file is not None:
        content = file.convert_to_extension()
        if head:
            content = '\n'.join(content.split('\n')[:head])
        elif tail:
            content = '\n'.join(content.split('\n')[-tail:])
    
    return ReadTextFileResponse(status=status, file_content=content)

@mcp.tool(
    'read_binary_file',
    description=cfg['tool_descriptions']['read_binary_file'],
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
def read_binary_file(id: int, path: str) -> ReadBinaryFileResponse:    
    file, status = simulator.read_file(seed=id, path=path)
    
    content = None
    mime_type = None
    if file is not None:
        content = file.convert_to_extension_base64()
        mime_type = EXTENSION_TO_MIME_TYPE[file.extension]
    
    return ReadBinaryFileResponse(status=status, base64_file_content=content, mime_type=mime_type)
