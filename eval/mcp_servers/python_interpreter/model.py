from typing import List

from pydantic import BaseModel


class StatusResponse(BaseModel):
    service: str
    version: str
    status: str = "running"

class GetVersionMetadataResponse(BaseModel):
    status: str
    python_version: str
    python_packages: List[str]

class RunPythonCodeResponse(BaseModel):
    status: str
    output: str
    error: str | None
