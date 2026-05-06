from typing import List

from pydantic import BaseModel


class StatusResponse(BaseModel):
    service: str
    version: str
    status: str = "running"

class Metadata(BaseModel):
    field: str | None = None
    domain: str | None = None
    subdomain: str | None = None
    project_title: str | None = None
    project_abstract: str | None = None

class GetMetadataResponse(BaseModel):
    status: str
    metadata: Metadata | None

class ListDirectoryResponse(BaseModel):
    status: str
    paths: List[str] | None

class ReadTextFileResponse(BaseModel):
    status: str
    file_content: str | None

class ReadBinaryFileResponse(BaseModel):
    status: str
    base64_file_content: str | None
    mime_type: str | None
