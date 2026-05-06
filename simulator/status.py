from enum import Enum


class SuccessStatus(Enum):
    OK = 'success'
    NOT_CACHED = "This repository id doesn't have a corresponding repository"
    INVALID_PATH = "This path doesn't exist as a file in this repository."
    HEAD_AND_TAIL = "You can specify either head or tail, but not both."
    ERROR = 'error'
