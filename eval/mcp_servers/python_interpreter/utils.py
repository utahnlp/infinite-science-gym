from typing import Any, Dict, List
import os
import pathlib
import yaml


def load_config(path: str = None) -> Dict[str, Any]:
    """Load config from YAML file.
    
    If path is not provided, looks for cfg/default.yaml relative to this file.
    """
    if path is None:
        # Construct path relative to this file's location
        current_dir = pathlib.Path(__file__).parent
        path = current_dir / 'cfg' / 'default.yaml'
    else:
        path = pathlib.Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def load_python_packages(requirements_path: str = None) -> List[str]:
    """Load list of packages available in the sandbox.
    
    If path is not provided, looks for sandbox_packages.txt relative to this file.
    """
    if requirements_path is None:
        # Construct path relative to this file's location
        current_dir = pathlib.Path(__file__).parent
        requirements_path = current_dir / 'sandbox_packages.txt'
    else:
        requirements_path = pathlib.Path(requirements_path)
    
    if not requirements_path.exists():
        raise FileNotFoundError(f"Sandbox packages file not found: {requirements_path}")
    
    with open(requirements_path) as f:
        packages = [x.strip() for x in f.readlines() if x.strip() and not x.strip().startswith('#')]
    return packages
