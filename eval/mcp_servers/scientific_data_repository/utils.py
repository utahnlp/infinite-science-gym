from typing import Any, Dict
import yaml


def merge_configs(cfg: Dict[str, Any], default_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge cfg over default_cfg.
    
    Returns a dictionary with all values from default_cfg, 
    overwritten by values from cfg where they exist.
    Handles nested dictionaries recursively.
    """
    result = default_cfg.copy()
    for key, value in cfg.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(value, result[key])
        else:
            result[key] = value
    return result


def load_config(path: str, default_path: str = './cfg/default.yaml'):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    with open(default_path, 'r') as f:
        default_cfg = yaml.safe_load(f)
    return merge_configs(cfg, default_cfg)
