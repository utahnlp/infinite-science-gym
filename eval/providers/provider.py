from typing import Any, Dict, List

class BaseProvider:

    provider: str = 'base'

    def __init__(self, model_name: str, tool_instructions: str, tool_cfg: Dict[str, Any], device_map: str):
        pass

    def run_prompt(self, fs_seed: int, question: str) -> Dict:
        pass

    def shutdown(self):
        pass
