from dataclasses import dataclass
from typing import Any, Dict, List

from .utils import default_field


SAMPLING_PARAMETERS = {
    '__default__': {},
    'qwen': {'temperature':0.7, 'top_p':0.8, 'top_k':20},
    'google': {'top_k': 128},
    'gpt-oss': {'top_k': 128}
}

@dataclass
class QAConfig:
    sig_figs: List[int] = default_field([2, 5]) # how many significant figures to truncate values to
    dist_sample_quants: List[float] = default_field([0.2, 0.8]) # will uniformily sample filter values from distribution's [low, high] quantile range
    max_path_conditions: int = 3
    max_file_conditions: int = 3

    # paraphrase
    num_paraphrases: int = 3
    paraphrase_model_name: str = 'Qwen/Qwen3-4B-Instruct-2507'
    paraphrase_device_map: str = 'auto'

    def __post_init__(self):
        assert len(self.sig_figs) == 2
        assert all(x >= 1 for x in self.sig_figs)
        assert self.sig_figs[0] < self.sig_figs[1]
        assert len(self.dist_sample_quants) == 2
        assert all(x > 0 and x < 1 for x in self.dist_sample_quants)
        assert self.dist_sample_quants[0] < self.dist_sample_quants[1]

    def get_sampling_parameters(self) -> Dict[str, Any]:
        for k, sampling_parameters in SAMPLING_PARAMETERS.items():
            if k in self.paraphrase_model_name.lower():
                return sampling_parameters
        return SAMPLING_PARAMETERS['__default__']
