import json
import math
import os
import pathlib
import random

import numpy as np
import torch


def set_global_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_taxonomy(taxonomy_dir: str) -> dict:
    """
    Load the taxonomy from a JSON file.
    Returns a dictionary representing the taxonomy.
    """
    taxonomy = {}
    for path in pathlib.Path(taxonomy_dir).glob("*.json"):
        field = path.stem
        with open(path, 'r') as f:
            taxonomy[field] = json.load(f)
    return taxonomy

def signif(x, digits):
    if x == 0 or not math.isfinite(x):
        return x
    digits -= math.ceil(math.log10(abs(x)))
    return round(x, digits)
