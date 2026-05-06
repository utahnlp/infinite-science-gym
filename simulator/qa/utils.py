import copy
from dataclasses import field
import math
import sys
from typing import Any

import numpy as np

def signif(x, digits):
    if x == 0 or not math.isfinite(x):
        return x
    # Handle subnormal/very small floats that are below the minimum normal float
    if abs(x) < sys.float_info.min:
        return 0.0
    digits -= math.ceil(math.log10(abs(x)))
    return round(x, digits)

def default_field(obj):
    return field(default_factory=lambda: copy.deepcopy(obj))

def is_none_or_nan(x: Any):
    if x is None:
        return True
    # Check for NaN (works for float and numpy types)
    try:
        if np.isnan(x):
            return True
    except (TypeError, ValueError):
        # np.isnan raises TypeError for non-numeric types
        pass
    return False
