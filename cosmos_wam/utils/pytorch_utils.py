"""PyTorch utilities."""
import torch
from typing import Dict, Any


def dict_apply(x: Dict[str, Any], func) -> Dict[str, Any]:
    """Apply a function to all tensor values in a dict."""
    result = {}
    for key, value in x.items():
        if isinstance(value, torch.Tensor):
            result[key] = func(value)
        elif isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = value
    return result
