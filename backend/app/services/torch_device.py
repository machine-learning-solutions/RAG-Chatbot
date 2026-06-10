from functools import lru_cache

import torch

from app.config import get_settings


@lru_cache
def get_inference_device() -> str:
    preferred = get_settings().torch_device.strip().lower()
    if preferred == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
