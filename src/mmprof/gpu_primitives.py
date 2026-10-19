"""General timing helper retained from earlier local profiling code."""
import torch
from typing import Any, Callable


def timed_encoder(encoder_fn: Callable[[], Any]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        torch.cuda.synchronize()
        start.record()
        _ = encoder_fn()
        end.record()
        torch.cuda.synchronize()
    return float(start.elapsed_time(end))
