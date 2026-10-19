"""General timing helper retained from earlier local profiling code."""
import inspect
from typing import Any, Callable


def call_with_supported_kwargs(fn: Callable[..., Any], values: dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    kwargs = {}
    accepts_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    for name in sig.parameters:
        if name in values:
            kwargs[name] = values[name]
    if accepts_var_kwargs:
        kwargs.update({k: v for k, v in values.items() if k not in kwargs})
    return fn(**kwargs)
