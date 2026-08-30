"""Strict JSON serialization helpers for evaluation reports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite numbers with JSON ``null``."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def strict_json_dumps(value: Any, **kwargs: Any) -> str:
    """Serialize JSON without the non-standard NaN and Infinity tokens."""
    return json.dumps(_json_safe(value), allow_nan=False, **kwargs)
