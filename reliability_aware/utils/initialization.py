"""Parameter-initialization seeds independent of training randomness."""
from __future__ import annotations

import secrets


def resolve_initialization_seed(value):
    """None keeps legacy initialization; 'random' draws a recordable seed."""
    if value is None:
        return None
    if value == "random":
        return secrets.randbits(32)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ValueError("initialization_seed must be an integer in [0, 2**32), 'random', or null")
    return value
