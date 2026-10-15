import math


def est_tokens(text: str) -> int:
    """Conservative token estimate (Session 07A used words * 1.3; we also floor at chars / 4)."""
    if not text:
        return 0
    return math.ceil(max(len(text.split()) * 1.3, len(text) / 4.0))
