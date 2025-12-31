"""Text processing utility functions."""
import re


def is_up(text: str) -> bool:
    """Check if text contains 'up', 'upper', or 'above'."""
    text = text.lower()
    return bool(re.search(r'(?<!\w)(up|upper|above)(?!\w)(-|\b)?', text))


def is_down(text: str) -> bool:
    """Check if text contains 'down', 'lower', or 'below'."""
    text = text.lower()
    return bool(re.search(r'(?<!\w)(down|lower|below)(?!\w)(-|\b)?', text))


def is_left(text: str) -> bool:
    """Check if text contains 'left'."""
    text = text.lower()
    return bool(re.search(r'(?<!\w)(left)(?!\w)(-|\b)?', text))


def is_right(text: str) -> bool:
    """Check if text contains 'right'."""
    text = text.lower()
    return bool(re.search(r'(?<!\w)(right)(?!\w)(-|\b)?', text))
