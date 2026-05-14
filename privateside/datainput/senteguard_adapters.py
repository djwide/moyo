"""Adapters to integrate with sente when available."""

try:
    from sente import index_utils
except ImportError:  # pragma: no cover
    index_utils = None


def get_index_utils():
    """Return sente index utilities if installed."""
    return index_utils
