def dedupe(items: list[str]) -> list[str]:
    """Remove duplicates from gathered items."""
    return list(dict.fromkeys(items))
