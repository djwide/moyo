"""GCS folder names for report orders.

Bucket objects live under ``reports/<prompt-words>_<order-suffix>/`` so the
prefix is readable in Cloud Storage instead of a generic ``ord_xxx`` id.
Firestore still keys documents by ``orderId``.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")


def slugify_first_words(
    text: str,
    *,
    words: int = 3,
    max_len: int = 48,
) -> str:
    """First couple of prompt words as a filesystem-safe slug."""
    tokens = _WORD_RE.findall((text or "").lower())
    n = max(1, int(words))
    slug = "_".join(tokens[:n]) if tokens else "report"
    slug = slug[:max_len].strip("_")
    return slug or "report"


def order_id_suffix(order_id: str, *, length: int = 8) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", (order_id or "").lower())
    if not compact:
        return "order"
    return compact[-max(1, int(length)) :]


def order_storage_folder(
    order_id: str,
    prompts: list[str] | None = None,
    *,
    words: int = 3,
) -> str:
    """GCS folder name: first prompt words plus a short unique order suffix."""
    prompt = ""
    for raw in prompts or []:
        prompt = str(raw).strip()
        if prompt:
            break
    label = slugify_first_words(prompt, words=words)
    suffix = order_id_suffix(order_id)
    return f"{label}_{suffix}"
