"""
Work around spaCy/Stanza token mismatches in VALUE (multivalue) on long or messy text.

Upstream create_coref_cluster() asserts token alignment; on failure we disable
coref for that span so morphosyntax + surface rules still run.
"""

from __future__ import annotations

import re
from typing import Any


_PATCHED = False


def apply_multivalue_compat_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    import multivalue.BaseDialect as bd

    original = bd.BaseDialect.create_coref_cluster

    def create_coref_cluster_safe(self, string: str) -> Any:
        try:
            return original(self, string)
        except (AssertionError, IndexError, ValueError):
            return None

    bd.BaseDialect.create_coref_cluster = create_coref_cluster_safe  # type: ignore[method-assign]
    _PATCHED = True


def transform_text(
    dialect: Any,
    text: str,
    *,
    split_paragraphs: bool = True,
) -> str:
    """Run dialect.transform; optionally split on blank lines to reduce NLP drift."""
    text = text.strip("\n")
    if not split_paragraphs or "\n\n" not in text:
        return dialect.transform(text)

    parts = re.split(r"\n\s*\n", text)
    transformed: list[str] = []
    for part in parts:
        if not part.strip():
            transformed.append(part)
            continue
        transformed.append(dialect.transform(part))
    return "\n\n".join(transformed)
