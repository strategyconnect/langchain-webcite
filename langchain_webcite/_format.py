"""Turn Webcite API responses into compact text a model can read.

The full response is always returned as the tool artifact, so nothing is dropped; these
helpers only choose what goes into the model's context.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def _citations(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    groups = response.get("claim_groups") or []
    if groups and groups[0].get("citations") is not None:
        return list(groups[0]["citations"])
    return list(response.get("citations") or [])


def citation_summary(citation: Mapping[str, Any]) -> dict[str, Any]:
    """The fields of one citation that identify the source and the bound passage."""
    passage = citation.get("passage") or {}
    summary = {
        "id": citation.get("id"),
        "url": citation.get("url"),
        "title": citation.get("title"),
        "stance": citation.get("stance"),
        "passage": passage.get("text") or citation.get("snippet"),
    }
    return {key: value for key, value in summary.items() if value is not None}


def verification_text(response: Mapping[str, Any], max_citations: int) -> str:
    groups = response.get("claim_groups") or []
    verdict: Mapping[str, Any] = (groups[0].get("verdict") if groups else None) or {}
    body = {
        "claim": verdict.get("claim") or (groups[0].get("claim") if groups else None),
        "verdict": verdict.get("result"),
        "summary": verdict.get("summary"),
        "citations": [citation_summary(c) for c in _citations(response)[:max_citations]],
        "credits": (response.get("usage") or {}).get("credits"),
    }
    return json.dumps({k: v for k, v in body.items() if v is not None}, ensure_ascii=False)


def search_results(response: Mapping[str, Any], limit: int | None) -> list[dict[str, Any]]:
    found = [citation_summary(c) for c in _citations(response)]
    return found[:limit] if limit else found


def search_text(response: Mapping[str, Any], limit: int | None) -> str:
    return json.dumps(search_results(response, limit), ensure_ascii=False)


def extraction_text(response: Mapping[str, Any], max_chars: int | None) -> str:
    markdown = str(response.get("markdown") or "")
    truncated = max_chars is not None and len(markdown) > max_chars
    header: dict[str, Any] = {
        "state": response.get("state"),
        "pages": (response.get("usage") or {}).get("pages"),
        "credits": (response.get("usage") or {}).get("credits"),
    }
    if response.get("reason"):
        header["reason"] = response.get("reason")
    if truncated:
        header["truncated_to_chars"] = max_chars
    text = markdown[:max_chars] if truncated else markdown
    return json.dumps({k: v for k, v in header.items() if v is not None}) + "\n\n" + text


def compact_list(values: Sequence[str] | None) -> list[str] | None:
    return list(values) if values else None
