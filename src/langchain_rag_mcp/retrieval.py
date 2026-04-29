import re
from typing import Iterable


STOP_WORDS = {
    "the",
    "and",
    "with",
    "how",
    "for",
    "from",
    "langchain",
    "langgraph",
    "langsmith",
}

QUERY_STOP_WORDS = STOP_WORDS | {
    "what",
    "where",
    "which",
    "when",
    "why",
    "who",
    "does",
    "use",
    "used",
    "using",
    "add",
    "get",
    "set",
    "into",
    "later",
    "around",
    "about",
    "explain",
    "explains",
    "document",
    "documented",
    "docs",
}

PAYLOAD_RANK_FIELDS = ("title", "breadcrumb", "source", "content_type", "symbols", "content")


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text.lower())
        if token not in STOP_WORDS
    }


def meaningful_query_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_-]{2,}", text.lower())
        if token not in QUERY_STOP_WORDS
    }


def _payload_text(results: Iterable) -> str:
    parts = []
    for result in results:
        payload = result.payload or {}
        parts.extend(str(payload.get(key, "")) for key in PAYLOAD_RANK_FIELDS)
    return " ".join(parts)


def query_term_coverage(query: str, results: Iterable) -> float:
    query_terms = meaningful_query_terms(query)
    if not query_terms:
        return 1.0

    result_terms = {
        token
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_-]{2,}", _payload_text(results).lower())
    }
    return len(query_terms & result_terms) / len(query_terms)


def _langchain_middleware_bonus(query_terms: set[str], source: str) -> float:
    if not {"agent", "middleware"} <= query_terms:
        return 0.0
    if not ({"model", "request"} & query_terms):
        return 0.0
    if "/oss/python/langchain/middleware/" not in source.lower():
        return 0.0
    return 0.08


def rerank(query: str, results: Iterable, top_k: int) -> list:
    query_tokens = tokens(query)
    query_terms = meaningful_query_terms(query)
    ranked = []
    for result in results:
        payload = result.payload or {}
        haystack = " ".join(str(payload.get(key, "")) for key in PAYLOAD_RANK_FIELDS)
        overlap = len(query_tokens & tokens(haystack))
        exact_bonus = 0.08 if query.lower() in haystack.lower() else 0.0
        source_bonus = _langchain_middleware_bonus(query_terms, str(payload.get("source", "")))
        ranked.append((result.score + (0.025 * overlap) + exact_bonus + source_bonus, result))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [result for _, result in ranked[:top_k]]


def format_results(results: Iterable, chunk_cap: int) -> str:
    parts = []
    for result in results:
        payload = result.payload
        chunk = f"## {payload['title']}"
        if payload.get("source"):
            chunk += f"\nSource: {payload['source']}"
        chunk += f"\n\n{payload['content'][:chunk_cap]}"
        parts.append(chunk)

    return "\n\n---\n\n".join(parts)
