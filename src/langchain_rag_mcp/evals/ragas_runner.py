from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class RagasEvalSample:
    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "user_input": self.user_input,
            "response": self.response,
            "retrieved_contexts": self.retrieved_contexts,
            "reference": self.reference,
        }


def build_contexts(results: Iterable, *, chunk_cap: int) -> list[str]:
    contexts = []
    for result in results:
        payload = result.payload or {}
        title = str(payload.get("title", "")).strip()
        source = str(payload.get("source", "")).strip()
        content = str(payload.get("content", ""))[:chunk_cap].strip()
        parts = []
        if title:
            parts.append(f"# {title}")
        if source:
            parts.append(f"Source: {source}")
        if content:
            parts.append(content)
        contexts.append("\n".join(parts))
    return contexts


def _reference(case) -> str:
    # RAGAS ContextRecall expects a natural-language ground truth answer, not
    # metadata. It classifies each sentence of `reference` as attributable or
    # not to the retrieved contexts. Metadata-style strings confuse the judge
    # and produce invalid JSON. We synthesise a short prose reference instead.
    if getattr(case, "negative", False):
        return "This query is out of domain. No relevant documentation should be returned."

    parts = []
    terms = getattr(case, "required_terms", [])
    sources = getattr(case, "expected_sources", [])

    if terms:
        parts.append(f"The retrieved context should contain information about {', '.join(terms)}.")
    if sources:
        doc_names = [
            s.rstrip("/").split("/")[-1].replace(".md", "").replace("-", " ")
            for s in sources
        ]
        parts.append(f"The relevant documentation covers: {', '.join(doc_names)}.")

    return " ".join(parts) if parts else "General documentation query."


def _response_from_contexts(contexts: list[str]) -> str:
    if not contexts:
        return "No relevant documentation found."
    return "\n\n".join(contexts)


def golden_case_to_sample(case, results, *, chunk_cap: int) -> RagasEvalSample:
    contexts = build_contexts(results, chunk_cap=chunk_cap)
    return RagasEvalSample(
        user_input=case.query,
        response=_response_from_contexts(contexts),
        retrieved_contexts=contexts,
        reference=_reference(case),
        metadata={
            "category": case.category,
            "negative": case.negative,
            "expected_sources": list(case.expected_sources),
            "required_terms": list(case.required_terms),
        },
    )


def require_ragas() -> None:
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Ragas is not installed. Install with: pip install ragas") from exc

