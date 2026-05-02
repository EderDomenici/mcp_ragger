"""
Golden-set benchmark for LangChain documentation retrieval.

Unlike scripts/benchmark.py, this checks expected source URLs and all required
answer terms for hand-written questions.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from langchain_rag_mcp.config import Settings
from langchain_rag_mcp.embeddings import create_embeddings
from langchain_rag_mcp.retrieval import query_term_coverage, rerank

CANDIDATE_K = 30
TOP_K = 5
CHUNK_CAP = 900
MIN_SCORE = 0.60
MIN_QUERY_TERM_COVERAGE = 0.75

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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class GoldenCase:
    query: str
    expected_sources: list[str]
    required_terms: list[str]
    category: str
    negative: bool = False
    max_score: float = 0.0


@dataclass
class GoldenResult:
    category: str
    query: str
    score: float
    latency_ms: float
    rank: int | None
    reciprocal_rank: float
    recall_at_3: bool
    required_terms_hit: bool
    query_term_coverage: float
    negative: bool
    passed: bool


GOLDEN_CASES: list[GoldenCase] = [
    GoldenCase(
        "In Python LangGraph, how do I pause execution for human input and resume it later?",
        ["/oss/python/langgraph/interrupts.md"],
        ["interrupt", "resume", "thread_id"],
        "langgraph_python",
    ),
    GoldenCase(
        "Where do LangGraph checkpoints fit into durable execution?",
        ["/oss/python/langgraph/durable-execution.md", "/oss/python/langgraph/persistence.md"],
        ["checkpoint", "durable", "execution"],
        "langgraph_python",
    ),
    GoldenCase(
        "What is the difference between LangChain short-term memory and long-term memory?",
        [
            "/oss/python/concepts/memory.md",
            "/oss/python/langchain/short-term-memory.md",
            "/oss/python/langchain/long-term-memory.md",
        ],
        ["memory", "thread", "long-term"],
        "langchain_python",
    ),
    GoldenCase(
        "How do I force structured output from a LangChain agent using a schema?",
        ["/oss/python/langchain/structured-output.md"],
        ["structured", "schema", "response_format"],
        "langchain_python",
    ),
    GoldenCase(
        "How do I add custom middleware around an agent model request?",
        ["/oss/python/langchain/middleware/custom.md"],
        ["middleware", "request", "agent"],
        "langchain_python",
    ),
    GoldenCase(
        "Which LangChain docs explain MCP integration for tools?",
        ["/oss/python/langchain/mcp.md", "/oss/javascript/langchain/mcp.md"],
        ["mcp", "tools"],
        "langchain_cross",
    ),
    GoldenCase(
        "In JavaScript LangGraph, where are conditional edges documented?",
        ["/oss/javascript/langgraph/graph-api.md", "/oss/javascript/langgraph/workflows-agents.md"],
        ["conditional", "edges"],
        "langgraph_javascript",
    ),
    GoldenCase(
        "How do I stream values from a LangGraph app in JavaScript?",
        ["/oss/javascript/langgraph/streaming.md"],
        ["stream", "values"],
        "langgraph_javascript",
    ),
    GoldenCase(
        "How do I configure LangSmith tracing without relying on environment variables?",
        ["/langsmith/trace-without-env-vars.md"],
        ["trace", "without", "environment"],
        "langsmith",
    ),
    GoldenCase(
        "Where does LangSmith document creating and managing datasets programmatically?",
        ["/langsmith/manage-datasets-programmatically.md"],
        ["dataset", "programmatically"],
        "langsmith",
    ),
    GoldenCase(
        "How do I run online evaluations using code in LangSmith?",
        ["/langsmith/online-evaluations-code.md"],
        ["online", "evaluation", "code"],
        "langsmith",
    ),
    GoldenCase(
        "Where are LangSmith prompt commits and prompt versioning explained?",
        ["/langsmith/prompt-commit.md", "/langsmith/manage-prompts.md"],
        ["prompt", "commit"],
        "langsmith",
    ),
    GoldenCase(
        "How do I use Zod refine in TypeScript?",
        [],
        [],
        "negative",
        negative=True,
        max_score=0.58,
    ),
    GoldenCase(
        "How do I deploy a Cloudflare Worker with wrangler?",
        [],
        [],
        "negative",
        negative=True,
        max_score=0.58,
    ),
    GoldenCase(
        "What is Prisma migrate dev used for?",
        [],
        [],
        "negative",
        negative=True,
        max_score=0.58,
    ),
]


def _source_matches(source: str, expected_sources: list[str]) -> bool:
    source = source.lower()
    return any(expected.lower() in source for expected in expected_sources)


def _rank(results, expected_sources: list[str]) -> int | None:
    if not expected_sources:
        return None
    for index, result in enumerate(results, start=1):
        source = str((result.payload or {}).get("source", ""))
        if _source_matches(source, expected_sources):
            return index
    return None


def evaluate_results(
    case: GoldenCase,
    results,
    combined_text: str,
    score: float,
    latency_ms: float,
    min_score: float = MIN_SCORE,
    min_query_term_coverage: float = MIN_QUERY_TERM_COVERAGE,
) -> GoldenResult:
    rank = _rank(results, case.expected_sources)
    recall_at_3 = rank is not None and rank <= 3
    reciprocal_rank = 1 / rank if rank else 0.0
    text = combined_text.lower()
    required_terms_hit = all(term.lower() in text for term in case.required_terms)
    coverage = query_term_coverage(case.query, results)

    if case.negative:
        passed = (score <= case.max_score or coverage < min_query_term_coverage) and rank is None
    else:
        passed = (
            score >= min_score
            and coverage >= min_query_term_coverage
            and recall_at_3
            and required_terms_hit
        )

    return GoldenResult(
        category=case.category,
        query=case.query,
        score=score,
        latency_ms=latency_ms,
        rank=rank,
        reciprocal_rank=reciprocal_rank,
        recall_at_3=recall_at_3,
        required_terms_hit=required_terms_hit,
        query_term_coverage=coverage,
        negative=case.negative,
        passed=passed,
    )


def run_query(case: GoldenCase, qdrant: QdrantClient, embeddings, settings: Settings) -> GoldenResult:
    t0 = time.perf_counter()
    query_vector = embeddings.embed_query(case.query)

    points = qdrant.query_points(
        collection_name=settings.collection,
        query=query_vector,
        limit=max(settings.candidate_k, CANDIDATE_K),
        with_payload=True,
    ).points
    latency_ms = (time.perf_counter() - t0) * 1000
    results = rerank(case.query, points, max(settings.top_k, TOP_K))
    score = results[0].score if results else 0.0
    chunk_cap = max(settings.chunk_cap, CHUNK_CAP)
    combined = "\n".join(str((r.payload or {}).get("content", ""))[:chunk_cap] for r in results)
    return evaluate_results(
        case,
        results,
        combined,
        score,
        latency_ms,
        min_score=settings.min_score,
        min_query_term_coverage=settings.min_query_term_coverage,
    )


def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-under", type=float, default=0.75)
    args = parser.parse_args()

    settings = Settings.from_env()
    qdrant = QdrantClient(**settings.qdrant_client_kwargs(), check_compatibility=False)
    embeddings = create_embeddings(
        settings.embedding_provider,
        llamacpp_url=settings.llamacpp_url,
        google_model=settings.google_embedding_model,
        google_output_dimensionality=settings.google_output_dimensionality,
    )
    print(f"Running {len(GOLDEN_CASES)} golden retrieval cases...\n")

    results = []
    for case in GOLDEN_CASES:
        result = run_query(case, qdrant, embeddings, settings)
        results.append(result)
        status = "✓" if result.passed else "✗"
        rank = "-" if result.rank is None else str(result.rank)
        print(
            f"{status} [{result.score:.3f}] {result.latency_ms:6.0f}ms "
            f"rank={rank:>2} cov={result.query_term_coverage:.2f} "
            f"terms={'Y' if result.required_terms_hit else 'n'} "
            f"{result.category:20s} {result.query[:72]}"
        )

    positives = [result for result in results if not result.negative]
    negatives = [result for result in results if result.negative]
    passed = sum(1 for result in results if result.passed)
    positive_recall = sum(1 for result in positives if result.recall_at_3) / len(positives)
    mrr = statistics.mean(result.reciprocal_rank for result in positives)
    negative_pass = sum(1 for result in negatives if result.passed) / len(negatives)
    hit_rate = passed / len(results)
    avg_latency = statistics.mean(result.latency_ms for result in results)

    print("\n" + "─" * 70)
    print(f"Golden hit rate : {passed}/{len(results)} ({hit_rate * 100:.0f}%)")
    print(f"Recall@3        : {positive_recall * 100:.0f}%")
    print(f"MRR             : {mrr:.3f}")
    print(f"Negative pass   : {negative_pass * 100:.0f}%")
    print(f"Avg latency     : {avg_latency:.0f}ms")

    return 0 if hit_rate >= args.fail_under else 1


if __name__ == "__main__":
    raise SystemExit(run())
