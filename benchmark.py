"""
Benchmark offline do RAG.

Executa 15 queries douradas e valida:
  - Score de similaridade (proxy de relevância)
  - Latência de embed + busca
  - Estimativa de tokens retornados
  - Presença de keywords esperadas nos resultados

Uso:
    .venv/bin/python benchmark.py
"""

import time
import httpx
from dataclasses import dataclass
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
LLAMACPP_URL = "http://localhost:8080/v1/embeddings"
COLLECTION = "langchain_docs"
TOP_K = 3
CHUNK_CAP = 600
SCORE_THRESHOLD = 0.60

qdrant = QdrantClient(url=QDRANT_URL)

# (query, keywords que devem aparecer nos chunks retornados)
GOLDEN: list[tuple[str, list[str]]] = [
    ("how to create a chain in LangChain", ["chain", "invoke", "pipe"]),
    ("LangGraph state management", ["state", "graph", "node"]),
    ("LangSmith tracing setup", ["trace", "langsmith", "api_key"]),
    ("tool calling with LangChain agents", ["tool", "agent", "function"]),
    ("LangChain memory and conversation history", ["memory", "history", "messages"]),
    ("streaming responses in LangChain", ["stream", "chunk", "astream"]),
    ("LangGraph checkpointing persistence", ["checkpoint", "persist", "memory"]),
    ("custom callbacks in LangChain", ["callback", "handler", "on_"]),
    ("LangSmith evaluation datasets", ["dataset", "eval", "example"]),
    ("LangChain Expression Language LCEL", ["runnable", "pipe", "lcel"]),
    ("how to deploy a LangGraph agent", ["deploy", "langgraph", "server"]),
    ("LangChain output parsers", ["parser", "output", "format"]),
    ("LangChain retriever interface", ["retriever", "get_relevant", "invoke"]),
    ("LangSmith prompt hub", ["prompt", "hub", "pull"]),
    ("LangGraph multi-agent orchestration", ["multi", "agent", "supervisor"]),
]


@dataclass
class Result:
    query: str
    latency_ms: float
    score_top1: float
    score_avg: float
    tokens_est: int
    keyword_hit: bool
    pass_threshold: bool


def run_query(query: str, expected_keywords: list[str]) -> Result:
    t0 = time.perf_counter()

    resp = httpx.post(LLAMACPP_URL, json={"input": [query]}, timeout=30)
    resp.raise_for_status()
    query_vector = resp.json()["data"][0]["embedding"]

    response = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    results = response.points

    if not results:
        return Result(query, latency_ms, 0.0, 0.0, 0, False, False)

    scores = [r.score for r in results]
    combined = " ".join(r.payload.get("content", "")[:CHUNK_CAP] for r in results).lower()
    tokens_est = len(combined) // 4

    keyword_hit = any(kw.lower() in combined for kw in expected_keywords)

    return Result(
        query=query,
        latency_ms=latency_ms,
        score_top1=scores[0],
        score_avg=sum(scores) / len(scores),
        tokens_est=tokens_est,
        keyword_hit=keyword_hit,
        pass_threshold=scores[0] >= SCORE_THRESHOLD,
    )


def run():
    print(f"Running {len(GOLDEN)} benchmark queries...\n")
    results: list[Result] = []

    for query, keywords in GOLDEN:
        r = run_query(query, keywords)
        results.append(r)
        status = "✓" if r.pass_threshold and r.keyword_hit else "✗"
        print(
            f"{status} [{r.score_top1:.3f}] {r.latency_ms:6.0f}ms "
            f"~{r.tokens_est:4d}tok  {r.query[:60]}"
        )

    passed = sum(1 for r in results if r.pass_threshold and r.keyword_hit)
    avg_score = sum(r.score_top1 for r in results) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    avg_tokens = sum(r.tokens_est for r in results) / len(results)

    print(f"\n{'─' * 65}")
    print(f"Hit rate      : {passed}/{len(results)} ({passed / len(results) * 100:.0f}%)")
    print(f"Avg score     : {avg_score:.3f}  (threshold: {SCORE_THRESHOLD})")
    print(f"Avg latency   : {avg_latency:.0f}ms")
    print(f"Avg tokens/q  : {avg_tokens:.0f}")
    print(f"\nHit rate ≥ 80% = retrieval confiável")
    print(f"Avg score ≥ 0.65 = embeddings bem alinhados com a doc")


if __name__ == "__main__":
    run()
