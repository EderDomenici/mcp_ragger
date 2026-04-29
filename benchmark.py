"""
Benchmark offline do RAG.

Executa um conjunto expandido de queries douradas e valida:
  - Score de similaridade (proxy de relevancia)
  - Latencia de embed + busca
  - Estimativa de tokens retornados
  - Presenca de keywords esperadas nos resultados
  - Fonte esperada no top 1 / top 3, quando definida

Uso:
    .venv/bin/python benchmark.py
"""

import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from langchain_rag_mcp.retrieval import rerank

QDRANT_URL = "http://localhost:6333"
LLAMACPP_URL = "http://localhost:8080/v1/embeddings"
COLLECTION = "langchain_docs"
TOP_K = 3
CANDIDATE_K = 20
CHUNK_CAP = 600
SCORE_THRESHOLD = 0.60

qdrant = QdrantClient(url=QDRANT_URL)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

@dataclass(frozen=True)
class Case:
    query: str
    keywords: list[str]
    category: str
    source_any: list[str] = field(default_factory=list)


CASES: list[Case] = [
    Case("how to create a chain in LangChain", ["chain", "invoke", "pipe"], "langchain_core", ["langchain"]),
    Case("LangChain runnable invoke batch stream", ["runnable", "invoke", "batch", "stream"], "langchain_core", ["langchain"]),
    Case("LangChain Expression Language LCEL", ["runnable", "pipe", "lcel"], "langchain_core", ["langchain"]),
    Case("compose LangChain runnables with pipe operator", ["runnable", "pipe", "sequence"], "langchain_core", ["langchain"]),
    Case("LangChain prompt template from messages", ["prompt", "template", "messages"], "langchain_core", ["langchain"]),
    Case("LangChain chat model invoke messages", ["chat", "model", "invoke", "messages"], "langchain_core", ["langchain"]),
    Case("LangChain structured output with schema", ["structured", "schema", "output"], "langchain_core", ["langchain"]),
    Case("LangChain output parsers", ["parser", "output", "format"], "langchain_core", ["langchain"]),
    Case("LangChain retriever interface", ["retriever", "get_relevant", "invoke"], "langchain_retrieval", ["langchain"]),
    Case("create a vector store retriever LangChain", ["vector", "store", "retriever"], "langchain_retrieval", ["langchain"]),
    Case("LangChain document loaders", ["loader", "document", "load"], "langchain_retrieval", ["langchain"]),
    Case("LangChain text splitter chunk documents", ["splitter", "chunk", "document"], "langchain_retrieval", ["langchain"]),
    Case("LangChain retrieval augmented generation RAG", ["retrieval", "retriever", "context"], "langchain_retrieval", ["langchain"]),
    Case("LangChain similarity search vector store", ["similarity", "search", "vector"], "langchain_retrieval", ["langchain"]),
    Case("tool calling with LangChain agents", ["tool", "agent", "function"], "langchain_agents", ["langchain"]),
    Case("create LangChain agent with tools", ["agent", "tool", "create"], "langchain_agents", ["langchain"]),
    Case("LangChain tool decorator", ["tool", "decorator", "schema"], "langchain_agents", ["langchain"]),
    Case("LangChain agent middleware", ["middleware", "agent", "request"], "langchain_agents", ["langchain"]),
    Case("LangChain human in the loop agent", ["human", "interrupt", "agent"], "langchain_agents", ["langchain"]),
    Case("LangChain memory and conversation history", ["memory", "history", "messages"], "langchain_agents", ["langchain"]),
    Case("streaming responses in LangChain", ["stream", "chunk", "astream"], "langchain_runtime", ["langchain"]),
    Case("custom callbacks in LangChain", ["callback", "handler", "on_"], "langchain_runtime", ["langchain", "langsmith"]),
    Case("LangChain async invocation", ["async", "ainvoke", "await"], "langchain_runtime", ["langchain"]),
    Case("LangChain config tags metadata callbacks", ["config", "tags", "metadata"], "langchain_runtime", ["langchain"]),
    Case("LangChain rate limiting retries", ["retry", "rate", "limit"], "langchain_runtime", ["langchain"]),
    Case("LangGraph state management", ["state", "graph", "node"], "langgraph_core", ["langgraph"]),
    Case("LangGraph nodes and edges", ["node", "edge", "graph"], "langgraph_core", ["langgraph"]),
    Case("LangGraph conditional edges", ["conditional", "edge", "route"], "langgraph_core", ["langgraph"]),
    Case("LangGraph reducer state update", ["reducer", "state", "update"], "langgraph_core", ["langgraph"]),
    Case("LangGraph START END nodes", ["start", "end", "node"], "langgraph_core", ["langgraph"]),
    Case("LangGraph checkpointing persistence", ["checkpoint", "persist", "memory"], "langgraph_persistence", ["langgraph"]),
    Case("LangGraph memory saver checkpointer", ["memory", "checkpointer", "thread"], "langgraph_persistence", ["langgraph", "langsmith"]),
    Case("LangGraph human interrupt resume", ["interrupt", "resume", "human"], "langgraph_persistence", ["langgraph"]),
    Case("LangGraph time travel checkpoint", ["checkpoint", "time", "travel"], "langgraph_persistence", ["langgraph"]),
    Case("LangGraph durable execution", ["durable", "execution", "thread"], "langgraph_persistence", ["langgraph"]),
    Case("how to deploy a LangGraph agent", ["deploy", "langgraph", "server"], "langgraph_deploy", ["langgraph", "langsmith", "langchain"]),
    Case("LangGraph Platform deployment", ["platform", "deploy", "server"], "langgraph_deploy", ["langgraph", "langsmith", "langchain"]),
    Case("LangGraph server configuration", ["server", "configuration", "graph"], "langgraph_deploy", ["langgraph", "langsmith", "langchain"]),
    Case("LangGraph Studio local development", ["studio", "local", "graph"], "langgraph_deploy", ["langgraph", "langsmith", "langchain"]),
    Case("LangGraph cloud environment variables", ["environment", "variables", "deploy"], "langgraph_deploy", ["langgraph", "langsmith", "langchain"]),
    Case("LangGraph multi-agent orchestration", ["multi", "agent", "supervisor"], "langgraph_agents", ["langgraph", "langchain"]),
    Case("LangGraph supervisor pattern", ["supervisor", "agent", "handoff"], "langgraph_agents", ["langgraph", "langchain"]),
    Case("LangGraph swarm handoff", ["swarm", "handoff", "agent"], "langgraph_agents", ["langgraph", "langchain"]),
    Case("LangGraph workers workflow", ["worker", "workflow", "agent"], "langgraph_agents", ["langgraph", "langchain"]),
    Case("LangGraph tool calling agent", ["tool", "agent", "graph"], "langgraph_agents", ["langgraph", "langchain"]),
    Case("LangSmith tracing setup", ["trace", "langsmith", "api_key"], "langsmith_tracing", ["langsmith"]),
    Case("LangSmith trace LangChain application", ["trace", "langchain", "langsmith"], "langsmith_tracing", ["langsmith"]),
    Case("LangSmith environment variables", ["LANGSMITH", "API_KEY", "TRACING"], "langsmith_tracing", ["langsmith"]),
    Case("LangSmith traceable decorator", ["traceable", "decorator", "run"], "langsmith_tracing", ["langsmith"]),
    Case("LangSmith log LLM calls", ["log", "llm", "trace"], "langsmith_tracing", ["langsmith"]),
    Case("LangSmith evaluation datasets", ["dataset", "eval", "example"], "langsmith_eval", ["langsmith"]),
    Case("create dataset in LangSmith", ["dataset", "create", "example"], "langsmith_eval", ["langsmith"]),
    Case("LangSmith evaluate application", ["evaluate", "experiment", "dataset"], "langsmith_eval", ["langsmith"]),
    Case("LangSmith feedback scores", ["feedback", "score", "run"], "langsmith_eval", ["langsmith"]),
    Case("LangSmith prompt hub", ["prompt", "hub", "pull"], "langsmith_prompts", ["langsmith"]),
    Case("LangSmith manage prompts", ["prompt", "manage", "commit"], "langsmith_prompts", ["langsmith"]),
    Case("LangSmith playground prompts", ["playground", "prompt", "model"], "langsmith_prompts", ["langsmith"]),
    Case("LangSmith prompt versioning", ["prompt", "version", "commit"], "langsmith_prompts", ["langsmith"]),
    Case("Python versus JavaScript LangChain docs", ["python", "javascript", "langchain"], "ambiguous"),
    Case("OpenAI integration in LangChain", ["openai", "chat", "model"], "ambiguous", ["langchain"]),
]


@dataclass
class Result:
    category: str
    query: str
    latency_ms: float
    score_top1: float
    score_avg: float
    tokens_est: int
    keyword_hit: bool
    top1_source_hit: bool | None
    top3_source_hit: bool | None
    pass_threshold: bool
    passed: bool


def _source_hit(results, expected_sources: list[str], top_n: int) -> bool | None:
    if not expected_sources:
        return None
    wanted = [item.lower() for item in expected_sources]
    for result in results[:top_n]:
        source = str((result.payload or {}).get("source", "")).lower()
        if any(item in source for item in wanted):
            return True
    return False


def run_query(case: Case) -> Result:
    t0 = time.perf_counter()

    resp = httpx.post(LLAMACPP_URL, json={"input": [case.query]}, timeout=30)
    resp.raise_for_status()
    query_vector = resp.json()["data"][0]["embedding"]

    response = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=CANDIDATE_K,
        with_payload=True,
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    results = rerank(case.query, response.points, TOP_K)

    if not results:
        return Result(case.category, case.query, latency_ms, 0.0, 0.0, 0, False, False, False, False, False)

    scores = [r.score for r in results]
    combined = " ".join(r.payload.get("content", "")[:CHUNK_CAP] for r in results).lower()
    tokens_est = len(combined) // 4

    keyword_hit = any(kw.lower() in combined for kw in case.keywords)
    top1_source_hit = _source_hit(results, case.source_any, 1)
    top3_source_hit = _source_hit(results, case.source_any, 3)
    pass_threshold = scores[0] >= SCORE_THRESHOLD
    passed = pass_threshold and keyword_hit and (top3_source_hit is not False)

    return Result(
        category=case.category,
        query=case.query,
        latency_ms=latency_ms,
        score_top1=scores[0],
        score_avg=sum(scores) / len(scores),
        tokens_est=tokens_est,
        keyword_hit=keyword_hit,
        top1_source_hit=top1_source_hit,
        top3_source_hit=top3_source_hit,
        pass_threshold=pass_threshold,
        passed=passed,
    )


def run():
    print(f"Running {len(CASES)} benchmark queries...\n")
    results: list[Result] = []

    for case in CASES:
        r = run_query(case)
        results.append(r)
        status = "✓" if r.passed else "✗"
        source = ""
        if r.top3_source_hit is not None:
            source = f" src1={'Y' if r.top1_source_hit else 'n'} src3={'Y' if r.top3_source_hit else 'n'}"
        print(
            f"{status} [{r.score_top1:.3f}] {r.latency_ms:6.0f}ms "
            f"~{r.tokens_est:4d}tok kw={'Y' if r.keyword_hit else 'n'}{source} "
            f"{r.category:20s} {r.query[:58]}"
        )

    passed = sum(1 for r in results if r.passed)
    keyword_hits = sum(1 for r in results if r.keyword_hit)
    threshold_hits = sum(1 for r in results if r.pass_threshold)
    source_cases = [r for r in results if r.top3_source_hit is not None]
    top1_source_hits = sum(1 for r in source_cases if r.top1_source_hit)
    top3_source_hits = sum(1 for r in source_cases if r.top3_source_hit)
    avg_score = sum(r.score_top1 for r in results) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    avg_tokens = sum(r.tokens_est for r in results) / len(results)

    print(f"\n{'─' * 65}")
    print(f"Hit rate      : {passed}/{len(results)} ({passed / len(results) * 100:.0f}%)")
    print(f"Keyword hit   : {keyword_hits}/{len(results)} ({keyword_hits / len(results) * 100:.0f}%)")
    print(f"Score pass    : {threshold_hits}/{len(results)} ({threshold_hits / len(results) * 100:.0f}%)")
    if source_cases:
        print(f"Top1 source   : {top1_source_hits}/{len(source_cases)} ({top1_source_hits / len(source_cases) * 100:.0f}%)")
        print(f"Top3 source   : {top3_source_hits}/{len(source_cases)} ({top3_source_hits / len(source_cases) * 100:.0f}%)")
    print(f"Avg score     : {avg_score:.3f}  (threshold: {SCORE_THRESHOLD})")
    print(f"Avg latency   : {avg_latency:.0f}ms")
    print(f"Avg tokens/q  : {avg_tokens:.0f}")

    by_category: dict[str, list[Result]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)

    print("\nBy category:")
    for category, items in sorted(by_category.items()):
        cat_passed = sum(1 for item in items if item.passed)
        cat_keyword = sum(1 for item in items if item.keyword_hit)
        print(
            f"  {category:20s} {cat_passed:2d}/{len(items):2d} pass "
            f"kw {cat_keyword:2d}/{len(items):2d}"
        )

    print("\nHit rate >= 80% = retrieval confiavel")
    print("Top3 source >= 90% = roteamento bom entre LangChain/LangGraph/LangSmith")
    print("Avg score >= 0.65 = embeddings bem alinhados com a doc")


if __name__ == "__main__":
    run()
