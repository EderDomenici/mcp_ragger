import sqlite3
import time
import re
import httpx
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
LLAMACPP_URL = "http://localhost:8080/v1/embeddings"
COLLECTION = "langchain_docs"
TOP_K = 3
CANDIDATE_K = 20
CHUNK_CAP = 600
MIN_SCORE = 0.60  # abaixo disso o chunk não é confiável

DB_PATH = Path(__file__).parent / "metrics.db"

mcp = FastMCP("langchain-rag")
qdrant = QdrantClient(url=QDRANT_URL)

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


def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         REAL NOT NULL,
            query      TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            results    INTEGER NOT NULL,
            score_top1 REAL,
            score_avg  REAL,
            tokens_est INTEGER NOT NULL
        )
    """)
    con.commit()
    con.close()


def _log(query: str, latency_ms: float, scores: list[float], tokens_est: int):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO queries (ts, query, latency_ms, results, score_top1, score_avg, tokens_est) VALUES (?,?,?,?,?,?,?)",
        (
            time.time(),
            query,
            round(latency_ms, 2),
            len(scores),
            round(scores[0], 4) if scores else None,
            round(sum(scores) / len(scores), 4) if scores else None,
            tokens_est,
        ),
    )
    con.commit()
    con.close()


def _embed(text: str) -> list[float]:
    response = httpx.post(
        LLAMACPP_URL,
        json={"input": [text]},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text.lower())
        if token not in STOP_WORDS
    }


def _rerank(query: str, results):
    query_tokens = _tokens(query)
    ranked = []
    for result in results:
        payload = result.payload or {}
        haystack = " ".join(
            str(payload.get(key, ""))
            for key in ("title", "breadcrumb", "source", "content_type", "symbols", "content")
        )
        overlap = len(query_tokens & _tokens(haystack))
        exact_bonus = 0.08 if query.lower() in haystack.lower() else 0.0
        ranked.append((result.score + (0.025 * overlap) + exact_bonus, result))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [result for _, result in ranked[:TOP_K]]


_init_db()


@mcp.tool()
def search_docs(query: str) -> str:
    """Search LangChain documentation (LangChain, LangGraph, LangSmith).
    Use this whenever the user asks about LangChain concepts, APIs, or usage."""

    t0 = time.perf_counter()
    query_vector = _embed(query)

    response = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=CANDIDATE_K,
        with_payload=True,
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    results = [r for r in _rerank(query, response.points) if r.score >= MIN_SCORE]

    if not results:
        _log(query, latency_ms, [], 0)
        return "No relevant documentation found."

    parts = []
    scores = []
    for r in results:
        p = r.payload
        scores.append(r.score)
        chunk = f"## {p['title']}"
        if p.get("source"):
            chunk += f"\nSource: {p['source']}"
        chunk += f"\n\n{p['content'][:CHUNK_CAP]}"
        parts.append(chunk)

    output = "\n\n---\n\n".join(parts)
    _log(query, latency_ms, scores, len(output) // 4)

    return output


if __name__ == "__main__":
    mcp.run()
