import sqlite3
import time
import httpx
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
LLAMACPP_URL = "http://localhost:8080/v1/embeddings"
COLLECTION = "langchain_docs"
TOP_K = 3
CHUNK_CAP = 600
MIN_SCORE = 0.60  # abaixo disso o chunk não é confiável

DB_PATH = Path(__file__).parent / "metrics.db"

mcp = FastMCP("langchain-rag")
qdrant = QdrantClient(url=QDRANT_URL)


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
        limit=TOP_K,
        with_payload=True,
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    results = [r for r in response.points if r.score >= MIN_SCORE]

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
