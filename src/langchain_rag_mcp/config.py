from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = "http://localhost:6333"
    llamacpp_url: str = "http://localhost:8080/v1/embeddings"
    collection: str = "langchain_docs"
    top_k: int = 3
    candidate_k: int = 20
    chunk_cap: int = 600
    min_score: float = 0.60
    min_query_term_coverage: float = 0.75
    db_path: Path = Path(__file__).resolve().parents[2] / "metrics.db"
