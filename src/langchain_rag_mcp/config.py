import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .env import load_project_env


def _env_int(env: Mapping[str, str], key: str) -> int | None:
    value = env.get(key)
    return int(value) if value else None


def _env_int_default(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key)
    return int(value) if value else default


def _env_float_default(env: Mapping[str, str], key: str, default: float) -> float:
    value = env.get(key)
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    embedding_provider: str = "google"
    llamacpp_url: str = "http://localhost:8080/v1/embeddings"
    google_embedding_model: str = "models/gemini-embedding-2"
    google_output_dimensionality: int | None = None
    collection: str = "langchain_docs"
    docs_url: str = "https://docs.langchain.com/llms-full.txt"
    top_k: int = 3
    candidate_k: int = 20
    chunk_cap: int = 600
    min_score: float = 0.60
    min_query_term_coverage: float = 0.75
    db_path: Path = Path(__file__).resolve().parents[2] / "data" / "metrics.db"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        if env is None:
            load_project_env()
            env = os.environ
        return cls(
            qdrant_url=env.get("QDRANT_URL", cls.qdrant_url),
            qdrant_api_key=env.get("QDRANT_API_KEY") or None,
            embedding_provider=env.get("EMBEDDING_PROVIDER", cls.embedding_provider),
            llamacpp_url=env.get("LLAMACPP_URL", cls.llamacpp_url),
            google_embedding_model=env.get("GOOGLE_EMBEDDING_MODEL", cls.google_embedding_model),
            google_output_dimensionality=_env_int(env, "GOOGLE_EMBEDDING_DIMENSIONS"),
            collection=env.get("QDRANT_COLLECTION", cls.collection),
            docs_url=env.get("DOCS_URL", cls.docs_url),
            top_k=_env_int_default(env, "TOP_K", cls.top_k),
            candidate_k=_env_int_default(env, "CANDIDATE_K", cls.candidate_k),
            chunk_cap=_env_int_default(env, "CHUNK_CAP", cls.chunk_cap),
            min_score=_env_float_default(env, "MIN_SCORE", cls.min_score),
            min_query_term_coverage=_env_float_default(
                env,
                "MIN_QUERY_TERM_COVERAGE",
                cls.min_query_term_coverage,
            ),
        )

    def qdrant_client_kwargs(self) -> dict[str, str]:
        kwargs = {"url": self.qdrant_url}
        if self.qdrant_api_key:
            kwargs["api_key"] = self.qdrant_api_key
        return kwargs
