from .config import Settings
from .embeddings import create_embeddings
from .metrics import init_metrics_db, log_query
from .search import SearchService


def create_mcp(settings: Settings | None = None):
    from mcp.server.fastmcp import FastMCP
    from qdrant_client import QdrantClient

    settings = settings or Settings.from_env()
    init_metrics_db(settings.db_path)

    mcp = FastMCP("langchain-rag")
    qdrant = QdrantClient(url=settings.qdrant_url)
    embeddings = create_embeddings(
        settings.embedding_provider,
        llamacpp_url=settings.llamacpp_url,
        google_model=settings.google_embedding_model,
        google_output_dimensionality=settings.google_output_dimensionality,
    )
    service = SearchService(
        settings,
        embedder=embeddings.embed_query,
        query_points=lambda vector, limit: qdrant.query_points(
            collection_name=settings.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        ),
        log_query=lambda query, latency_ms, scores, tokens_est: log_query(
            settings.db_path,
            query,
            latency_ms,
            scores,
            tokens_est,
        ),
    )

    @mcp.tool()
    def search_docs(query: str) -> str:
        """Search LangChain documentation (LangChain, LangGraph, LangSmith).
        Use this whenever the user asks about LangChain concepts, APIs, or usage."""

        return service.search(query)

    return mcp

