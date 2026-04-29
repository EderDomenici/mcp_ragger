import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_rag_mcp.config import Settings
from langchain_rag_mcp.search import SearchService


class SearchServiceTests(unittest.TestCase):
    def test_search_returns_no_results_message_and_logs_empty_query(self):
        logs = []
        embedder = lambda query: [0.1, 0.2]
        vector_store = lambda vector, limit: SimpleNamespace(points=[])
        service = SearchService(
            Settings(min_score=0.60),
            embedder=embedder,
            query_points=vector_store,
            log_query=lambda query, latency_ms, scores, tokens_est: logs.append(
                (query, scores, tokens_est)
            ),
        )

        output = service.search("RunnableConfig")

        self.assertEqual(output, "No relevant documentation found.")
        self.assertEqual(logs[0][0], "RunnableConfig")
        self.assertEqual(logs[0][1], [])
        self.assertEqual(logs[0][2], 0)

    def test_search_reranks_filters_formats_and_logs_scores(self):
        logs = []
        points = [
            SimpleNamespace(
                score=0.61,
                payload={"title": "Other", "content": "unrelated"},
            ),
            SimpleNamespace(
                score=0.62,
                payload={"title": "RunnableConfig", "source": "docs", "content": "abcdef"},
            ),
            SimpleNamespace(
                score=0.20,
                payload={"title": "Low", "content": "RunnableConfig"},
            ),
        ]
        service = SearchService(
            Settings(top_k=2, candidate_k=20, chunk_cap=3, min_score=0.60),
            embedder=lambda query: [0.1, 0.2],
            query_points=lambda vector, limit: SimpleNamespace(points=points),
            log_query=lambda query, latency_ms, scores, tokens_est: logs.append(
                (query, scores, tokens_est)
            ),
        )

        output = service.search("RunnableConfig")

        self.assertEqual(output, "## RunnableConfig\nSource: docs\n\nabc\n\n---\n\n## Other\n\nunr")
        self.assertEqual(logs[0][1], [0.62, 0.61])
        self.assertEqual(logs[0][2], len(output) // 4)

    def test_search_rejects_high_score_out_of_domain_result_with_low_term_coverage(self):
        logs = []
        points = [
            SimpleNamespace(
                score=0.80,
                payload={
                    "title": "Deploy on Cloud",
                    "source": "https://docs.example/deploy.md",
                    "content": "Deploy an app to cloud with GitHub.",
                },
            )
        ]
        service = SearchService(
            Settings(min_score=0.60, min_query_term_coverage=0.75),
            embedder=lambda query: [0.1, 0.2],
            query_points=lambda vector, limit: SimpleNamespace(points=points),
            log_query=lambda query, latency_ms, scores, tokens_est: logs.append(
                (query, scores, tokens_est)
            ),
        )

        output = service.search("How do I deploy a Cloudflare Worker with wrangler?")

        self.assertEqual(output, "No relevant documentation found.")
        self.assertEqual(logs[0][1], [])


if __name__ == "__main__":
    unittest.main()
