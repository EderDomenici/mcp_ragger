import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_rag_mcp import retrieval


class RetrievalTests(unittest.TestCase):
    def test_tokens_drop_common_langchain_words(self):
        tokens = retrieval.tokens("How does LangChain invoke RunnableConfig?")

        self.assertEqual(tokens, {"does", "invoke", "runnableconfig"})

    def test_rerank_prefers_payload_overlap_and_exact_match(self):
        weak_vector_match = SimpleNamespace(
            score=0.70,
            payload={"title": "Other", "content": "unrelated content"},
        )
        strong_payload_match = SimpleNamespace(
            score=0.66,
            payload={"title": "RunnableConfig", "content": "RunnableConfig callbacks"},
        )

        ranked = retrieval.rerank("RunnableConfig", [weak_vector_match, strong_payload_match], top_k=2)

        self.assertEqual(ranked[0], strong_payload_match)

    def test_rerank_prefers_langchain_agent_middleware_source_for_agent_model_request(self):
        generic_deepagents_match = SimpleNamespace(
            score=0.66,
            payload={
                "title": "Prebuilt middleware",
                "breadcrumb": "Customize Deep Agents",
                "source": "https://docs.langchain.com/oss/python/deepagents/customization.md",
                "content": "Use custom middleware around an agent model request.",
            },
        )
        specific_langchain_match = SimpleNamespace(
            score=0.63,
            payload={
                "title": "Dynamic model selection",
                "breadcrumb": "Custom middleware",
                "source": "https://docs.langchain.com/oss/python/langchain/middleware/custom.md",
                "content": "Wrap an agent model request with middleware.",
            },
        )

        ranked = retrieval.rerank(
            "How do I add custom middleware around an agent model request?",
            [generic_deepagents_match, specific_langchain_match],
            top_k=2,
        )

        self.assertEqual(ranked[0], specific_langchain_match)

    def test_rerank_filters_to_top_k(self):
        results = [
            SimpleNamespace(score=0.90, payload={"title": "A", "content": "alpha"}),
            SimpleNamespace(score=0.80, payload={"title": "B", "content": "beta"}),
            SimpleNamespace(score=0.70, payload={"title": "C", "content": "gamma"}),
        ]

        ranked = retrieval.rerank("alpha beta gamma", results, top_k=2)

        self.assertEqual(len(ranked), 2)

    def test_format_results_includes_title_source_and_capped_content(self):
        result = SimpleNamespace(
            score=0.91,
            payload={
                "title": "RunnableConfig",
                "source": "https://docs.example/runnable",
                "content": "abcdef",
            },
        )

        output = retrieval.format_results([result], chunk_cap=3)

        self.assertEqual(output, "## RunnableConfig\nSource: https://docs.example/runnable\n\nabc")

    def test_query_term_coverage_requires_meaningful_query_terms(self):
        results = [
            SimpleNamespace(
                score=0.80,
                payload={
                    "title": "Deploy on Cloud",
                    "source": "https://docs.example/deploy.md",
                    "content": "Deploy an app to cloud with GitHub.",
                },
            )
        ]

        coverage = retrieval.query_term_coverage(
            "How do I deploy a Cloudflare Worker with wrangler?",
            results,
        )

        self.assertLess(coverage, 0.75)

    def test_query_term_coverage_passes_when_answer_terms_are_present(self):
        results = [
            SimpleNamespace(
                score=0.80,
                payload={
                    "title": "Custom middleware",
                    "content": "Use middleware around an agent model request.",
                },
            )
        ]

        coverage = retrieval.query_term_coverage(
            "How do I add custom middleware around an agent model request?",
            results,
        )

        self.assertGreaterEqual(coverage, 0.75)


if __name__ == "__main__":
    unittest.main()
