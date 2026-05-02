import unittest
from types import SimpleNamespace

from langchain_rag_mcp.evals.ragas_runner import (
    RagasEvalSample,
    build_contexts,
    golden_case_to_sample,
)


class RagasRunnerTests(unittest.TestCase):
    def test_build_contexts_includes_source_and_content(self):
        results = [
            SimpleNamespace(
                payload={
                    "title": "Intro",
                    "source": "https://docs.example.com/intro.md",
                    "content": "Use callbacks for tracing.",
                }
            )
        ]

        contexts = build_contexts(results, chunk_cap=40)

        self.assertEqual(len(contexts), 1)
        self.assertIn("Source: https://docs.example.com/intro.md", contexts[0])
        self.assertIn("Use callbacks", contexts[0])

    def test_golden_case_to_sample_preserves_expected_sources(self):
        case = SimpleNamespace(
            query="Where is tracing documented?",
            expected_sources=["/trace.md"],
            required_terms=["trace"],
            category="langsmith",
            negative=False,
        )
        results = [
            SimpleNamespace(payload={"title": "Trace", "source": "https://docs.example.com/trace.md", "content": "trace"})
        ]

        sample = golden_case_to_sample(case, results, chunk_cap=100)

        self.assertIsInstance(sample, RagasEvalSample)
        self.assertEqual(sample.user_input, "Where is tracing documented?")
        self.assertIn("trace", sample.reference)
        self.assertNotIn("Expected sources:", sample.reference)
        self.assertEqual(sample.metadata["category"], "langsmith")


if __name__ == "__main__":
    unittest.main()
