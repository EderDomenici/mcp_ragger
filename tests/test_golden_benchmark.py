import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_golden


class GoldenBenchmarkTests(unittest.TestCase):
    def test_memory_case_accepts_concepts_overview_as_relevant_source(self):
        memory_case = next(
            case
            for case in benchmark_golden.GOLDEN_CASES
            if case.query == "What is the difference between LangChain short-term memory and long-term memory?"
        )

        self.assertIn("/oss/python/concepts/memory.md", memory_case.expected_sources)

    def test_expected_url_metrics_use_ranked_results(self):
        results = [
            SimpleNamespace(payload={"source": "https://docs.example.com/other.md"}),
            SimpleNamespace(payload={"source": "https://docs.example.com/target.md", "content": "answer question"}),
            SimpleNamespace(payload={"source": "https://docs.example.com/third.md"}),
        ]
        case = benchmark_golden.GoldenCase(
            query="answer question",
            expected_sources=["/target.md"],
            required_terms=["answer"],
            category="unit",
        )

        metrics = benchmark_golden.evaluate_results(case, results, "answer question text", 0.71, 42)

        self.assertTrue(metrics.recall_at_3)
        self.assertEqual(metrics.rank, 2)
        self.assertEqual(metrics.reciprocal_rank, 0.5)
        self.assertTrue(metrics.passed)

    def test_negative_case_passes_when_score_is_low_and_no_expected_source_hits(self):
        results = [SimpleNamespace(payload={"source": "https://docs.example.com/unrelated.md"})]
        case = benchmark_golden.GoldenCase(
            query="unrelated framework question",
            expected_sources=[],
            required_terms=[],
            category="negative",
            negative=True,
            max_score=0.50,
        )

        metrics = benchmark_golden.evaluate_results(case, results, "unrelated", 0.42, 12)

        self.assertTrue(metrics.passed)
        self.assertTrue(metrics.negative)

    def test_negative_case_passes_when_high_score_has_low_query_term_coverage(self):
        results = [
            SimpleNamespace(
                payload={
                    "source": "https://docs.example.com/deploy.md",
                    "title": "Deploy on Cloud",
                    "content": "Deploy an app to cloud with GitHub.",
                }
            )
        ]
        case = benchmark_golden.GoldenCase(
            query="How do I deploy a Cloudflare Worker with wrangler?",
            expected_sources=[],
            required_terms=[],
            category="negative",
            negative=True,
            max_score=0.58,
        )

        metrics = benchmark_golden.evaluate_results(case, results, "Deploy an app to cloud.", 0.65, 12)

        self.assertLess(metrics.query_term_coverage, benchmark_golden.MIN_QUERY_TERM_COVERAGE)
        self.assertTrue(metrics.passed)

    def test_required_terms_are_all_required(self):
        case = benchmark_golden.GoldenCase(
            query="question",
            expected_sources=["/target.md"],
            required_terms=["interrupt", "resume"],
            category="unit",
        )
        results = [SimpleNamespace(payload={"source": "https://docs.example.com/target.md"})]

        metrics = benchmark_golden.evaluate_results(case, results, "interrupt only", 0.80, 10)

        self.assertFalse(metrics.required_terms_hit)
        self.assertFalse(metrics.passed)


if __name__ == "__main__":
    unittest.main()
