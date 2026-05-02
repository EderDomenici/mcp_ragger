import os
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_ragas


class BenchmarkRagasCliTests(unittest.TestCase):
    def test_dry_run_loads_project_env_before_reading_openrouter_key(self):
        output = StringIO()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("benchmark_ragas.load_project_env", side_effect=lambda: os.environ.__setitem__("OPENROUTER_API_KEY", "sk-env")),
            patch("sys.stdout", output),
        ):
            exit_code = benchmark_ragas.run(["--dry-run"])

        self.assertEqual(exit_code, 0)
        self.assertIn("deepseek/deepseek-v4-flash", output.getvalue())

    def test_dry_run_prints_openrouter_fallback_order(self):
        env = {"OPENROUTER_API_KEY": "sk-test"}
        output = StringIO()

        with (
            patch.dict(os.environ, env, clear=True),
            patch("benchmark_ragas.load_project_env"),
            patch("sys.stdout", output),
        ):
            exit_code = benchmark_ragas.run(["--dry-run"])

        self.assertEqual(exit_code, 0)
        self.assertIn("deepseek/deepseek-v4-flash", output.getvalue())
        self.assertIn("google/gemini-2.5-flash", output.getvalue())

    def test_smoke_runs_openrouter_smoke_judge(self):
        env = {"OPENROUTER_API_KEY": "sk-test"}
        output = StringIO()

        class FakeClient:
            def smoke_judge(self):
                return benchmark_ragas.JudgeSmokeResult(text="ok", model="free-model")

        with (
            patch.dict(os.environ, env, clear=True),
            patch("benchmark_ragas.create_openai_compatible_client", return_value=FakeClient()),
            patch("sys.stdout", output),
        ):
            exit_code = benchmark_ragas.run(["--smoke"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Smoke judge response: ok", output.getvalue())
        self.assertIn("Model used: free-model", output.getvalue())

    def test_limit_is_accepted_for_partial_runs(self):
        env = {"OPENROUTER_API_KEY": "sk-test"}
        output = StringIO()

        with (
            patch.dict(os.environ, env, clear=True),
            patch("benchmark_ragas.load_project_env"),
            patch("sys.stdout", output),
        ):
            exit_code = benchmark_ragas.run(["--dry-run", "--limit", "3"])

        self.assertEqual(exit_code, 0)

    def test_build_metrics_returns_initialised_metric_objects(self):
        metrics = benchmark_ragas._build_metrics(llm=object())

        self.assertEqual([metric.name for metric in metrics], ["context_recall", "context_precision"])
        self.assertTrue(all(not isinstance(metric, type) for metric in metrics))

    def test_extract_scores_reads_ragas_evaluation_result_repr_dict(self):
        result = type("EvaluationResult", (), {"_repr_dict": {"context_recall": 0.5, "ignored": "x"}})()

        self.assertEqual(benchmark_ragas._extract_scores(result), {"context_recall": 0.5})

    def test_extract_scores_still_accepts_mapping_results(self):
        result = {"context_recall": 0.5, "ignored": "x"}

        self.assertEqual(benchmark_ragas._extract_scores(result), {"context_recall": 0.5})

    def test_average_score_requires_at_least_one_score(self):
        self.assertEqual(benchmark_ragas._average_score({}), 0.0)


if __name__ == "__main__":
    unittest.main()
