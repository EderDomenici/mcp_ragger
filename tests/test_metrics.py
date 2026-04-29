import sqlite3
import tempfile
import unittest
from pathlib import Path

from langchain_rag_mcp import metrics


class MetricsTests(unittest.TestCase):
    def test_init_and_log_query_create_query_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "metrics.db"

            metrics.init_metrics_db(db_path)
            metrics.log_query(db_path, "RunnableConfig", 12.345, [0.91, 0.83], 42)

            con = sqlite3.connect(db_path)
            row = con.execute(
                "SELECT query, latency_ms, results, score_top1, score_avg, tokens_est FROM queries"
            ).fetchone()
            con.close()

            self.assertEqual(row, ("RunnableConfig", 12.35, 2, 0.91, 0.87, 42))


if __name__ == "__main__":
    unittest.main()
