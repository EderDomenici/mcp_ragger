import sqlite3
import time
from pathlib import Path


def init_metrics_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
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


def log_query(path: Path, query: str, latency_ms: float, scores: list[float], tokens_est: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
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
