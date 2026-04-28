"""
Relatório de uso real do MCP em produção.

Lê metrics.db gerado pelo mcp_server.py e imprime resumo.

Uso:
    .venv/bin/python stats.py
    .venv/bin/python stats.py --last 50   # últimas 50 queries
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "metrics.db"

SCORE_WARN = 0.60  # abaixo disso o retrieval é suspeito


def run(limit: int | None = None):
    if not DB_PATH.exists():
        print("metrics.db não encontrado — faça ao menos uma query pelo MCP primeiro.")
        sys.exit(1)

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    limit_clause = f"LIMIT {limit}" if limit else ""

    rows = con.execute(
        f"SELECT * FROM queries ORDER BY ts DESC {limit_clause}"
    ).fetchall()

    if not rows:
        print("Nenhuma query registrada ainda.")
        return

    total = len(rows)
    avg_score = sum(r["score_top1"] or 0 for r in rows) / total
    avg_latency = sum(r["latency_ms"] for r in rows) / total
    avg_tokens = sum(r["tokens_est"] for r in rows) / total
    low_score = sum(1 for r in rows if (r["score_top1"] or 1) < SCORE_WARN)
    no_result = sum(1 for r in rows if r["results"] == 0)

    print(f"{'─' * 60}")
    print(f"Queries analisadas : {total}")
    print(f"Avg score top-1    : {avg_score:.3f}  (warn < {SCORE_WARN})")
    print(f"Avg latência       : {avg_latency:.0f}ms")
    print(f"Avg tokens/resposta: {avg_tokens:.0f}")
    print(f"Scores baixos      : {low_score} ({low_score / total * 100:.0f}%)")
    print(f"Sem resultado      : {no_result} ({no_result / total * 100:.0f}%)")

    print(f"\n{'─' * 60}")
    print("Últimas 10 queries:\n")
    for r in rows[:10]:
        score_flag = " ⚠" if (r["score_top1"] or 1) < SCORE_WARN else ""
        print(
            f"  [{r['score_top1'] or 0:.3f}{score_flag}] "
            f"{r['latency_ms']:5.0f}ms  ~{r['tokens_est']:4d}tok  "
            f"{r['query'][:55]}"
        )

    con.close()


if __name__ == "__main__":
    limit = None
    if "--last" in sys.argv:
        idx = sys.argv.index("--last")
        limit = int(sys.argv[idx + 1])
    run(limit)
