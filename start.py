#!/usr/bin/env python3
"""Compatibility wrapper for the package startup command."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from langchain_rag_mcp.startup import main


if __name__ == "__main__":
    raise SystemExit(main(ROOT))
