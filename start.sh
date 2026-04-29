#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -x "$DIR/.venv/bin/python" ]; then
  "$DIR/.venv/bin/python" "$DIR/start.py"
else
  python3 "$DIR/start.py"
fi
