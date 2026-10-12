#!/usr/bin/env bash
# Usage: ./run.sh /path/to/audio/folder [--language Hindi] [--prompt "..."]   (see: ./run.sh --help)
D="$(cd "$(dirname "$0")" && pwd)"
[ -x "$D/.venv/bin/python" ] || { echo "Run $D/setup.sh first"; exit 1; }
PYTHONUTF8=1 exec "$D/.venv/bin/python" "$D/run_qc.py" "$@"
