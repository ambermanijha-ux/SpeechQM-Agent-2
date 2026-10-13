#!/usr/bin/env bash
# Usage: ./run.sh /path/to/audio/folder [--language Hindi] [--prompt "..."]   (see: ./run.sh --help)
D="$(cd "$(dirname "$0")" && pwd)"
[ -x "$D/.venv/bin/python" ] || { echo "Run $D/setup.sh first"; exit 1; }
# macOS: keep the computer from idle-sleeping during the run (a full run is long; sleep pauses it)
SLEEP_GUARD=(); command -v caffeinate >/dev/null 2>&1 && SLEEP_GUARD=(caffeinate -i)
PYTHONUTF8=1 exec "${SLEEP_GUARD[@]}" "$D/.venv/bin/python" "$D/run_qc.py" "$@"
