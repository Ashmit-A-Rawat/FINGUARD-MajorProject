#!/usr/bin/env bash
# Show the latest progress bar line and finished models from a benchmark log.
# usage: scripts/run_status.sh <logfile>
log="${1:?usage: run_status.sh <logfile>}"
grep -E "^\[[#-]+\]" "$log" | tail -1
grep -E "PR-AUC=" "$log" | grep -v "^|"
pgrep -f run_.*_benchmark >/dev/null && echo "(running)" || echo "(not running)"
