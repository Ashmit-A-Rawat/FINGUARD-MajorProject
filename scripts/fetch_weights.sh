#!/usr/bin/env bash
# Resumable weight download for flaky connections: reconnects whenever the transfer stalls.
# usage: scripts/fetch_weights.sh <url> <output-file>
url="${1:?url}"; out="${2:?output file}"
until curl -sS -L -C - --fail --connect-timeout 20 --speed-limit 30000 --speed-time 25 -o "$out" "$url"; do
  echo "stalled or interrupted at $(stat -f%z "$out" 2>/dev/null || echo 0) bytes; reconnecting..." >&2
  sleep 2
done
echo "complete: $(stat -f%z "$out") bytes"
