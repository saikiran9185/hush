#!/bin/bash
# Open Hush Studio in your browser:  ./studio.sh            or with a file:  ./studio.sh recording.m4a
cd "$(dirname "$0")"
if ! curl -s -o /dev/null http://127.0.0.1:8740/api/job; then
  nohup .venv/bin/python studio/server.py > "$HOME/Library/Logs/Hush-Studio.log" 2>&1 &
  for _ in $(seq 20); do curl -s -o /dev/null http://127.0.0.1:8740/api/job && break; sleep 0.5; done
fi
if [ -n "${1:-}" ]; then
  path=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
  curl -s -X POST http://127.0.0.1:8740/api/open -d "{\"path\": \"$path\"}" > /dev/null
fi
open http://127.0.0.1:8740
