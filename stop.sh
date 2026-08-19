#!/usr/bin/env bash
# 백그라운드로 띄운 Giftie AI Service를 종료합니다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT/.uvicorn.pid"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d).log"

if [ ! -f "$PID_FILE" ]; then
  echo "PID 파일이 없습니다. 실행 중이 아닌 것 같습니다."
  exit 0
fi

PID="$(cat "$PID_FILE")"

if ! kill -0 "$PID" 2>/dev/null; then
  echo "pid=$PID 프로세스가 이미 죽어 있습니다. PID 파일만 정리합니다."
  rm -f "$PID_FILE"
  exit 0
fi

echo "종료 중... (pid=$PID)"
kill "$PID" 2>/dev/null || true

for _ in $(seq 1 10); do
  kill -0 "$PID" 2>/dev/null || break
  sleep 1
done

if kill -0 "$PID" 2>/dev/null; then
  echo "정상 종료되지 않아 강제 종료합니다 (kill -9)."
  kill -9 "$PID" 2>/dev/null || true
fi

rm -f "$PID_FILE"
mkdir -p "$LOG_DIR"
echo "==================== $(date '+%Y-%m-%d %H:%M:%S') stop (pid=$PID) ====================" >> "$LOG_FILE"
echo "종료 완료."
