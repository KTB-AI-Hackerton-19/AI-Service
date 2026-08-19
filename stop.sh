#!/usr/bin/env bash
# 백그라운드로 띄운 Giftie AI Service(+ngrok)를 종료합니다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT/.uvicorn.pid"
NGROK_PID_FILE="$ROOT/.ngrok.pid"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d).log"
NGROK_LOG_FILE="$LOG_DIR/ngrok-$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

stop_pid_file() {  # stop_pid_file <pid_file> <log_file> <label>
  local pid_file="$1" log_file="$2" label="$3"

  if [ ! -f "$pid_file" ]; then
    echo "$label: PID 파일이 없습니다. 실행 중이 아닌 것 같습니다."
    return 0
  fi

  local pid
  pid="$(cat "$pid_file")"

  if ! kill -0 "$pid" 2>/dev/null; then
    echo "$label: pid=$pid 프로세스가 이미 죽어 있습니다. PID 파일만 정리합니다."
    rm -f "$pid_file"
    return 0
  fi

  echo "$label 종료 중... (pid=$pid)"
  kill "$pid" 2>/dev/null || true

  for _ in $(seq 1 10); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done

  if kill -0 "$pid" 2>/dev/null; then
    echo "$label: 정상 종료되지 않아 강제 종료합니다 (kill -9)."
    kill -9 "$pid" 2>/dev/null || true
  fi

  rm -f "$pid_file"
  echo "==================== $(date '+%Y-%m-%d %H:%M:%S') stop (pid=$pid) ====================" >> "$log_file"
  echo "$label 종료 완료."
}

stop_pid_file "$NGROK_PID_FILE" "$NGROK_LOG_FILE" "ngrok"
stop_pid_file "$PID_FILE" "$LOG_FILE" "API"
