#!/usr/bin/env bash
# Giftie AI Service를 백그라운드로 띄웁니다.
#
#   ./start.sh            # :8999 에서 기동 (이미 떠 있으면 아무것도 안 함)
#   PORT=8000 ./start.sh   # 다른 포트로 기동
#
# 로그는 logs/YYYY-MM-DD.log 에 날짜별로 쌓입니다(append).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-8999}"
HOST="${HOST:-0.0.0.0}"
VENV="$ROOT/.venv-runtime"
PID_FILE="$ROOT/.uvicorn.pid"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

if [ ! -x "$VENV/bin/uvicorn" ]; then
  echo "오류: $VENV/bin/uvicorn 이 없습니다. README의 Bedrock 실행 섹션대로 .venv-runtime 을 먼저 만들어 주세요." >&2
  exit 1
fi

if [ ! -f "$ROOT/.env" ]; then
  echo "오류: $ROOT/.env 가 없습니다. .env.example 을 복사해 채워 주세요." >&2
  exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "이미 실행 중입니다 (pid=$(cat "$PID_FILE"), :$PORT). 재시작하려면 ./restart.sh 를 쓰세요."
  exit 0
fi
rm -f "$PID_FILE"

{
  echo ""
  echo "==================== $(date '+%Y-%m-%d %H:%M:%S') start (:$PORT) ===================="
} >> "$LOG_FILE"

nohup "$VENV/bin/uvicorn" app.main:app --host "$HOST" --port "$PORT" \
  >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

# 뜨는지 확인 (최대 15초)
for _ in $(seq 1 15); do
  if curl -sf -o /dev/null -m 2 "http://127.0.0.1:$PORT/openapi.json"; then
    echo "기동 완료 (pid=$(cat "$PID_FILE")). Swagger: http://127.0.0.1:$PORT/docs"
    echo "로그: $LOG_FILE"
    exit 0
  fi
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "기동 실패. 로그를 확인하세요: $LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
  sleep 1
done

echo "경고: 15초 안에 응답이 없습니다. 계속 뜨는 중일 수 있으니 로그를 확인하세요: $LOG_FILE" >&2
