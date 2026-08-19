#!/usr/bin/env bash
# Giftie AI Service를 백그라운드로 띄웁니다.
# macOS(로컬 개발)에서는 기본적으로 ngrok 터널링을 함께 켜고,
# Ubuntu 등 그 외 OS(서버 배포)에서는 ngrok 없이 IP:PORT로 바로 접근하는 것을 기본값으로 합니다.
#
#   ./start.sh              # :8000 에서 기동 (이미 떠 있으면 아무것도 안 함)
#   PORT=8000 ./start.sh    # 다른 포트로 기동
#   NGROK=0 ./start.sh      # ngrok 없이 로컬만 기동 (OS 기본값과 무관하게 강제)
#   NGROK=1 ./start.sh      # ngrok 강제 사용 (OS 기본값과 무관하게 강제)
#
# 로그는 logs/YYYY-MM-DD.log(API), logs/ngrok-YYYY-MM-DD.log(ngrok) 에 날짜별로 쌓입니다(append).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

OS_NAME="$(uname -s)"
if [ "$OS_NAME" = "Darwin" ]; then
  NGROK_DEFAULT=1
else
  NGROK_DEFAULT=0
fi

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
NGROK="${NGROK:-$NGROK_DEFAULT}"
VENV="$ROOT/.venv-runtime"
PID_FILE="$ROOT/.uvicorn.pid"
NGROK_PID_FILE="$ROOT/.ngrok.pid"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d).log"
NGROK_LOG_FILE="$LOG_DIR/ngrok-$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

if [ ! -x "$VENV/bin/uvicorn" ]; then
  echo "오류: $VENV/bin/uvicorn 이 없습니다. README의 Bedrock 실행 섹션대로 .venv-runtime 을 먼저 만들어 주세요." >&2
  exit 1
fi

if [ ! -f "$ROOT/.env" ]; then
  echo "오류: $ROOT/.env 가 없습니다. .env.example 을 복사해 채워 주세요." >&2
  exit 1
fi

API_ALREADY_RUNNING=0
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "이미 실행 중입니다 (pid=$(cat "$PID_FILE"), :$PORT)."
  API_ALREADY_RUNNING=1
else
  rm -f "$PID_FILE"
fi

if [ "$API_ALREADY_RUNNING" -eq 0 ]; then
  {
    echo ""
    echo "==================== $(date '+%Y-%m-%d %H:%M:%S') start (:$PORT) ===================="
  } >> "$LOG_FILE"

  nohup "$VENV/bin/uvicorn" app.main:app --host "$HOST" --port "$PORT" \
    >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"

  # 뜨는지 확인 (최대 15초)
  API_UP=0
  for _ in $(seq 1 15); do
    if curl -sf -o /dev/null -m 2 "http://127.0.0.1:$PORT/openapi.json"; then
      API_UP=1
      break
    fi
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "기동 실패. 로그를 확인하세요: $LOG_FILE" >&2
      rm -f "$PID_FILE"
      exit 1
    fi
    sleep 1
  done

  if [ "$API_UP" -eq 1 ]; then
    echo "기동 완료 (pid=$(cat "$PID_FILE")). Swagger: http://127.0.0.1:$PORT/docs"
  else
    echo "경고: 15초 안에 응답이 없습니다. 계속 뜨는 중일 수 있으니 로그를 확인하세요: $LOG_FILE" >&2
  fi
  echo "로그: $LOG_FILE"
fi

if [ "$NGROK" != "1" ]; then
  LOCAL_IP=""
  case "$OS_NAME" in
    Darwin) LOCAL_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)" ;;
    *) LOCAL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')" ;;
  esac
  if [ -n "$LOCAL_IP" ]; then
    echo "ngrok 미사용: 백엔드에서 http://$LOCAL_IP:$PORT 로 API를 호출하세요."
  else
    echo "ngrok 미사용: 백엔드에서 http://<서버 IP>:$PORT 로 API를 호출하세요."
  fi
fi

# ---------------------------------------------------------------- ngrok
if [ "$NGROK" != "1" ]; then
  exit 0
fi
if ! command -v ngrok >/dev/null 2>&1; then
  echo "ngrok 이 설치돼 있지 않아 터널링을 건너뜁니다 (NGROK=0 이면 이 메시지도 안 뜹니다)."
  exit 0
fi

if [ -f "$NGROK_PID_FILE" ] && kill -0 "$(cat "$NGROK_PID_FILE")" 2>/dev/null; then
  PUBLIC_URL="$(curl -sf -m 3 http://127.0.0.1:4040/api/tunnels \
    | grep -o '"public_url":"[^"]*"' | head -1 | cut -d'"' -f4 || true)"
  echo "ngrok 이미 실행 중입니다 (pid=$(cat "$NGROK_PID_FILE")). 공개 URL: ${PUBLIC_URL:-확인 실패, http://127.0.0.1:4040 참고}"
  exit 0
fi
rm -f "$NGROK_PID_FILE"

{
  echo ""
  echo "==================== $(date '+%Y-%m-%d %H:%M:%S') ngrok start (:$PORT) ===================="
} >> "$NGROK_LOG_FILE"

nohup ngrok http "$PORT" --log=stdout >> "$NGROK_LOG_FILE" 2>&1 &
echo $! > "$NGROK_PID_FILE"

PUBLIC_URL=""
for _ in $(seq 1 15); do
  PUBLIC_URL="$(curl -sf -m 2 http://127.0.0.1:4040/api/tunnels \
    | grep -o '"public_url":"[^"]*"' | head -1 | cut -d'"' -f4 || true)"
  [ -n "$PUBLIC_URL" ] && break
  sleep 1
done

if [ -n "$PUBLIC_URL" ]; then
  echo "ngrok 기동 완료 (pid=$(cat "$NGROK_PID_FILE")). 공개 URL: $PUBLIC_URL"
else
  echo "경고: ngrok 공개 URL을 못 가져왔습니다. 로그를 확인하세요: $NGROK_LOG_FILE" >&2
fi
echo "ngrok 로그: $NGROK_LOG_FILE"
