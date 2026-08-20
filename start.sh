#!/usr/bin/env bash
# Giftie AI Service를 백그라운드로 띄웁니다.
# macOS(로컬 개발)에서는 기본적으로 ngrok 터널링을 함께 켜고,
# Ubuntu 등 그 외 OS(서버 배포)에서는 ngrok 없이 IP:PORT로 바로 접근하는 것을 기본값으로 합니다.
# Google Calendar MCP 서버(:8300)도 기본으로 같이 띄웁니다 — 이게 안 떠 있으면
# /confirm 의 캘린더 등록이 "MCP 서버에 연결할 수 없습니다"로 실패합니다.
#
#   ./start.sh              # :8000 API + :8300 MCP 에서 기동 (이미 떠 있으면 아무것도 안 함)
#   PORT=8000 ./start.sh    # API를 다른 포트로 기동
#   NGROK=0 ./start.sh      # ngrok 없이 로컬만 기동 (OS 기본값과 무관하게 강제)
#   NGROK=1 ./start.sh      # ngrok 강제 사용 (OS 기본값과 무관하게 강제)
#   MCP=0 ./start.sh        # Calendar MCP 서버 없이 기동 (캘린더는 초안까지만)
#   MCP_PORT=8300 ./start.sh # MCP 포트 변경 (.env 의 CALENDAR_MCP_URL 도 맞춰야 함)
#
# 로그는 logs/YYYY-MM-DD.log(API), logs/mcp-YYYY-MM-DD.log(MCP), logs/ngrok-YYYY-MM-DD.log(ngrok)
# 에 날짜별로 쌓입니다(append).
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
MCP="${MCP:-1}"
MCP_PORT="${MCP_PORT:-8300}"
MCP_HOST="${MCP_HOST:-0.0.0.0}"
VENV="$ROOT/.venv-runtime"
PID_FILE="$ROOT/.uvicorn.pid"
NGROK_PID_FILE="$ROOT/.ngrok.pid"
MCP_PID_FILE="$ROOT/.mcp.pid"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d).log"
NGROK_LOG_FILE="$LOG_DIR/ngrok-$(date +%Y-%m-%d).log"
MCP_LOG_FILE="$LOG_DIR/mcp-$(date +%Y-%m-%d).log"

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

  # 표준출력은 버린다 — 앱이 logs/YYYY-MM-DD.log 에 직접, 날짜별로 쓰므로
  # 여기서 파일로 또 리다이렉트하면 모든 줄이 두 번씩 남는다(app/core/logging_config.py 참고).
  nohup "$VENV/bin/uvicorn" app.main:app --host "$HOST" --port "$PORT" \
    > /dev/null 2>&1 &
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
      echo "(앱 로깅이 초기화되기 전에 죽었다면 여기 안 남을 수 있습니다 — 그땐 포그라운드로 직접 실행해서 원인을 보세요: $VENV/bin/uvicorn app.main:app --port $PORT)" >&2
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

# ---------------------------------------------------------------- Calendar MCP
if [ "$MCP" = "1" ]; then
  if [ -f "$MCP_PID_FILE" ] && kill -0 "$(cat "$MCP_PID_FILE")" 2>/dev/null; then
    echo "Calendar MCP 이미 실행 중입니다 (pid=$(cat "$MCP_PID_FILE"), :$MCP_PORT)."
  else
    rm -f "$MCP_PID_FILE"
    {
      echo ""
      echo "==================== $(date '+%Y-%m-%d %H:%M:%S') mcp start (:$MCP_PORT) ===================="
    } >> "$MCP_LOG_FILE"

    CALENDAR_MCP_HOST="$MCP_HOST" CALENDAR_MCP_PORT="$MCP_PORT" \
      nohup "$VENV/bin/python" -m mcp_servers.google_calendar \
      >> "$MCP_LOG_FILE" 2>&1 &
    echo $! > "$MCP_PID_FILE"

    MCP_UP=0
    for _ in $(seq 1 15); do
      if (exec 3<>"/dev/tcp/127.0.0.1/$MCP_PORT") 2>/dev/null; then
        exec 3>&- 3<&- 2>/dev/null || true
        MCP_UP=1
        break
      fi
      if ! kill -0 "$(cat "$MCP_PID_FILE")" 2>/dev/null; then
        echo "Calendar MCP 기동 실패. 로그를 확인하세요: $MCP_LOG_FILE" >&2
        rm -f "$MCP_PID_FILE"
        break
      fi
      sleep 1
    done

    if [ "$MCP_UP" -eq 1 ]; then
      echo "Calendar MCP 기동 완료 (pid=$(cat "$MCP_PID_FILE"), :$MCP_PORT)."
    else
      [ -f "$MCP_PID_FILE" ] && echo "경고: Calendar MCP가 15초 안에 포트를 열지 않았습니다. 로그를 확인하세요: $MCP_LOG_FILE" >&2
    fi
    echo "Calendar MCP 로그: $MCP_LOG_FILE"
  fi
else
  echo "Calendar MCP 비활성화(MCP=0) — /confirm 의 캘린더 등록은 초안까지만 만들어집니다."
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
