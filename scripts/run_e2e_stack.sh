#!/usr/bin/env bash
# 백엔드 연동용 실제 E2E 스택을 한 번에 띄웁니다.
#
#   ./scripts/run_e2e_stack.sh           # 기동
#   ./scripts/run_e2e_stack.sh --stop    # 종료
#   ./scripts/run_e2e_stack.sh --status  # 상태 확인
#
# 구성
#   :8001  vLLM (Gemma4-12B-QAT + MTP)   ← GPU 필요
#   :8300  Google Calendar MCP 서버
#   :8000  Giftie AI Service             ← 백엔드가 호출할 주소
#
# mock 이 아니라 실제 모델·실제 검색·실제 캘린더로 동작합니다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_VENV="${VLLM_VENV:-$(dirname "$ROOT")/.venv}"
LOG_DIR="${LOG_DIR:-$ROOT/.e2e-logs}"
HOST="${BIND_HOST:-0.0.0.0}"

VLLM_MODEL_ID="${VLLM_MODEL_ID:-google/gemma-4-12B-it-qat-w4a16-ct}"
VLLM_DRAFT_ID="${VLLM_DRAFT_ID:-google/gemma-4-12B-it-assistant}"
SERVED_NAME="${SERVED_NAME:-gemma4-12b-qat}"

mkdir -p "$LOG_DIR"

port_pid() { ss -ltnp 2>/dev/null | grep ":$1 " | grep -oP 'pid=\K[0-9]+' | head -1; }

wait_for() {  # wait_for <url> <초> <이름>
  for _ in $(seq 1 "$2"); do
    curl -sf -o /dev/null -m 2 "$1" && return 0
    sleep 1
  done
  return 1
}

stop_all() {
  for port in 8000 8300 8001; do
    pid="$(port_pid "$port" || true)"
    if [ -n "$pid" ]; then
      echo "  :$port 종료 (pid=$pid)"
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 3
  echo "종료 완료."
}

status() {
  for entry in "8001 vLLM" "8300 MCP" "8000 API"; do
    set -- $entry
    pid="$(port_pid "$1" || true)"
    if [ -n "$pid" ]; then
      printf "  :%s %-5s 실행 중 (pid=%s)\n" "$1" "$2" "$pid"
    else
      printf "  :%s %-5s 내려감\n" "$1" "$2"
    fi
  done
}

case "${1:-start}" in
  --stop)   stop_all; exit 0 ;;
  --status) status; exit 0 ;;
esac

# ---------------------------------------------------------------- 사전 점검
if [ ! -f "$ROOT/.env" ]; then
  echo "오류: $ROOT/.env 가 없습니다. .env.example 을 복사해 채워 주세요." >&2
  exit 1
fi
if [ ! -x "$VLLM_VENV/bin/vllm" ]; then
  echo "오류: vLLM 을 찾을 수 없습니다: $VLLM_VENV/bin/vllm" >&2
  echo "      VLLM_VENV 환경변수로 경로를 지정하세요." >&2
  exit 1
fi

# ---------------------------------------------------------------- vLLM
if [ -n "$(port_pid 8001 || true)" ]; then
  echo "1. vLLM 이미 실행 중 (:8001)"
else
  echo "1. vLLM 기동 ($VLLM_MODEL_ID + MTP)"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  PATH="$VLLM_VENV/bin:$PATH" \
  nohup "$VLLM_VENV/bin/vllm" serve "$VLLM_MODEL_ID" \
    --served-model-name "$SERVED_NAME" \
    --port 8001 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.844 \
    --max-num-seqs 16 \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    --limit-mm-per-prompt '{"image": 2}' \
    --speculative-config "{\"model\": \"$VLLM_DRAFT_ID\", \"num_speculative_tokens\": 1}" \
    > "$LOG_DIR/vllm.log" 2>&1 &
  echo "   모델 적재와 컴파일에 2~3분 걸립니다..."
  wait_for "http://127.0.0.1:8001/v1/models" 420 vLLM \
    || { echo "   실패. $LOG_DIR/vllm.log 를 확인하세요." >&2; exit 1; }
  echo "   준비 완료"
fi

# ---------------------------------------------------------------- MCP
if [ -n "$(port_pid 8300 || true)" ]; then
  echo "2. Calendar MCP 이미 실행 중 (:8300)"
else
  echo "2. Calendar MCP 기동"
  CALENDAR_MCP_HOST="$HOST" CALENDAR_MCP_PORT=8300 \
    nohup "$ROOT/.venv/bin/python" -m mcp_servers.google_calendar \
    > "$LOG_DIR/mcp.log" 2>&1 &
  sleep 3
  echo "   준비 완료"
fi

# ---------------------------------------------------------------- API
if [ -n "$(port_pid 8000 || true)" ]; then
  echo "3. AI Service 재기동 (:8000)"
  kill "$(port_pid 8000)" 2>/dev/null || true
  sleep 2
else
  echo "3. AI Service 기동 (:8000)"
fi

MODEL_BACKEND=vllm \
VLLM_BASE_URL=http://127.0.0.1:8001 \
CALENDAR_MCP_URL=http://127.0.0.1:8300/mcp \
  nohup "$ROOT/.venv/bin/python" -m uvicorn app.main:app \
  --host "$HOST" --port 8000 > "$LOG_DIR/api.log" 2>&1 &

wait_for "http://127.0.0.1:8000/openapi.json" 60 API \
  || { echo "   실패. $LOG_DIR/api.log 를 확인하세요." >&2; exit 1; }
echo "   준비 완료"

# ---------------------------------------------------------------- 안내
LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
API_KEY="$(grep -E '^API_KEY=' "$ROOT/.env" | cut -d= -f2- || echo local-development-key)"

cat <<EOF

────────────────────────────────────────────────────────────
실제 E2E 스택이 떴습니다. mock 이 아닙니다.

  백엔드에서 호출할 주소 : http://${LAN_IP:-<이-머신-IP>}:8000
  X-API-KEY             : ${API_KEY}
  Swagger               : http://${LAN_IP:-<이-머신-IP>}:8000/docs

빠른 확인:
  curl -X POST http://${LAN_IP:-127.0.0.1}:8000/api/v1/agent/from-gift-data \\
    -H 'Content-Type: application/json' -H 'X-API-KEY: ${API_KEY}' \\
    -d '{"gift_data":{"gift_name":"스타벅스 케이크","gift_price":35000,"person_name":"김민수"}}'

로그: $LOG_DIR/{vllm,mcp,api}.log
종료: ./scripts/run_e2e_stack.sh --stop
────────────────────────────────────────────────────────────
EOF
