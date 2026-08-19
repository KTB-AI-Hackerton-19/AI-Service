# ---------------------------------------------------------------------------
# mock 스테이지: 백엔드 연동 테스트용. GPU 도 모델도 필요 없습니다.
# 계약(요청·응답 형태)은 운영과 완전히 동일하고 응답만 고정값입니다.
#
#   docker build --target mock -t giftie-ai:mock .
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS mock

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_BACKEND=mock \
    TAVILY_ENABLED=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt

COPY app ./app
COPY mcp_servers ./mcp_servers

EXPOSE 8000 8300

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/openapi.json || exit 1

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


# ---------------------------------------------------------------------------
# 운영 스테이지(기본). 마지막 스테이지이므로 `docker build .` 는 이쪽을 만듭니다.
# MODEL_BACKEND=vllm 로 쓰면 이 컨테이너는 모델을 적재하지 않으므로 --gpus 도 필요 없습니다.
# GPU 는 별도의 vLLM 컨테이너 하나만 씁니다.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-qwen.txt ./
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r requirements-qwen.txt

COPY app ./app
COPY mcp_servers ./mcp_servers

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD curl -f http://localhost:8000/openapi.json || exit 1

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
