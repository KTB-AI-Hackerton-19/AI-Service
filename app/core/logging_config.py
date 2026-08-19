"""요청·응답을 시각·파라미터와 함께 기록하는 로깅 설정.

기존 로그는 시각이 없고(``INFO:     1.2.3.4:0 - "POST ..." 200 OK``), 어떤
파라미터로 들어온 요청인지, 무엇을 돌려줬는지, 실패했다면 무엇이 원인인지가
남지 않아 ngrok 너머에서 발생한 문제를 로그만 보고 재현할 수 없었다. 이 모듈은
요청 한 줄(``REQ``, 파라미터)과 응답 한 줄(``RES``, 상태·응답 내용)로 나눠 남기고,
같은 요청의 두 줄을 ``req=<id>``로 묶는다.
"""

import json
import logging
import time
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 요청·응답 본문에 섞여 들어올 수 있는 비밀값. 로그에는 절대 원문을 남기지 않는다.
_REDACT_KEYS = {
    "google_access_token",
    "access_token",
    "api_key",
    "x-api-key",
    "authorization",
    "token",
}
_BODY_PREVIEW_LIMIT = 2000

access_logger = logging.getLogger("giftie.access")


def configure_logging(level: int = logging.INFO) -> None:
    """루트 로거와 uvicorn 로거에 같은 타임스탬프 포맷을 적용한다.

    ``uvicorn app.main:app`` 로 실행하면 uvicorn 이 앱을 임포트하기 전에
    자체 로깅 설정(``dictConfig``)을 먼저 적용한다. 여기서 핸들러를 다시
    꽂아 주지 않으면 uvicorn 접속 로그와 우리 서비스 로그(``app.services.*``)의
    시각·형식이 서로 달라진다.
    """
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

    for name in ("uvicorn", "uvicorn.error"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers = []
        uv_logger.propagate = True

    # uvicorn.access 는 시각도 파라미터도 없는 한 줄(``"POST ..." 200``)만 남긴다.
    # RequestLoggingMiddleware 가 같은 정보를 시각·파라미터·소요시간과 함께 남기므로
    # 완전히 꺼서 로그가 두 줄로 중복되지 않게 한다.
    logging.getLogger("uvicorn.access").disabled = True


def _redact(value: Any) -> Any:
    """토큰·키처럼 로그에 남으면 안 되는 값을 가린다."""
    if isinstance(value, dict):
        return {
            k: ("***redacted***" if k.lower() in _REDACT_KEYS else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _preview(raw: bytes) -> str:
    """요청·응답 본문을 로그 한 줄에 넣을 수 있게 요약한다.

    JSON이면 비밀값을 가린 뒤 한 줄로 압축하고, JSON이 아니거나(예: 빈 본문)
    깨진 인코딩이면 있는 그대로 잘라서 보여준다. 어느 쪽이든 요청을 막지
    않는다 — 로깅 실패가 API 응답에 영향을 주면 안 되기 때문이다.
    """
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
        text = json.dumps(_redact(parsed), ensure_ascii=False)
    except (ValueError, UnicodeDecodeError):
        text = raw.decode("utf-8", errors="replace")
    if len(text) > _BODY_PREVIEW_LIMIT:
        text = text[:_BODY_PREVIEW_LIMIT] + "...(truncated)"
    return text


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """요청 한 줄, 응답 한 줄로 나눠서 남긴다.

    요청 줄에는 어떤 파라미터로 들어왔는지, 응답 줄에는 무엇을 돌려줬는지(성공이면
    상태 코드, 실패면 ``error_code``·``detail``까지)가 담긴다. 동시 요청이 섞여도
    어느 응답이 어느 요청 것인지 알 수 있도록 둘을 같은 ``req=<id>``로 묶는다.
    예상하지 못한 예외도 같은 req id로 남겨, 화면 재현 없이 로그만으로
    무엇이·왜·어떤 입력으로 실패했는지 추적할 수 있게 한다.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        started = time.monotonic()
        # Starlette Request.body() 는 결과를 캐시하므로, 여기서 한 번 읽어도
        # 이후 FastAPI 의 Pydantic 바디 파싱이 스트림을 다시 읽지 않고 캐시를 그대로 쓴다.
        body = await request.body()
        client = request.client.host if request.client else "-"
        query = dict(request.query_params)

        access_logger.info(
            "REQ req=%s %s %s client=%s query=%s body=%s",
            request_id,
            request.method,
            request.url.path,
            client,
            query,
            _preview(body),
        )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.monotonic() - started) * 1000
            access_logger.exception(
                "RES req=%s %s %s -> EXCEPTION duration=%.1fms",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        response_body = b"".join([chunk async for chunk in response.body_iterator])

        async def _replay():
            yield response_body

        response.body_iterator = _replay()

        duration_ms = (time.monotonic() - started) * 1000
        log = access_logger.warning if response.status_code >= 400 else access_logger.info
        log(
            "RES req=%s %s %s -> %s duration=%.1fms response=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            _preview(response_body),
        )
        return response
