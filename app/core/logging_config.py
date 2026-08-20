"""요청·응답을 시각·파라미터와 함께 기록하는 로깅 설정.

기존 로그는 시각이 없고(``INFO:     1.2.3.4:0 - "POST ..." 200 OK``), 어떤
파라미터로 들어온 요청인지, 무엇을 돌려줬는지, 실패했다면 무엇이 원인인지가
남지 않아 ngrok 너머에서 발생한 문제를 로그만 보고 재현할 수 없었다. 이 모듈은
요청을 다음 블록으로 남긴다.

.. code-block:: text

    [2026-08-20 08:57:09] [INFO] [211.244.225.224] [REQ] POST /api/v1/agent/confirm req=534faeb5
    PARAMETER: {"workflow_id": "...", ...}
    [2026-08-20 08:57:09] [INFO] [211.244.225.224] [RES] POST /api/v1/agent/confirm -> 200 (11.0ms) req=534faeb5
    RESULT: {"workflow_id": "...", ...}

같은 요청의 두 줄은 ``req=<id>``로 묶여서, 동시에 여러 요청이 들어와도 로그만
보고 어느 응답이 어느 요청 것인지 알 수 있다.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.config import settings

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

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


class GiftieLogFormatter(logging.Formatter):
    """일반 로그와 REQ/RES 블록을 같은 형식으로 찍는 포매터.

    시각은 컨테이너의 시스템 시간대(대개 UTC)가 아니라 ``settings.default_timezone``
    (Asia/Seoul) 기준으로 고정한다. ``app.services.clock`` 이 업무 로직에서 같은
    이유로 KST를 쓰는 것과 같은 문제를 로그에서도 피하기 위함이다 — 그러지 않으면
    로그 시각과 캘린더·알림에 찍히는 시각이 9시간 어긋나 보인다.
    """

    def __init__(self, tz: ZoneInfo) -> None:
        super().__init__()
        self._tz = tz

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, self._tz).strftime(datefmt or DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record)
        prefix = f"[{timestamp}] [{record.levelname}]"

        direction = getattr(record, "direction", None)
        if direction in ("REQ", "RES"):
            client = getattr(record, "client", "-")
            label = "PARAMETER" if direction == "REQ" else "RESULT"
            detail = getattr(record, "detail", "")
            lines = [
                f"{prefix} [{client}] [{direction}] {record.getMessage()}",
                f"{label}: {detail}",
            ]
        else:
            lines = [f"{prefix} [{record.name}] {record.getMessage()}"]

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            lines.append(record.exc_text)
        if record.stack_info:
            lines.append(self.formatStack(record.stack_info))
        return "\n".join(lines)


class DailyFileHandler(logging.Handler):
    """``logs/YYYY-MM-DD.log`` 에 직접 쓰고, 날짜가 바뀌면 스스로 다음 파일로 넘어간다.

    이전에는 ``start.sh`` 가 쉘 리다이렉트(``>> "$LOG_FILE"``)로 프로세스 시작
    시각에 파일명을 한 번만 고정했다. 그래서 서버를 안 내리고 자정을 넘기면
    다음날 로그도 계속 어제 날짜 파일에 쌓이는 문제가 있었다. 이 핸들러는 매
    기록마다(락 안에서) 오늘 날짜를 다시 확인해, 날짜가 바뀌었으면 이전 파일을
    닫고 새 파일을 연다 — 서버를 재시작하지 않아도 자정에 자동으로 넘어간다.
    """

    def __init__(self, log_dir: Path, tz: ZoneInfo) -> None:
        super().__init__()
        self._log_dir = log_dir
        self._tz = tz
        self._current_date = None
        self._stream = None

    def _ensure_stream(self) -> None:
        today = datetime.now(self._tz).date()
        if today == self._current_date:
            return
        if self._stream is not None:
            self._stream.close()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._stream = (self._log_dir / f"{today.isoformat()}.log").open("a", encoding="utf-8")
        self._current_date = today

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            self.handleError(record)
            return
        with self.lock:
            try:
                self._ensure_stream()
                self._stream.write(msg + "\n")
                self._stream.flush()
            except Exception:
                self.handleError(record)

    def close(self) -> None:
        with self.lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        super().close()


def configure_logging(level: int = logging.INFO) -> None:
    """루트 로거에 통일된 포맷 + 자동 날짜 로테이션 파일 핸들러를 붙인다.

    ``uvicorn app.main:app`` 로 실행하면 uvicorn 이 앱을 임포트하기 전에
    자체 로깅 설정(``dictConfig``)을 먼저 적용한다. 여기서 핸들러를 다시
    꽂아 주지 않으면 uvicorn 접속 로그와 우리 서비스 로그의 시각·형식이
    서로 달라진다.

    ``start.sh`` 는 이제 uvicorn의 표준출력을 파일로 리다이렉트하지 않는다
    (``> /dev/null``) — 이 핸들러가 ``logs/`` 에 직접, 날짜별로 쓰기 때문에
    이중 기록을 피하기 위함이다. 포그라운드에서 직접 실행할 때는 콘솔에도
    그대로 보인다(``StreamHandler``).
    """
    tz = ZoneInfo(settings.default_timezone)
    formatter = GiftieLogFormatter(tz)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = DailyFileHandler(LOG_DIR, tz)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [stream_handler, file_handler]

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
    """요청 블록(REQ, 파라미터), 응답 블록(RES, 결과)으로 나눠서 남긴다.

    동시 요청이 섞여도 어느 응답이 어느 요청 것인지 알 수 있도록 둘을 같은
    ``req=<id>``로 묶는다. 예상하지 못한 예외도 같은 req id로 남겨, 화면
    재현 없이 로그만으로 무엇이·왜·어떤 입력으로 실패했는지 추적할 수 있게 한다.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        started = time.monotonic()
        # Starlette Request.body() 는 결과를 캐시하므로, 여기서 한 번 읽어도
        # 이후 FastAPI 의 Pydantic 바디 파싱이 스트림을 다시 읽지 않고 캐시를 그대로 쓴다.
        body = await request.body()
        client = request.client.host if request.client else "-"

        access_logger.info(
            "%s %s req=%s",
            request.method,
            request.url.path,
            request_id,
            extra={"direction": "REQ", "client": client, "detail": _preview(body)},
        )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.monotonic() - started) * 1000
            access_logger.exception(
                "%s %s -> EXCEPTION (%.1fms) req=%s",
                request.method,
                request.url.path,
                duration_ms,
                request_id,
                extra={"direction": "RES", "client": client, "detail": "예외 발생 — 파라미터는 위 REQ 로그 참고"},
            )
            raise

        response_body = b"".join([chunk async for chunk in response.body_iterator])

        async def _replay():
            yield response_body

        response.body_iterator = _replay()

        duration_ms = (time.monotonic() - started) * 1000
        log = access_logger.warning if response.status_code >= 400 else access_logger.info
        log(
            "%s %s -> %s (%.1fms) req=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
            extra={"direction": "RES", "client": client, "detail": _preview(response_body)},
        )
        return response
