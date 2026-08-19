"""FastAPI에서 발생하는 모든 예외를 Giftie 공통 오류 JSON으로 변환합니다."""

import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.errors import ApiErrorResponse, ErrorCode

logger = logging.getLogger(__name__)

_STATUS_ERROR_CODES = {
    400: ErrorCode.BAD_REQUEST,
    401: ErrorCode.AUTHENTICATION_ERROR,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.METHOD_NOT_ALLOWED,
    409: ErrorCode.CONFLICT,
    422: ErrorCode.VALIDATION_ERROR,
    429: ErrorCode.RATE_LIMITED,
    500: ErrorCode.INTERNAL_SERVER_ERROR,
    502: ErrorCode.UPSTREAM_SERVICE_ERROR,
    503: ErrorCode.SERVICE_UNAVAILABLE,
    504: ErrorCode.UPSTREAM_TIMEOUT,
}


def _response(status_code: int, error_code: ErrorCode, detail: str, errors=None) -> JSONResponse:
    """오류 모델을 JSONResponse로 직렬화합니다."""
    body = ApiErrorResponse(
        error_code=error_code,
        detail=detail,
        errors=errors,
    ).model_dump(exclude_none=True)
    return JSONResponse(status_code=status_code, content=body)


def register_exception_handlers(app: FastAPI) -> None:
    """업무·검증·예상하지 못한 오류 처리기를 애플리케이션에 등록합니다."""

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_: Request, exc: HTTPException) -> JSONResponse:
        error_code = getattr(
            exc,
            "error_code",
            _STATUS_ERROR_CODES.get(exc.status_code, ErrorCode.HTTP_ERROR),
        )
        detail = exc.detail if isinstance(exc.detail, str) else "요청을 처리할 수 없습니다."
        return _response(exc.status_code, error_code, detail)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        errors = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ()) if part != "body"),
                "message": "입력값의 형식이나 범위가 올바르지 않습니다.",
                "type": error.get("type", "validation_error"),
            }
            for error in exc.errors()
        ]
        return _response(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.VALIDATION_ERROR,
            "요청 데이터 형식이 올바르지 않습니다. 입력값을 확인해 주세요.",
            errors,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("처리되지 않은 API 오류 path=%s", request.url.path, exc_info=exc)
        return _response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCode.INTERNAL_SERVER_ERROR,
            "요청 처리 중 내부 오류가 발생했습니다.",
        )
