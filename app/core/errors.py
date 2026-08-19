"""Giftie API가 외부에 반환하는 오류 코드와 예외 타입을 정의합니다."""

from enum import StrEnum
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    """API가 반환할 수 있는 오류 코드 전체 목록.

    Spring Boot와 프론트엔드는 한글 ``detail``을 파싱하지 않고 이 값을 기준으로
    분기합니다. 새 오류를 추가할 때는 임의 문자열을 쓰지 말고 반드시 여기에 먼저
    선언합니다.
    """

    # 인증·요청 형식
    INVALID_API_KEY = "INVALID_API_KEY"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    GIFT_INPUT_INVALID = "GIFT_INPUT_INVALID"

    # Giftie 업무 처리
    IMAGE_ANALYSIS_FAILED = "IMAGE_ANALYSIS_FAILED"
    RECOMMENDATION_FAILED = "RECOMMENDATION_FAILED"
    CONFIRMATION_FAILED = "CONFIRMATION_FAILED"
    AGENT_EXECUTION_FAILED = "AGENT_EXECUTION_FAILED"

    # 일반 HTTP·인프라 오류
    BAD_REQUEST = "BAD_REQUEST"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    CONFLICT = "CONFLICT"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    UPSTREAM_SERVICE_ERROR = "UPSTREAM_SERVICE_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
    HTTP_ERROR = "HTTP_ERROR"


class ApiErrorResponse(BaseModel):
    """모든 HTTP 오류가 공유하는 응답 형식."""

    status: Literal["ERROR"] = "ERROR"
    error_code: ErrorCode = Field(description="클라이언트가 분기 처리할 수 있는 오류 코드")
    detail: str = Field(description="사용자와 개발자가 읽을 수 있는 한글 오류 설명")
    errors: list[dict[str, Any]] | None = Field(
        default=None,
        description="요청 필드 검증 오류가 있을 때만 포함하는 세부 정보",
    )


class GiftieHTTPException(HTTPException):
    """HTTP 상태와 안정적인 오류 코드를 함께 전달하는 업무 예외."""

    def __init__(self, status_code: int, error_code: ErrorCode, detail: str) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
