"""Giftie API가 외부에 반환하는 오류 코드와 예외 타입을 정의합니다."""

from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field


class ApiErrorResponse(BaseModel):
    """모든 HTTP 오류가 공유하는 응답 형식."""

    status: Literal["ERROR"] = "ERROR"
    error_code: str = Field(description="클라이언트가 분기 처리할 수 있는 영문 오류 코드")
    detail: str = Field(description="사용자와 개발자가 읽을 수 있는 한글 오류 설명")
    errors: list[dict[str, Any]] | None = Field(
        default=None,
        description="요청 필드 검증 오류가 있을 때만 포함하는 세부 정보",
    )


class GiftieHTTPException(HTTPException):
    """HTTP 상태와 안정적인 오류 코드를 함께 전달하는 업무 예외."""

    def __init__(self, status_code: int, error_code: str, detail: str) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code

