"""Spring Boot와 Giftie AI 서비스 사이의 API 키 인증을 담당합니다."""

import secrets

from fastapi import Header, HTTPException, status

from app.core.config import settings


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-KEY")) -> None:
    """요청의 X-API-KEY를 서버 설정과 상수 시간 방식으로 비교합니다.

    Args:
        x_api_key: Spring Boot가 HTTP 헤더로 전달한 내부 서비스 키.

    Raises:
        HTTPException: 키가 없거나 일치하지 않으면 401.
    """
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 AI 서비스 API 키입니다.",
        )
