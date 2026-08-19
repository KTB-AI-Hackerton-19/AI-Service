"""Giftie 에이전트 공개 API의 요청·응답 데이터 모델."""

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from app.schemas.recommendation import SimpleGiftRecommendationResponse


class HeartData(BaseModel):
    """모든 후속 작업이 공통으로 사용하는 정규화된 선물 정보."""
    gift_name: str = Field(min_length=1, max_length=200)
    gift_price: int = Field(gt=0, le=100_000_000)
    age: int | None = Field(default=None, ge=0, le=120)
    person_name: str | None = Field(default=None, max_length=50)
    relationship: str | None = Field(default=None, max_length=50)
    received_at: date | None = None
    target_date: date | None = None

    @field_validator("received_at", "target_date", mode="before")
    @classmethod
    def normalize_optional_date(cls, value: Any) -> date | None:
        """날짜를 변환하며 빈 값과 잘못된 값은 정책상 ``None``으로 처리합니다.

        Args:
            value: JSON에서 받은 날짜, 문자열, 빈 값 또는 null.

        Returns:
            유효한 날짜면 ``date``, 그 외에는 ``None``.
        """
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None


class HeartDataRequest(BaseModel):
    """마음데이터 직접 전달 API의 요청 본문."""
    heart_data: HeartData


class ImageRequest(BaseModel):
    """이미지 분석 API의 요청 본문. 현재는 S3 HTTP(S) 주소를 받습니다."""
    image_url: HttpUrl


class TaskStatus(StrEnum):
    """각 비동기 준비 작업의 성공/실패 상태."""
    READY = "READY"
    ERROR = "ERROR"


class PreparedData(BaseModel):
    """마음 기록·캘린더·알림 mock 함수의 공통 결과."""
    status: TaskStatus = TaskStatus.READY
    payload: dict[str, Any] | None = None
    error: str | None = None


class GiftRecommendationInfo(BaseModel):
    """실제 Qwen 추천 결과와 사용자에게 보낼 감사 메시지."""
    model_config = ConfigDict(populate_by_name=True)
    status: TaskStatus = TaskStatus.READY
    recommendations: SimpleGiftRecommendationResponse | None = Field(
        default=None, serialization_alias="추천선물"
    )
    message: dict[str, str] | None = Field(default=None, serialization_alias="텍스트")
    error: str | None = None


class GiftAgentResponse(BaseModel):
    """네 비동기 작업 결과를 합친 최종 HTTP 응답."""
    model_config = ConfigDict(populate_by_name=True)
    heart_data: PreparedData = Field(serialization_alias="마음데이터")
    calendar_info: PreparedData = Field(serialization_alias="캘린더정보")
    notification_info: PreparedData = Field(serialization_alias="알림정보")
    gift_recommendation_info: GiftRecommendationInfo = Field(serialization_alias="추천선물정보")
