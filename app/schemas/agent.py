"""Giftie 에이전트 공개 API의 요청·응답 데이터 모델."""

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator
from pydantic.alias_generators import to_camel

from app.schemas.recommendation import SimpleGiftRecommendationResponse


def _normalize_date(value: Any) -> date | None:
    """빈 값과 잘못된 날짜를 ``None``으로 정규화합니다."""
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


class RecordKind(StrEnum):
    """기록의 종류."""
    GIFT = "gift"
    MONEY = "money"
    EVENT_INVITATION = "event_invitation"
    RECEIPT = "receipt"
    UNKNOWN = "unknown"


class RecordDirection(StrEnum):
    """내가 받은 것인지 보낸 것인지."""
    RECEIVED = "received"
    SENT = "sent"
    UNKNOWN = "unknown"


class PriceBasis(StrEnum):
    """가격이 이미지에 적혀 있던 값인지 추정치인지."""
    STATED = "stated"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class GiftRecordItem(BaseModel):
    """이미지에서 뽑아낸 기록 1건.

    ``GiftData``의 평면 필드는 대표 1건만 표현할 수 있지만, 실제 이미지에는
    계좌 거래내역 5건이나 선물함 목록 4건이 들어 있습니다. 그 전부가 여기에 담깁니다.

    ``gift_price``와 달리 여기의 ``price``는 ``None``을 허용합니다.
    청첩장처럼 금액이 아예 없는 항목을 있는 그대로 표현하기 위해서입니다.
    """
    record_id: str = Field(description="요청 안에서만 유효한 식별자. 사용자 수정본을 대응시킬 때 사용")
    record_type: RecordKind = RecordKind.UNKNOWN
    direction: RecordDirection = RecordDirection.UNKNOWN
    person_name: str | None = Field(default=None, max_length=50)
    gift_name: str = Field(default="", max_length=200)
    price: int | None = Field(default=None, ge=0, le=100_000_000)
    price_basis: PriceBasis = PriceBasis.UNKNOWN
    received_at: date | None = None
    event_date: date | None = Field(default=None, description="청첩장의 예식일 등")
    event: str | None = Field(default=None, max_length=50)
    category: str | None = Field(default=None, max_length=50)
    brand: str | None = Field(default=None, max_length=100)
    memo: str | None = Field(default=None, max_length=500)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = Field(default=False, description="true면 확인 화면에서 강조해 사용자 확인을 유도")
    review_reasons: list[str] = Field(default_factory=list)
    selected: bool = Field(default=True, description="사용자가 저장 대상으로 선택했는지")

    @field_validator("received_at", "event_date", mode="before")
    @classmethod
    def normalize_optional_date(cls, value: Any) -> date | None:
        return _normalize_date(value)


class GiftData(BaseModel):
    """모든 후속 작업이 공통으로 사용하는 정규화된 선물 정보.

    앞쪽 일곱 필드는 기존 계약 그대로이며 **대표 1건**을 담습니다.
    (여러 건이면 받은 금액이 가장 큰 건)
    뒤쪽 필드는 전부 기본값이 있는 선택 항목이라, 이를 모르는 코드도 그대로 동작합니다.
    """
    gift_name: str = Field(min_length=1, max_length=200)
    gift_price: int = Field(gt=0, le=100_000_000)
    age: int | None = Field(default=None, ge=0, le=120)
    person_name: str | None = Field(default=None, max_length=50)
    relationship: str | None = Field(default=None, max_length=50)
    received_at: date | None = None
    target_date: date | None = None

    # ── 이하 확장 필드. 전부 선택이며 기존 사용처에 영향을 주지 않습니다. ──
    records: list[GiftRecordItem] = Field(
        default_factory=list,
        description="이미지에서 읽은 전체 기록. 비어 있으면 위 평면 필드가 유일한 기록입니다.",
    )
    record_type: RecordKind = RecordKind.GIFT
    direction: RecordDirection = RecordDirection.RECEIVED
    price_basis: PriceBasis = Field(
        default=PriceBasis.STATED,
        description="estimated면 gift_price가 이미지에 적힌 값이 아니라 추정치입니다.",
    )
    event: str | None = Field(default=None, max_length=50)
    event_date: date | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    @field_validator("received_at", "target_date", "event_date", mode="before")
    @classmethod
    def normalize_optional_date(cls, value: Any) -> date | None:
        """날짜를 변환하며 빈 값과 잘못된 값은 정책상 ``None``으로 처리합니다.

        Args:
            value: JSON에서 받은 날짜, 문자열, 빈 값 또는 null.

        Returns:
            유효한 날짜면 ``date``, 그 외에는 ``None``.
        """
        return _normalize_date(value)

    @property
    def selected_records(self) -> list[GiftRecordItem]:
        """사용자가 저장 대상으로 남겨 둔 기록만 돌려줍니다."""
        return [r for r in self.records if r.selected]


class GiftDataRequest(BaseModel):
    """선물데이터 직접 전달 API의 요청 본문."""
    gift_data: GiftData


class ImageRequest(BaseModel):
    """이미지 분석 API의 요청 본문. 현재는 S3 HTTP(S) 주소를 받습니다."""
    image_url: HttpUrl


class TaskStatus(StrEnum):
    """각 비동기 준비 작업의 성공/실패 상태."""
    READY = "READY"
    ERROR = "ERROR"


class PreparedData(BaseModel):
    """선물 기록·캘린더·알림 준비 함수의 공통 결과."""
    status: TaskStatus = TaskStatus.READY
    payload: dict[str, Any] | None = None
    error: str | None = None


class CalendarDraft(BaseModel):
    """캘린더 등록용 초안. ``calendar_info.payload``의 타입을 명시한 것입니다.

    JSON에서는 camelCase(``startTime``, ``targetDate`` …)로 나가고 파이썬에서는
    snake_case로 다룹니다. 사용자가 확인 화면에서 이 구조를 수정해 ``/confirm``으로
    되돌려주면 그대로 Google Calendar에 등록됩니다.
    """
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    provider: str = "GOOGLE_MCP_DRAFT"
    registered: bool = False
    workflow_id: str | None = None
    title: str = Field(max_length=200)
    description: str = ""
    scheduled_date: date = Field(alias="date", description="일정 날짜(답례 준비일)")
    start_time: str = "10:00"
    duration_minutes: int = Field(default=30, ge=1, le=1440)
    timezone: str = "Asia/Seoul"
    reminders_minutes: list[int] = Field(default_factory=lambda: [0, 24 * 60])
    calendar_id: str = "primary"
    target_date: date | None = Field(default=None, description="답례를 실행할 날짜")

    # 등록 이후에만 채워집니다.
    event_id: str | None = None
    html_link: str | None = None
    register_error: str | None = None

    @field_validator("scheduled_date", "target_date", mode="before")
    @classmethod
    def normalize_dates(cls, value: Any) -> date | None:
        return _normalize_date(value)

    def to_payload(self) -> dict[str, Any]:
        """``PreparedData.payload``에 넣을 camelCase dict로 바꿉니다."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


class GiftRecommendationInfo(BaseModel):
    """실제 Qwen 추천 결과와 사용자에게 보낼 감사 메시지."""
    status: TaskStatus = TaskStatus.READY
    recommend_gift: SimpleGiftRecommendationResponse | None = None
    message: dict[str, str] | None = None
    error: str | None = None


class GiftAgentResponse(BaseModel):
    """네 비동기 작업 결과를 합친 최종 HTTP 응답."""
    gift_data: PreparedData
    calendar_info: PreparedData
    noti_info: PreparedData
    recommend_gift_info: GiftRecommendationInfo

    # ── 이하 확장 필드. 기존 사용처에 영향을 주지 않습니다. ──
    workflow_id: str | None = Field(
        default=None, description="네 작업 결과를 연결하는 ID. /confirm 에 그대로 돌려주세요."
    )
    requires_confirmation: bool = Field(
        default=False,
        description="true면 아직 캘린더에 등록되지 않았습니다. 사용자 확인 후 /confirm 을 호출하세요.",
    )


class ConfirmRequest(BaseModel):
    """사용자가 확인 화면에서 검토·수정한 뒤 보내는 확정 요청.

    AI 서비스는 상태를 보관하지 않습니다. 백엔드가 ``/from-image`` 응답을 들고 있다가
    사용자 수정본과 함께 그대로 되돌려주면 됩니다.
    """
    workflow_id: str = Field(min_length=1, max_length=64)
    gift_data: GiftData = Field(description="사용자가 수정한 기록. records 에 남은 항목만 저장됩니다.")
    calendar: CalendarDraft | None = Field(
        default=None, description="사용자가 수정한 일정 초안. 생략하면 gift_data 로 다시 계산합니다."
    )
    approved: bool = Field(default=True, description="false면 아무것도 등록하지 않습니다.")
    register_calendar: bool = True
    google_access_token: str | None = Field(
        default=None, description="사용자 Google OAuth access token. 없으면 서버 설정값을 씁니다."
    )
    calendar_id: str | None = None


class ConfirmResponse(BaseModel):
    """확정 처리 결과."""
    workflow_id: str
    approved: bool
    gift_data: PreparedData
    calendar_info: PreparedData
    noti_info: PreparedData
