"""Giftie 에이전트 공개 API의 요청·응답 데이터 모델."""

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator
from pydantic.alias_generators import to_camel

from app.schemas.recommendation import (
    Gender,
    MessageSource,
    SimpleGiftRecommendationResponse,
)


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
    """가격이 이미지에 적혀 있던 값인지, 검색으로 찾은 값인지, 추정치인지."""
    STATED = "stated"
    # 이미지에 금액이 없어 상품명으로 실제 판매가를 검색해 채운 값입니다.
    # 같은 상품의 다른 용량·구성이 섞일 수 있으므로 확인 화면에서 보여 주세요.
    SEARCHED = "searched"
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
    # 이미지에 금액이 없고 검색으로도 못 찾으면 비웁니다. 임의로 추정한 값을 채우면
    # 사용자가 그 값을 사실로 믿고, 답례 가격대까지 그 값에서 나옵니다.
    gift_price: int | None = Field(default=None, gt=0, le=100_000_000)
    age: int | None = Field(default=None, ge=0, le=120)
    gender: Gender | None = Field(
        default=None,
        description="답례 선물을 받을 상대의 성별. male/female, 모르면 생략 또는 null",
    )
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
        description="stated=이미지에 적힌 값, searched=상품명 검색으로 찾은 값, unknown=확인 불가.",
    )
    event: str | None = Field(default=None, max_length=50)
    event_date: date | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    @field_validator("gift_name", mode="before")
    @classmethod
    def normalize_required_text(cls, value: Any) -> Any:
        """필수 선물명의 앞뒤 공백을 제거합니다.

        공백 제거 후 빈 문자열이면 ``min_length`` 검증이 요청을 거부합니다.
        """
        return value.strip() if isinstance(value, str) else value

    @field_validator("age", mode="before")
    @classmethod
    def normalize_optional_age(cls, value: Any) -> Any:
        """0, 문자열 0, 빈 문자열과 null을 나이 미입력으로 통일합니다."""
        if value is None or value == 0:
            return None
        if isinstance(value, str) and value.strip() in {"", "0"}:
            return None
        return value

    @field_validator("gender", mode="before")
    @classmethod
    def normalize_optional_gender(cls, value: Any) -> Any:
        """빈 값과 unknown은 미입력으로, 대소문자·한국어 표기는 enum 값으로 통일합니다."""
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "unknown", "none", "null"}:
                return None
            aliases = {
                "m": Gender.MALE,
                "남": Gender.MALE,
                "남성": Gender.MALE,
                "f": Gender.FEMALE,
                "여": Gender.FEMALE,
                "여성": Gender.FEMALE,
            }
            return aliases.get(normalized, normalized)
        return value

    @field_validator("person_name", "relationship", "event", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: Any) -> Any:
        """선택 문자열의 앞뒤 공백을 제거하고 빈 값은 ``None``으로 바꿉니다."""
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip() or None
        return value

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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gift_data": {
                    "gift_name": "꽃",
                    "gift_price": 23333,
                    "age": 32,
                    "gender": "male",
                    "person_name": "김영삼",
                    "relationship": "친구",
                    "received_at": "2026-08-19",
                    "target_date": None,
                }
            }
        }
    )


class InputCategory(StrEnum):
    """사용자가 사진을 올릴 때 화면에서 직접 고른 종류.

    모델 추론보다 우선합니다. 사용자가 "경조사"를 골랐다면 이미지에서 무엇이
    읽히든 답례 "선물" 추천을 만들지 않습니다.
    """
    GIFT = "gift"          # 선물
    OCCASION = "occasion"  # 경조사 (축의금·조의금·청첩장·부고장)


# 백엔드가 한글이나 다른 표기로 보내도 받도록 별칭을 둡니다. 값 하나 때문에
# 연동이 막히는 것보다, 아는 표기를 모두 받아 주는 편이 낫습니다.
_INPUT_CATEGORY_ALIASES = {
    "gift": InputCategory.GIFT,
    "선물": InputCategory.GIFT,
    "present": InputCategory.GIFT,
    "occasion": InputCategory.OCCASION,
    "경조사": InputCategory.OCCASION,
    "event": InputCategory.OCCASION,
    "congratulation": InputCategory.OCCASION,
    "condolence": InputCategory.OCCASION,
}


class ImageRequest(BaseModel):
    """이미지 분석 API의 요청 본문. 현재는 S3 HTTP(S) 주소를 받습니다."""
    image_url: HttpUrl
    category: InputCategory | None = Field(
        default=None,
        description=(
            "사용자가 화면에서 고른 종류. gift(선물) 또는 occasion(경조사). "
            "생략하면 이미지 분석 결과로 판단합니다."
        ),
    )

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value: Any) -> Any:
        """한글·대문자·공백 표기를 허용합니다. 모르는 값은 미지정으로 봅니다."""
        if not isinstance(value, str):
            return value
        return _INPUT_CATEGORY_ALIASES.get(value.strip().lower())


class TaskStatus(StrEnum):
    """각 비동기 준비 작업의 상태."""
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    # 실패가 아니라 "이 입력에는 해당 작업이 필요 없음". 화면에 오류로 표시하지 마세요.
    SKIPPED = "SKIPPED"


class PreparedData(BaseModel):
    """선물 기록·캘린더·알림 준비 함수의 공통 결과."""
    status: TaskStatus = TaskStatus.SUCCESS
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


class ThankYouMessage(BaseModel):
    """사용자에게 보낼 감사 메시지와 그 문장이 어디서 나왔는지.

    ``generated_by`` 와 ``message_source`` 는 **서로 다른 것**을 말합니다.
    둘을 같은 것으로 읽으면 품질 지표가 뒤집힙니다.

    - ``generated_by``: 추천 **전체**를 만든 백엔드. ``recommend_gift.source`` 와 같은
      값이며, 모델 응답을 JSON 으로 읽는 데 성공했는지까지만 반영합니다.
    - ``message_source``: ``content`` **한 필드**를 누가 썼는지. 파싱에 성공해도
      메시지가 짧으면 정책이 템플릿으로 교체하는데, 그 사실은 여기에만 나타납니다.

    그래서 ``generated_by="BEDROCK_CLAUDE"`` 와 ``message_source="TEMPLATE_TOO_SHORT"``
    가 한 응답에 함께 나올 수 있고, 그것이 정상입니다. 카테고리·가격은 모델이 냈지만
    메시지만 템플릿이라는 뜻입니다. "모델이 쓴 문장인가" 는 ``message_source`` 로만
    판정하세요(``message_source == "MODEL"``).
    """

    tone: str = Field(description="메시지 말투. 화면 안내 문구로 쓸 수 있습니다.")
    content: str = Field(description="화면에 그대로 보여 줄 메시지 본문.")
    generated_by: str = Field(
        description=(
            "추천을 만든 백엔드. recommend_gift.source 와 같은 값입니다. "
            "예: BEDROCK_CLAUDE / BEDROCK_CLAUDE_FALLBACK / GEMMA_VLLM / MOCK. "
            "메시지 문장의 출처가 아닙니다 — 그것은 message_source 입니다."
        )
    )
    message_source: MessageSource = Field(
        description=(
            "content 를 누가 썼는지. MODEL 이면 모델 문장이고, "
            "그 밖의 값은 전부 정책 템플릿입니다."
        )
    )


class GiftRecommendationInfo(BaseModel):
    """추천 결과와 사용자에게 보낼 감사 메시지.

    ``status`` 가 ``SKIPPED`` 면 이 입력에 답례 선물 추천이 맞지 않는다는 뜻이며,
    ``reason`` 에 그 이유가 들어갑니다. 오류가 아니므로 화면에 실패로 표시하지 마세요.
    """
    status: TaskStatus = TaskStatus.SUCCESS
    recommend_gift: SimpleGiftRecommendationResponse | None = None
    # 예전에는 dict[str, str] 이라 OpenAPI 에 키가 하나도 드러나지 않았습니다.
    # 모델로 바꿔 계약을 명시합니다. JSON 키는 그대로이고 message_source 만 늘었습니다.
    message: ThankYouMessage | None = None
    error: str | None = None
    reason: str | None = Field(default=None, description="status=SKIPPED 일 때의 사유")


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


class RecommendRequest(BaseModel):
    """추천만 단독으로 요청합니다.

    나이·가격대·카테고리·성별만으로도 추천이 나옵니다. 사용자가 확인 화면에서
    조건을 바꿔 다시 추천받을 때, 이미지 분석을 다시 돌릴 이유가 없기 때문입니다.
    """

    age: int | None = Field(default=None, ge=0, le=120)
    gender: Gender = Gender.UNKNOWN
    budget_min: int | None = Field(default=None, ge=0, le=100_000_000)
    budget_max: int | None = Field(default=None, ge=0, le=100_000_000)
    categories: list[str] = Field(
        default_factory=list, max_length=3, description="지정하면 이 안에서만 추천합니다"
    )

    # 아래는 있으면 추천 품질이 올라가는 선택 입력입니다.
    gift_name: str | None = Field(default=None, max_length=200, description="받은 것. 답례 추천일 때")
    gift_price: int | None = Field(default=None, gt=0, le=100_000_000)
    person_name: str | None = Field(default=None, max_length=50)
    relationship: str | None = Field(default=None, max_length=50)
    event: str | None = Field(default=None, max_length=50)
    interests: list[str] = Field(default_factory=list, max_length=5)
    dislikes: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("gender", mode="before")
    @classmethod
    def normalize_optional_gender(cls, value: Any) -> Gender:
        """추천 단독 API에서도 빈 성별을 unknown으로 처리합니다."""
        normalized = GiftData.normalize_optional_gender(value)
        return normalized or Gender.UNKNOWN

    @model_validator(mode="after")
    def require_a_price_reference(self) -> "RecommendRequest":
        """금액이나 예산 중 하나는 있어야 합니다.

        답례 가격대는 받은 금액의 80~120% 로 정해집니다. 아무 기준도 없으면 예전에는
        30,000원을 채워 넣고 "받은 금액 30,000원의 80%~120%" 라고 근거까지 적었습니다.
        지어낸 값을 사실처럼 돌려주느니 무엇이 필요한지 알려 주는 편이 낫습니다.
        """
        if self.gift_price is None and self.budget_min is None and self.budget_max is None:
            raise ValueError("gift_price 또는 budget_min/budget_max 중 하나는 필요합니다.")
        return self


class RecommendResponse(BaseModel):
    """추천 단독 요청의 결과."""

    recommend_gift_info: GiftRecommendationInfo


class ConfirmResponse(BaseModel):
    """확정 처리 결과."""
    workflow_id: str
    approved: bool
    gift_data: PreparedData
    calendar_info: PreparedData
    noti_info: PreparedData
