"""이미지 분석 내부에서만 쓰는 데이터 모델.

공개 API 계약은 ``app/schemas/agent.py`` 의 ``GiftData`` 하나입니다.
여기 있는 타입은 VLM 출력에서 ``GiftData`` 로 가는 중간 단계이며 HTTP 응답에 나가지 않습니다.

이미지 한 장에서 여러 건이 나오는 경우(계좌 거래내역, 선물함 목록, 영수증)를 다루기 위해
내부에서는 항상 목록으로 들고 다니다가, ``GiftData`` 로 넘길 때 대표 1건만 남깁니다.
"""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class RecordType(StrEnum):
    """추출된 기록의 종류."""

    GIFT = "gift"
    MONEY = "money"
    EVENT_INVITATION = "event_invitation"
    RECEIPT = "receipt"
    UNKNOWN = "unknown"


class Direction(StrEnum):
    """내가 받은 것인지 보낸 것인지."""

    RECEIVED = "received"
    SENT = "sent"
    UNKNOWN = "unknown"


class PriceBasis(StrEnum):
    """gift_price 값이 어디서 왔는지."""

    STATED = "stated"  # 이미지에 적혀 있던 금액
    ESTIMATED = "estimated"  # 카테고리 기준 추정치
    UNKNOWN = "unknown"


class ExtractedRecord(BaseModel):
    """이미지에서 뽑아 정규화한 기록 1건."""

    record_type: RecordType = RecordType.UNKNOWN
    direction: Direction = Direction.UNKNOWN
    counterpart_name: str | None = None
    occurred_date: date | None = Field(default=None, description="실제로 주고받은 날짜")
    event_date: date | None = Field(default=None, description="청첩장의 예식일 등 행사 날짜")
    item_name: str | None = None
    brand: str | None = None
    category: str | None = None
    event: str | None = Field(default=None, description="생일 / 결혼 / 조의 등 계기")
    amount: int | None = Field(default=None, description="원화 정수. 음수·할인 항목은 제거됨")
    memo: str | None = None
    confidence: float = 0.0
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """이미지 한 장의 추출 결과 전체."""

    image_kind: str = "other"
    records: list[ExtractedRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
