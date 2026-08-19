"""추출 결과를 공개 계약인 ``GiftData`` 로 안전하게 변환합니다.

``recommendation_policy`` 가 Qwen 출력을 안전한 추천 결과로 다듬는 것과 같은 역할입니다.

``GiftData`` 는 선물 1건만 표현하고 ``gift_price`` 가 1 이상 필수입니다.
이미지에는 여러 건이 있을 수 있고 금액이 아예 없는 경우(청첩장·부고장)도 있으므로,
이 모듈이 그 간극을 메웁니다.

- 다건이면 "대표 1건"만 남깁니다. 남은 건수는 ``dropped_records`` 로 알려 호출 측이 기록합니다.
- 금액이 없으면 ``strict_price`` 설정에 따라 실패시키거나 카테고리 추정가로 채웁니다.
"""

import logging
from dataclasses import dataclass, field

from app.core.config import settings
from app.schemas.agent import GiftData
from app.schemas.vision import Direction, ExtractedRecord, ExtractionResult, PriceBasis, RecordType

logger = logging.getLogger(__name__)

_GIFT_NAME_MAX = 200
_PERSON_NAME_MAX = 50
_PRICE_MAX = 100_000_000

# 이미지에서 금액을 못 읽었을 때 쓰는 카테고리별 추정가(원).
# 실제 값이 아니라 계약을 만족시키기 위한 자리표시자이므로, 이 값을 쓰면 이름에 "(금액 미상)"을 붙입니다.
_ESTIMATED_PRICE_BY_KEYWORD: tuple[tuple[tuple[str, ...], int], ...] = (
    (("조의", "부의", "장례", "근조"), 50_000),
    (("축의", "결혼", "혼례", "청첩"), 50_000),
    (("돌잔치", "백일", "출산"), 50_000),
    (("기프티콘", "음료", "카페", "커피"), 10_000),
    (("케이크", "디저트", "치킨", "외식"), 25_000),
    (("화장품", "향수", "뷰티"), 60_000),
    (("상품권",), 50_000),
)
_ESTIMATED_PRICE_DEFAULT = 30_000

_UNKNOWN_PRICE_SUFFIX = " (금액 미상)"


class GiftDataPolicyError(ValueError):
    """추출 결과로 유효한 ``GiftData`` 를 만들 수 없을 때 발생합니다."""


@dataclass
class GiftDataBuild:
    """``GiftData`` 와, 계약 밖으로 밀려난 정보들.

    Attributes:
        gift_data: 공개 API 로 나가는 결과.
        primary: 대표로 선택된 기록.
        dropped_records: 계약상 전달되지 못한 나머지 기록들.
        price_basis: ``gift_price`` 가 실제 값인지 추정치인지.
        warnings: 호출 측이 로그로 남길 주의사항.
    """

    gift_data: GiftData
    primary: ExtractedRecord
    dropped_records: list[ExtractedRecord] = field(default_factory=list)
    price_basis: PriceBasis = PriceBasis.STATED
    warnings: list[str] = field(default_factory=list)


def select_primary(records: list[ExtractedRecord]) -> ExtractedRecord | None:
    """대표 1건을 고릅니다.

    받은 것 중 금액이 가장 큰 건을 씁니다. 받은 게 없으면 초대장, 그것도 없으면 첫 건입니다.

    Args:
        records: 정규화된 기록 목록.

    Returns:
        대표 기록. 목록이 비어 있으면 ``None``.
    """
    if not records:
        return None

    received = [r for r in records if r.direction is Direction.RECEIVED]
    if received:
        return max(received, key=lambda r: (r.amount or 0, r.confidence))

    invitations = [r for r in records if r.record_type is RecordType.EVENT_INVITATION]
    if invitations:
        return invitations[0]

    return records[0]


def estimate_price(record: ExtractedRecord) -> int:
    """금액을 읽지 못했을 때 카테고리로 추정가를 고릅니다."""
    haystack = " ".join(
        filter(None, (record.category, record.event, record.item_name, record.brand, record.memo))
    )
    for keywords, price in _ESTIMATED_PRICE_BY_KEYWORD:
        if any(keyword in haystack for keyword in keywords):
            return price
    return _ESTIMATED_PRICE_DEFAULT


def build_gift_name(record: ExtractedRecord) -> str:
    """``gift_name`` 에 넣을 사람이 읽을 수 있는 이름을 만듭니다."""
    if record.item_name:
        name = f"{record.brand} {record.item_name}" if record.brand else record.item_name
    elif record.record_type is RecordType.EVENT_INVITATION:
        name = f"{record.event or '행사'} 초대"
    elif record.record_type is RecordType.MONEY:
        name = record.event or record.category or "현금"
    else:
        name = record.category or record.event or "받은 선물"

    return name.strip()[:_GIFT_NAME_MAX] or "받은 선물"


def build_gift_data(result: ExtractionResult) -> GiftDataBuild:
    """추출 결과를 ``GiftData`` 로 변환합니다.

    Args:
        result: ``vision_response_parser`` 가 정규화한 결과.

    Returns:
        ``GiftData`` 와 변환 과정에서 밀려난 정보들.

    Raises:
        GiftDataPolicyError: 기록이 하나도 없거나,
            ``strict_price`` 가 켜진 상태에서 금액을 읽지 못한 경우.
    """
    primary = select_primary(result.records)
    if primary is None:
        raise GiftDataPolicyError("이미지에서 선물·부조금 기록을 찾지 못했습니다.")

    warnings = list(result.warnings)
    dropped = [r for r in result.records if r is not primary]
    if dropped:
        warnings.append(
            f"이미지에서 {len(result.records)}건을 읽었지만 현재 GiftData 계약상 대표 1건만 전달합니다"
        )

    price_basis = PriceBasis.STATED
    gift_name = build_gift_name(primary)

    if primary.amount is not None:
        gift_price = min(primary.amount, _PRICE_MAX)
    elif settings.strict_price:
        raise GiftDataPolicyError(
            "이미지에서 금액을 읽지 못했습니다. 금액을 직접 입력해 주세요."
        )
    else:
        gift_price = estimate_price(primary)
        price_basis = PriceBasis.ESTIMATED
        # GiftData 에는 추정 여부를 담을 필드가 없으므로 이름에 드러내 사용자가 알아볼 수 있게 합니다.
        gift_name = f"{gift_name}{_UNKNOWN_PRICE_SUFFIX}"[:_GIFT_NAME_MAX]
        warnings.append(f"금액을 읽지 못해 {gift_price:,}원으로 추정했습니다")

    if primary.needs_review:
        warnings.append("사용자 확인 필요: " + ", ".join(primary.review_reasons))

    gift_data = GiftData(
        gift_name=gift_name,
        gift_price=gift_price,
        age=None,  # 이미지에서 나이를 알 수 있는 경우는 없습니다.
        person_name=(primary.counterpart_name or None) and primary.counterpart_name[:_PERSON_NAME_MAX],
        relationship=None,  # 관계는 백엔드가 인물 DB 에서 채우는 값입니다.
        received_at=primary.occurred_date,
        target_date=primary.event_date,
    )

    return GiftDataBuild(
        gift_data=gift_data,
        primary=primary,
        dropped_records=dropped,
        price_basis=price_basis,
        warnings=warnings,
    )
