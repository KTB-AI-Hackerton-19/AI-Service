"""추출 결과를 공개 계약인 ``GiftData`` 로 안전하게 변환합니다.

``recommendation_policy`` 가 Qwen 출력을 안전한 추천 결과로 다듬는 것과 같은 역할입니다.

``GiftData`` 는 선물 1건만 표현하는데 이미지에는 여러 건이 있을 수 있고, 금액이
아예 없는 경우(청첩장·부고장, 금액이 안 보이는 선물 카드)도 있습니다.
이 모듈이 그 간극을 메웁니다.

- 다건이면 "대표 1건"만 남깁니다. 남은 건수는 ``dropped_records`` 로 알려 호출 측이 기록합니다.
- 금액이 없으면 비운 채로 둡니다. 카테고리로 추정하지 않습니다.
  브랜드를 모르는 추정가는 실제와 몇 배씩 어긋나는데(TWG Tea 를 10,000원으로 추정,
  실제 3~7만원) 사용자는 그 값을 사실로 받아들입니다. ``strict_price`` 를 켜면
  비우는 대신 실패시킵니다.
"""

import logging
import re
from dataclasses import dataclass, field

from app.core.config import settings
from app.schemas.agent import GiftData, GiftRecordItem, PriceBasis, RecordDirection, RecordKind
from app.schemas.vision import Direction, ExtractedRecord, ExtractionResult, RecordType
from app.services.recommendation_policy import canonical_category, is_condolence

logger = logging.getLogger(__name__)

_GIFT_NAME_MAX = 200
_PERSON_NAME_MAX = 50
_PRICE_MAX = 100_000_000
# 청첩장은 결혼에만 씁니다. 조의 판정은 recommendation_policy 의 목록 하나를 같이 씁니다.
_WEDDING_KEYWORDS = ("결혼", "혼인", "웨딩", "화혼", "약혼")
_MONEY_KINDS = ("축의금", "조의금", "부의금", "부조금", "축하금", "세뱃돈", "용돈")



# ── 기록 분류 정규화 ──────────────────────────────────────────────────────
# VLM 은 category 를 자유 서술로 씁니다("기프티콘/음료", "기프티콘/상품권", "화장품").
# 백엔드는 여섯 개(``recommendation_policy.SAFE_EXAMPLES`` 다섯 + "기타")만 받고
# 나머지는 전부 "기타" 로 떨어뜨리므로, 그대로 내보내면 대부분이 기타로 뭉개집니다.
#
# 프롬프트로 목록을 강제하지 않고 여기서 결정론적으로 바꿉니다. 이유는 두 가지입니다.
# 하나, 같은 값을 ``build_gift_name`` 이 선물명 대체로 씁니다 — 상품명이 없는 기록에서
# "기프티콘/음료" 는 이름 구실을 하지만 "디저트" 는 하지 않습니다. 그래서 내부
# ``ExtractedRecord.category`` 는 원문 그대로 두고 **계약으로 나갈 때만** 바꿉니다.
# 둘, 목록을 프롬프트에 실으면 모델이 애매한 기록을 억지로 다섯 개 중 하나에 밀어
# 넣습니다. 조의금·축의금처럼 다섯 개 어디에도 속하지 않는 기록이 실제로 많습니다.
#
# 못 맞추면 원문을 그대로 내보냅니다. 백엔드가 스스로 "기타" 로 분류하므로 결과는
# 같고, 로그에는 모델이 무엇이라고 불렀는지가 남아 다음 별칭을 여기에 더할 수 있습니다.
# "·" 는 구분자가 아닙니다. 허용 목록의 이름 자체가 "꽃·식물" 처럼 가운뎃점을
# 품고 있어, 여기서 자르면 이름 대조가 조각으로만 성립합니다.
_CATEGORY_SEPARATOR = re.compile(r"[/>|,\\]")

# 별칭표(이름 대조)가 놓친 값을 잡는 두 번째 그물입니다. 순서가 우선순위입니다.
# 한 글자 핵심어는 넣지 않습니다 — "티" 는 "티셔츠" 를, "립" 은 "드립백" 을,
# "차" 는 "자동차" 를 끌어옵니다.
_RECORD_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("꽃·식물", ("꽃", "플라워", "화분", "식물", "다육", "생화")),
    ("상품권", (
        "상품권", "금액권", "교환권", "이용권", "기프트카드", "기프티콘", "기프트콘",
        "바우처", "쿠폰", "포인트", "관람권", "e카드", "모바일교환권",
    )),
    ("디저트", (
        "디저트", "케이크", "쿠키", "베이커리", "제과", "초콜릿", "마카롱", "과일",
        "한과", "약과", "젤리", "아이스크림", "간식", "식품", "음식", "먹거리",
        "커피", "음료", "드립백", "티백", "녹차", "홍차", "주스", "에이드", "와인", "주류",
    )),
    ("패션·잡화", (
        "화장품", "뷰티", "향수", "스킨", "로션", "립밤", "립스틱", "메이크업", "핸드크림",
        "패션", "의류", "지갑", "가방", "파우치", "액세서리", "잡화", "주얼리", "시계",
        "셔츠", "니트", "양말", "신발", "모자", "머플러", "스카프",
    )),
    ("생활용품", (
        "생활", "리빙", "인테리어", "타월", "수건", "텀블러", "식기", "주방", "세제",
        "가전", "디지털", "충전", "케이블", "이어폰", "거치대", "도서", "문구",
        "완구", "장난감", "육아", "건강", "헬스",
    )),
)


def normalize_record_category(raw: str | None) -> str | None:
    """이미지에서 읽은 분류를 백엔드가 아는 이름으로 바꿉니다.

    "기프티콘/음료" 처럼 여러 층으로 적힌 값은 **뒤쪽(구체적인 쪽)부터** 봅니다.
    앞쪽을 먼저 보면 "기프티콘/음료" 가 음료가 아니라 상품권이 됩니다.

    Args:
        raw: VLM 이 쓴 분류. ``None`` 이나 빈 값이면 그대로 돌려줍니다.

    Returns:
        허용 목록의 이름, 또는 맞추지 못했으면 ``raw`` 원문 그대로.
    """
    if not raw or not raw.strip():
        return raw

    segments = [s for s in _CATEGORY_SEPARATOR.split(raw) if s.strip()]
    candidates = list(dict.fromkeys([*reversed(segments), raw]))

    for candidate in candidates:
        matched = canonical_category(candidate)
        if matched:
            return matched

    for candidate in candidates:
        compact = re.sub(r"\s+", "", candidate).lower()
        for name, keywords in _RECORD_CATEGORY_KEYWORDS:
            if any(keyword in compact for keyword in keywords):
                return name

    logger.info("기록 분류를 허용 목록에 맞추지 못해 원문을 그대로 내보냅니다: %s", raw)
    return raw


class GiftDataPolicyError(ValueError):
    """추출 결과로 유효한 ``GiftData`` 를 만들 수 없을 때 발생합니다."""


@dataclass
class GiftDataBuild:
    """``GiftData`` 와, 계약 밖으로 밀려난 정보들.

    Attributes:
        gift_data: 공개 API 로 나가는 결과.
        primary: 대표로 선택된 기록.
        dropped_records: 계약상 전달되지 못한 나머지 기록들.
        price_basis: ``gift_price`` 가 이미지에 적힌 값인지, 검색으로 찾은 값인지, 미상인지.
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


def _invitation_name(record: ExtractedRecord) -> str:
    """초대장 이름. 청첩장·부고장·그 밖의 초대장을 구분합니다.

    ``event`` 를 그대로 "{event} 청첩장" 에 끼우면 부고장이 "조의 청첩장" 이 되고,
    돌잔치 초대장이 "돌잔치 청첩장" 이 됩니다. 청첩장은 결혼에만 쓰는 말입니다.
    """
    event = (record.event or "").strip()
    if is_condolence(event, record.category, record.memo):
        return "부고장"
    if any(keyword in event for keyword in _WEDDING_KEYWORDS):
        return f"{event} 청첩장"
    return f"{event} 초대장" if event else "초대장"


def _money_name(record: ExtractedRecord) -> str:
    """현금 기록의 이름을 "계기 + 종류" 로 조립합니다.

    ``event`` 만 쓰면 선물 이름 자리에 "생일" 같은 계기가 그대로 들어가,
    답례 메시지가 "선물해 주신 생일 정말 고마웠어요" 가 됩니다.
    """
    event = (record.event or "").strip()
    category = (record.category or "").strip()
    if is_condolence(event, category, record.memo):
        kind = "조의금"
    elif matched := next((k for k in _MONEY_KINDS if k in category), ""):
        kind = matched
    elif any(keyword in event for keyword in _WEDDING_KEYWORDS):
        kind = "축의금"
    elif event:
        kind = "축하금"
    else:
        kind = "현금"

    if event and event not in kind and kind not in event:
        return f"{event} {kind}"
    return kind


def build_gift_name(record: ExtractedRecord) -> str:
    """``gift_name`` 에 넣을 사람이 읽을 수 있는 이름을 만듭니다."""
    if record.item_name:
        # 상품명에 이미 브랜드가 들어 있으면 덧붙이지 않습니다.
        # "TWG Tea" + "TWG Tea Teabags Collection" 이 그대로 이어붙던 문제입니다.
        brand = (record.brand or "").strip()
        has_brand = bool(brand) and brand.lower() in record.item_name.lower()
        name = f"{brand} {record.item_name}" if brand and not has_brand else record.item_name
    elif record.record_type is RecordType.EVENT_INVITATION:
        # 사용자가 하객이라는 사실은 GiftData.record_type 과 추천 프롬프트가 전달한다.
        # 여기에 설명을 덧붙이면 그대로 사용자 문장에 새어 나온다.
        name = _invitation_name(record)
    elif record.record_type is RecordType.MONEY:
        name = _money_name(record)
    elif record.category:
        name = record.category
    elif record.event:
        # 계기만 남으면 그것도 이름이 아닙니다. "생일" 이 아니라 "생일 선물" 입니다.
        name = f"{record.event} 선물"
    else:
        name = "받은 선물"

    return name.strip()[:_GIFT_NAME_MAX] or "받은 선물"


def to_record_item(record: ExtractedRecord, index: int) -> GiftRecordItem:
    """내부 추출 타입을 공개 계약의 기록 항목으로 옮깁니다.

    금액이 없는 청첩장도 ``price=None`` 으로 있는 그대로 표현됩니다.
    """
    return GiftRecordItem(
        record_id=f"r{index}",
        record_type=RecordKind(record.record_type.value),
        direction=RecordDirection(record.direction.value),
        person_name=(record.counterpart_name or None) and record.counterpart_name[:_PERSON_NAME_MAX],
        gift_name=build_gift_name(record),
        price=record.amount,
        price_basis=PriceBasis.STATED if record.amount is not None else PriceBasis.UNKNOWN,
        received_at=record.occurred_date,
        event_date=record.event_date,
        event=record.event,
        # 원문이 아니라 정규화한 이름을 내보냅니다. 내부에서 쓰는
        # ``record.category`` 는 원문 그대로라 ``build_gift_name`` 의 이름 대체가
        # "디저트" 같은 뭉뚱그린 말로 바뀌지 않습니다.
        category=normalize_record_category(record.category),
        brand=record.brand,
        memo=record.memo,
        confidence=record.confidence,
        needs_review=record.needs_review,
        review_reasons=list(record.review_reasons),
    )


def build_gift_data(result: ExtractionResult) -> GiftDataBuild:
    """추출 결과를 ``GiftData`` 로 변환합니다.

    평면 필드에는 대표 1건이 들어가고(기존 계약 유지), 읽어 낸 전체 기록은
    ``records`` 에 함께 담깁니다. 이를 모르는 코드는 평면 필드만 읽으면 됩니다.

    Args:
        result: ``vision_response_parser`` 가 정규화한 결과.

    Returns:
        ``GiftData`` 와 변환 과정의 부가 정보.

    Raises:
        GiftDataPolicyError: 기록이 하나도 없거나,
            ``strict_price`` 가 켜진 상태에서 금액을 읽지 못한 경우.
    """
    primary = select_primary(result.records)
    if primary is None:
        raise GiftDataPolicyError("이미지에서 선물·부조금 기록을 찾지 못했습니다.")

    warnings = list(result.warnings)
    dropped = [r for r in result.records if r is not primary]

    price_basis = PriceBasis.STATED
    gift_name = build_gift_name(primary)

    gift_price: int | None = None
    if primary.amount is not None:
        gift_price = min(primary.amount, _PRICE_MAX)
        # 이미지가 아니라 검색으로 채운 값이면 그 사실을 남깁니다. 사용자가 확인
        # 화면에서 고칠 수 있어야 하고, 답례 가격대가 이 값에서 나오기 때문입니다.
        if primary.price_searched:
            price_basis = PriceBasis.SEARCHED
            warnings.append(f"이미지에 금액이 없어 검색으로 {gift_price:,}원을 채웠습니다")
    elif settings.strict_price:
        raise GiftDataPolicyError(
            "이미지에서 금액을 읽지 못했습니다. 금액을 직접 입력해 주세요."
        )
    else:
        # 카테고리로 추정하지 않습니다. 브랜드를 모르는 추정가는 실제와 몇 배씩
        # 어긋나고(TWG Tea 를 10,000원으로 추정, 실제 3~7만원), 사용자는 그 값을
        # 사실로 받아들입니다. 모르는 것은 모르는 채로 두고 확인 화면에서 채웁니다.
        price_basis = PriceBasis.UNKNOWN
        warnings.append("금액을 확인하지 못했습니다. 사용자에게 입력받으세요")

    if primary.needs_review:
        warnings.append("사용자 확인 필요: " + ", ".join(primary.review_reasons))

    records = [to_record_item(r, i) for i, r in enumerate(result.records)]
    if len(records) > 1:
        warnings.append(f"이미지에서 {len(records)}건을 읽었습니다. 대표 1건은 {primary.counterpart_name or '이름 미상'}.")

    gift_data = GiftData(
        gift_name=gift_name,
        gift_price=gift_price,
        age=None,  # 이미지에서 나이를 알 수 있는 경우는 없습니다.
        person_name=(primary.counterpart_name or None) and primary.counterpart_name[:_PERSON_NAME_MAX],
        relationship=None,  # 관계는 백엔드가 인물 DB 에서 채우는 값입니다.
        received_at=primary.occurred_date,
        target_date=primary.event_date,
        records=records,
        record_type=RecordKind(primary.record_type.value),
        direction=RecordDirection(primary.direction.value),
        price_basis=price_basis,
        event=primary.event,
        event_date=primary.event_date,
        confidence=primary.confidence,
        needs_review=any(r.needs_review for r in records),
        review_reasons=list(primary.review_reasons),
    )

    return GiftDataBuild(
        gift_data=gift_data,
        primary=primary,
        dropped_records=dropped,
        price_basis=price_basis,
        warnings=warnings,
    )
