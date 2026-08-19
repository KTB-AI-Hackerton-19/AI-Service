"""VLM 원시 출력을 정규화된 기록 목록으로 바꿉니다.

``model_response_parser`` 가 텍스트에서 JSON 을 꺼내는 역할이라면,
이 모듈은 꺼낸 JSON 의 값들을 실제로 믿을 수 있는 형태로 바로잡습니다.

여기가 정확도를 실제로 좌우하는 지점이라 vLLM 없이 단위 테스트할 수 있도록
서비스 클래스가 아니라 순수 함수로 두었습니다.
"""

import logging
import re
from datetime import date
from typing import Any

from app.schemas.vision import Direction, ExtractedRecord, ExtractionResult, RecordType

logger = logging.getLogger(__name__)

# 영수증에서 상품이 아닌 줄. 사전 실측에서 할인·합계 줄이 금액으로 올라오는 사례가 있었습니다.
_NON_ITEM_PATTERN = re.compile(
    r"(할인|에누리|적립|포인트|쿠폰|부가세|과세|면세|합계|소계|총액|총\s*금액|"
    r"받을\s*금액|거스름|잔액|현금영수증|승인번호)"
)
_DIGITS_PATTERN = re.compile(r"-?\d+")
_NULLISH = {"", "null", "none", "n/a", "-", "unknown", "미상", "없음"}

# 화면에 "김수현 님"으로 보이면 모델도 그대로 옮겨 적습니다. 백엔드가 인물을 매칭할 때
# "김수현"과 "김수현 님"이 다른 사람이 되므로 호칭을 떼어 냅니다.
_HONORIFIC_SUFFIX = re.compile(r"(?P<space>\s*)(?P<title>선생님|고객님|선배|후배|님|씨|군|양)$")
# 호칭을 뗀 뒤 이름으로 인정할 최소 길이. "김선배"에서 "선배"를 떼면 "김"만 남는데
# 이건 호칭이 아니라 이름의 일부입니다.
_MIN_NAME_LENGTH = 2

MIN_CONFIDENCE = 0.6

# 날짜 구분자는 ``.`` ``-`` ``/`` ``년월일`` 뿐입니다. ``:`` 와 ``,`` 를 허용하면
# "오전 10:30" 이 10월 30일로, "12,300원" 이 12월 30일로 둔갑합니다(실측).
_YMD_PATTERN = re.compile(r"(\d{4})\D{0,3}(\d{1,2})\D{0,3}(\d{1,2})")
_YY_MD_PATTERN = re.compile(r"(?<!\d)(\d{2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})(?!\d)")
_MD_PATTERN = re.compile(r"(?<![\d:])(\d{1,2})\s*[.\-/]\s*(\d{1,2})(?![\d:])")
_MD_KOREAN_PATTERN = re.compile(r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일?")


def parse_date_value(value: Any, default_year: int) -> date | None:
    """모델이 내놓는 여러 날짜 표기를 ``date`` 로 바꿉니다.

    연도가 없는 표기는 날짜 구분자(``.`` ``-`` ``/`` ``월``)가 있을 때만 날짜로 봅니다.
    시각("오전 10:30")과 금액("12,300원")이 날짜로 둔갑해 캘린더·알림까지 흘러가던
    문제를 여기서 막습니다.

    Args:
        value: ``2026-03-14``, ``2026.3.14``, ``26.03.14``, ``3월 14일`` 등.
        default_year: 연도가 없을 때 사용할 연도.

    Returns:
        해석에 성공하면 ``date``, 실패하면 ``None``.
    """
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if text.lower() in _NULLISH:
        return None

    match = _YMD_PATTERN.search(text)
    if match:
        return _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    match = _YY_MD_PATTERN.search(text)
    if match:
        return _safe_date(2000 + int(match.group(1)), int(match.group(2)), int(match.group(3)))

    match = _MD_KOREAN_PATTERN.search(text) or _MD_PATTERN.search(text)
    if match:
        return _safe_date(default_year, int(match.group(1)), int(match.group(2)))
    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    """존재하지 않는 날짜를 ``None`` 으로 흘려보냅니다."""
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_amount_value(value: Any) -> int | None:
    """금액을 원화 정수로 바꿉니다. 0 이하는 ``None`` 입니다.

    할인 줄의 음수 금액이 선물 금액으로 올라오는 것을 막기 위해
    양수만 유효한 금액으로 인정합니다.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if value > 0 else None
    if not isinstance(value, str):
        return None

    match = _DIGITS_PATTERN.search(value.replace(",", ""))
    if not match:
        return None
    amount = int(match.group())
    return amount if amount > 0 else None


def _clean_text(value: Any) -> str | None:
    """공백과 빈 값 표기를 ``None`` 으로 정리합니다."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return None if text.lower() in _NULLISH else text


def clean_person_name(value: Any) -> str | None:
    """사람 이름에서 호칭을 떼어 냅니다. "김수현 님" -> "김수현".

    붙여 쓴 호칭은 이름의 마지막 글자와 구분되지 않습니다. "김선배"에서 "선배"를
    떼면 "김"만 남는데 이건 호칭이 아니라 이름입니다. 그래서 공백이 없을 때는
    남는 이름이 두 글자 이상일 때만 뗍니다.
    """
    text = _clean_text(value)
    if not text:
        return None

    match = _HONORIFIC_SUFFIX.search(text)
    if match is None:
        return text

    stripped = text[: match.start()].strip()
    if not stripped:
        # 호칭만 있는 경우("님")는 이름이 없는 것으로 봅니다.
        return None
    if not match.group("space") and len(stripped) < _MIN_NAME_LENGTH:
        return text
    return stripped


def _coerce_enum(value: Any, enum_cls, fallback):
    """모델이 열거형 밖의 값을 내놓으면 기본값으로 되돌립니다."""
    try:
        return enum_cls(value)
    except ValueError:
        return fallback


def _is_receipt_noise(raw: dict) -> bool:
    """영수증의 할인·합계처럼 상품이 아닌 줄인지 판단합니다.

    상품명만 봅니다. 메모·카테고리까지 합쳐서 보면 "할인받아서 샀어" 같은 정상
    선물의 메모 한 줄 때문에 기록이 통째로 사라집니다.
    """
    return bool(_NON_ITEM_PATTERN.search(str(raw.get("item_name") or "")))


def _is_invitation_account_row(record_type: RecordType, amount: int | None, image_kind: str) -> bool:
    """청첩장·부고장에 적힌 계좌 안내인지 판단합니다.

    실측에서 청첩장의 "신랑측 국민 123-45-678901" 같은 줄이 record_type=money 로 올라왔습니다.
    이건 내가 받은 돈이 아니라 앞으로 보낼 계좌 안내이므로 기록이 아닙니다.
    금액이 적혀 있지 않다는 점으로 실제 송금 기록과 구분됩니다.
    """
    return image_kind == "invitation" and record_type is RecordType.MONEY and amount is None


def _infer_direction(record_type: RecordType, raw_direction: Any) -> Direction:
    """방향이 비어 있을 때 기록 종류로 보완합니다."""
    direction = _coerce_enum(raw_direction, Direction, Direction.UNKNOWN)
    if direction is not Direction.UNKNOWN:
        return direction
    if record_type is RecordType.RECEIPT:
        return Direction.SENT  # 영수증은 내가 구매한 것
    return Direction.UNKNOWN


def flag_review(record: ExtractedRecord, today: date) -> None:
    """사용자 확인이 필요한 항목에 사유를 붙입니다.

    사유 문구는 확인 화면에 그대로 보이므로, 사용자가 무엇을 하면 되는지가 드러나는
    말로 씁니다. 금액을 검색으로 채우는 등 값이 바뀐 뒤에 다시 부를 수 있도록
    이전 사유를 남기지 않고 매번 새로 계산합니다.
    """
    reasons: list[str] = []
    if record.confidence < MIN_CONFIDENCE:
        reasons.append("이미지에서 내용을 또렷하게 읽지 못했습니다. 확인해 주세요")
    if not record.counterpart_name and record.record_type is not RecordType.RECEIPT:
        reasons.append("상대방 이름을 확인하지 못했습니다. 직접 입력해 주세요")
    if record.record_type is RecordType.MONEY and record.amount is None:
        reasons.append("금액을 확인하지 못했습니다. 직접 입력해 주세요")
    if record.occurred_date is None and record.event_date is None:
        reasons.append("날짜를 확인하지 못했습니다. 직접 입력해 주세요")
    if record.occurred_date and record.occurred_date > today:
        reasons.append("주고받은 날짜가 오늘 이후로 적혀 있습니다. 날짜를 확인해 주세요")
    if record.price_searched:
        # 검색으로 찾은 값은 같은 상품의 다른 용량·구성일 수 있습니다.
        # 이 금액이 답례 가격대의 기준이 되므로 반드시 눈으로 확인받습니다.
        reasons.append("이미지에 금액이 없어 상품 검색으로 채운 금액입니다. 맞는지 확인해 주세요")

    record.review_reasons = reasons
    record.needs_review = bool(reasons)


def refresh_review_flags(result: ExtractionResult, today: date) -> None:
    """금액을 채우는 등 값이 바뀐 뒤 확인 사유를 다시 계산합니다."""
    for record in result.records:
        flag_review(record, today)


def parse_extraction(payload: dict, today: date) -> ExtractionResult:
    """VLM 원시 출력을 정규화된 ``ExtractionResult`` 로 바꿉니다.

    Args:
        payload: ``image_kind`` 와 ``records`` 를 담은 VLM 출력.
        today: 미래 날짜 판정과 기본 연도의 기준이 되는 날짜.

    Returns:
        정규화·중복 제거를 마친 결과.
    """
    warnings: list[str] = []
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        return ExtractionResult(warnings=["VLM 응답에 records 배열이 없습니다"])

    image_kind = str(payload.get("image_kind") or "other")
    records: list[ExtractedRecord] = []

    for raw in raw_records:
        if not isinstance(raw, dict):
            continue

        record_type = _coerce_enum(raw.get("record_type"), RecordType, RecordType.UNKNOWN)
        if (record_type is RecordType.RECEIPT or image_kind == "receipt") and _is_receipt_noise(raw):
            continue

        amount = parse_amount_value(raw.get("amount"))
        if _is_invitation_account_row(record_type, amount, image_kind):
            continue

        occurred = parse_date_value(raw.get("occurred_date"), today.year)
        event_date = parse_date_value(raw.get("event_date"), today.year)

        # 청첩장인데 날짜가 미래라면 그건 수령일이 아니라 예식일입니다.
        if record_type is RecordType.EVENT_INVITATION and event_date is None and occurred and occurred > today:
            occurred, event_date = None, occurred

        try:
            confidence = min(1.0, max(0.0, float(raw.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0

        record = ExtractedRecord(
            record_type=record_type,
            direction=_infer_direction(record_type, raw.get("direction")),
            counterpart_name=clean_person_name(raw.get("counterpart_name")),
            occurred_date=occurred,
            event_date=event_date,
            item_name=_clean_text(raw.get("item_name")),
            brand=_clean_text(raw.get("brand")),
            category=_clean_text(raw.get("category")),
            event=_clean_text(raw.get("event")),
            amount=amount,
            memo=_clean_text(raw.get("memo")),
            confidence=confidence,
        )
        flag_review(record, today)
        records.append(record)

    records = deduplicate(records)
    if not records:
        warnings.append("이미지에서 선물·부조금 기록을 찾지 못했습니다")

    return ExtractionResult(image_kind=image_kind, records=records, warnings=warnings)


def deduplicate(records: list[ExtractedRecord]) -> list[ExtractedRecord]:
    """같은 건이 중복 추출된 경우 신뢰도가 높은 쪽만 남깁니다."""
    best: dict[tuple, ExtractedRecord] = {}
    order: list[tuple] = []

    for record in records:
        key = (
            (record.counterpart_name or "").replace(" ", ""),
            record.occurred_date,
            record.amount,
            (record.item_name or "").replace(" ", ""),
        )
        if key not in best:
            best[key] = record
            order.append(key)
        elif record.confidence > best[key].confidence:
            best[key] = record

    return [best[key] for key in order]
