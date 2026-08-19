"""여러 건의 기록을 사람이 읽는 문구로 요약합니다.

선물 기록·캘린더·알림 세 작업이 같은 문구 규칙을 써야 합니다.
캘린더에는 "김도윤님 외 3명"이라고 적혀 있는데 알림에는 "최은비님"이라고 오면
사용자에게는 그냥 버그로 보입니다.
"""

from app.schemas.agent import GiftData, GiftRecordItem, RecordDirection

_ANONYMOUS = "상대방"


def received_records(gift_data: GiftData) -> list[GiftRecordItem]:
    """사용자가 선택해 둔 기록 중 "받은 것"만 돌려줍니다.

    영수증처럼 내가 보낸(sent) 항목은 답례 대상이 아니므로 제외합니다.
    """
    return [r for r in gift_data.selected_records if r.direction is not RecordDirection.SENT]


def people_label(gift_data: GiftData) -> str:
    """대상자를 한 구절로. "김도윤님 외 3명" 처럼 만듭니다."""
    names: list[str] = []
    for record in received_records(gift_data):
        if record.person_name and record.person_name not in names:
            names.append(record.person_name)

    if not names:
        return f"{gift_data.person_name}님" if gift_data.person_name else _ANONYMOUS
    if len(names) == 1:
        return f"{names[0]}님"
    return f"{names[0]}님 외 {len(names) - 1}명"


def total_amount(gift_data: GiftData) -> int | None:
    """받은 기록의 금액 합계. 어디에도 금액이 없으면 ``None`` 입니다."""
    records = received_records(gift_data)
    if not records:
        return gift_data.gift_price
    total = sum(r.price for r in records if r.price)
    return total or gift_data.gift_price


def is_multi(gift_data: GiftData) -> bool:
    """받은 기록이 두 건 이상인지."""
    return len(received_records(gift_data)) > 1


def headline(gift_data: GiftData) -> str:
    """한 줄 요약. "김도윤님 외 3명에게 받은 축의금 (총 400,000원)" 형태입니다."""
    who = people_label(gift_data)
    amount = total_amount(gift_data)
    what = _common_label(gift_data) or "마음" if is_multi(gift_data) else gift_data.gift_name
    # 금액을 모르면 괄호를 아예 붙이지 않습니다. "(0원)" 같은 표기는 사실과 다릅니다.
    if amount is None:
        return f"{who}에게 받은 {what}"
    total = "총 " if is_multi(gift_data) else ""
    return f"{who}에게 받은 {what} ({total}{amount:,}원)"


def detail_lines(gift_data: GiftData) -> list[str]:
    """건별 상세. 캘린더 설명에 그대로 넣습니다."""
    records = received_records(gift_data)
    if len(records) <= 1:
        return []

    lines = []
    for record in records:
        parts = [f"- {record.person_name or '이름 미상'}"]
        if record.price:
            parts.append(f"{record.price:,}원")
        elif record.gift_name:
            parts.append(record.gift_name)
        if record.received_at:
            parts.append(f"({record.received_at.isoformat()})")
        lines.append(" ".join(parts))
    return lines


def _common_label(gift_data: GiftData) -> str | None:
    """여러 건이 같은 계기·분류면 그 이름을 씁니다. "축의금" 처럼."""
    records = received_records(gift_data)
    for attr in ("category", "event"):
        values = {getattr(r, attr) for r in records if getattr(r, attr)}
        if len(values) == 1:
            return values.pop()
    return None
