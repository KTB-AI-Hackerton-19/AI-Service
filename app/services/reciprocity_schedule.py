"""답례 일정과 알림 시각을 계산합니다.

캘린더 작업과 알림 작업이 같은 날짜를 근거로 움직여야 하므로 규칙을 한곳에 모았습니다.
날짜 계산을 언어모델에 맡기지 않는 이유는 두 가지입니다.

- 날짜 산술은 규칙으로 정확히 나오는 값입니다. 모델에 맡기면 정확도가 확률이 되고,
  그 대가로 출력 토큰과 지연만 늘어납니다.
- 캘린더에 잘못된 날짜가 등록되면 사용자가 직접 지워야 합니다.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from app.core.config import settings
from app.schemas.agent import GiftData
from app.services.clock import service_now

_NOTIFICATION_HOUR = 10
_CALENDAR_HOUR = "10:00"


@dataclass(frozen=True)
class ReciprocitySchedule:
    """답례 준비일과 그에 딸린 알림 시각."""

    target_date: date
    """답례를 실행할 날짜. GiftData.target_date 가 있으면 그 값."""

    prepare_date: date
    """준비를 시작할 날짜. 답례일에서 notification_lead_days 만큼 앞."""

    notify_at: datetime
    """알림을 보낼 시각. 준비일 오전 10시. 이미 지난 시각이면 다음 정시로 밀린다."""

    calendar_start_time: str = _CALENDAR_HOUR
    """캘린더 일정 시작 시각 HH:MM. notify_at 과 항상 같다."""

    is_target_estimated: bool = False
    """target_date 가 입력값이 아니라 기본 규칙으로 만들어졌는지."""


def resolve_schedule(
    gift_data: GiftData,
    today: date | None = None,
    now: datetime | None = None,
) -> ReciprocitySchedule:
    """선물데이터에서 답례 일정을 계산합니다.

    Args:
        gift_data: 받은 날짜와 답례 예정일이 들어 있는 공통 입력.
        today: 기준 날짜. ``now`` 를 주면 무시됩니다.
        now: 기준 시각. 테스트에서 고정할 때 사용하며 기본값은 현재 시각.
            ``today`` 만 주면 그날 자정을 기준 시각으로 봅니다.

    Returns:
        답례일, 준비일, 알림 시각.
    """
    if now is None:
        now = datetime.combine(today, time(0, 0)) if today else service_now()
    today = now.date()

    target = gift_data.target_date
    is_estimated = target is None
    if target is None:
        base = gift_data.received_at or today
        target = base + timedelta(days=settings.calendar_default_lead_days)

    # 이미 지난 날짜로 일정을 잡으면 알림이 울리지 않습니다.
    if target <= today:
        target = today + timedelta(days=1)
        is_estimated = True

    prepare = target - timedelta(days=settings.notification_lead_days)
    if prepare < today:
        prepare = today

    notify_at = datetime.combine(prepare, time(_NOTIFICATION_HOUR))
    if notify_at <= now:
        # 준비일이 오늘인데 오전 10시가 이미 지난 경우다. 그대로 두면 알림이 울리지 않으므로
        # 다음 정시로 미룬다. 답례일 당일이라도 최소 한 번은 알림이 가야 한다.
        notify_at = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        prepare = notify_at.date()

    return ReciprocitySchedule(
        target_date=target,
        prepare_date=prepare,
        notify_at=notify_at,
        calendar_start_time=notify_at.strftime("%H:%M"),
        is_target_estimated=is_estimated,
    )
