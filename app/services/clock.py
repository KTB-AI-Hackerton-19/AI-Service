"""서비스 기준 시각을 한곳에서 만듭니다.

``datetime.now()`` 를 그대로 쓰면 컨테이너의 타임존을 따라갑니다. Dockerfile 의 베이스
이미지(`nvidia/cuda:...ubuntu22.04`)는 기본이 UTC 라서, 서버에서 돌리면 KST 로 계산했다고
믿은 값이 9시간 어긋납니다. 캘린더 일정 시각과 알림 시각이 전부 틀어집니다.

그래서 항상 ``settings.default_timezone`` 기준의 벽시계 시각을 씁니다.
Google Calendar 에는 ``timeZone`` 필드를 따로 보내므로 naive 로 돌려줘도 모호하지 않습니다.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

from app.core.config import settings


def service_timezone() -> ZoneInfo:
    """설정된 서비스 타임존."""
    return ZoneInfo(settings.default_timezone)


def service_now() -> datetime:
    """서비스 타임존 기준 현재 시각(naive 벽시계).

    Returns:
        타임존 정보가 없는 ``datetime``. 값 자체는 설정된 지역 시각입니다.
    """
    return datetime.now(service_timezone()).replace(tzinfo=None)


def service_today() -> date:
    """서비스 타임존 기준 오늘 날짜."""
    return service_now().date()
