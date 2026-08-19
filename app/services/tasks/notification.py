"""답례 알림 예약 요청 데이터를 준비하는 작업."""

import logging

from app.core.config import settings
from app.schemas.agent import GiftData, PreparedData
from app.services.reciprocity_schedule import resolve_schedule

logger = logging.getLogger(__name__)

_TYPE_PREPARE = "RECIPROCITY_PREPARE"
_CHANNEL_WEB = "WEB"


class NotificationPreparationService:
    """답례 준비 알림 예약 JSON 을 만드는 서비스.

    알림 시각은 캘린더 일정과 같은 규칙(``reciprocity_schedule``)에서 나옵니다.
    두 작업이 서로 다른 날짜를 쓰면 사용자에게는 그냥 버그로 보이기 때문입니다.
    """

    async def prepare(
        self,
        gift_data: GiftData,
        workflow_id: str,
    ) -> PreparedData:
        """선물데이터로 알림 예약용 JSON을 준비합니다.

        Args:
            gift_data: 알림 내용과 시각을 계산할 공통 선물 정보.
            workflow_id: 네 작업의 결과를 연결하는 요청 추적 ID.

        Returns:
            알림 시스템에 전달할 JSON을 담은 ``PreparedData``.
        """
        schedule = resolve_schedule(gift_data)
        person = gift_data.person_name or "상대방"
        scheduled_at = schedule.notify_at.isoformat(timespec="seconds")

        notification = {
            "type": _TYPE_PREPARE,
            "channel": _CHANNEL_WEB,
            "title": "답례 선물을 준비할 시간이에요",
            "body": (
                f"{person}님에게 받은 {gift_data.gift_name}, 기억하고 계시죠? "
                f"{schedule.target_date.isoformat()}까지 답례를 준비해 보세요."
            ),
            "scheduledAt": scheduled_at,
            "deepLink": f"/records/{workflow_id}",
        }

        payload = {
            "workflowId": workflow_id,
            "timezone": settings.default_timezone,
            "notifications": [notification],
            # 기존 계약을 읽는 쪽이 있을 수 있어 최상위 title/scheduledAt 도 유지합니다.
            "title": notification["title"],
            "scheduledAt": scheduled_at,
        }

        logger.info("알림 예약 준비 workflow=%s at=%s", workflow_id, scheduled_at)
        return PreparedData(payload=payload)


notification_preparation_service = NotificationPreparationService()
