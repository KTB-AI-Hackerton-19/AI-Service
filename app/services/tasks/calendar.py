"""Google MCP 캘린더 등록 요청 데이터를 준비하는 작업."""

import logging

from app.core.config import settings
from app.schemas.agent import GiftData, PreparedData
from app.services.calendar_mcp_client import CalendarMcpError, calendar_mcp_client
from app.services.reciprocity_schedule import ReciprocitySchedule, resolve_schedule

logger = logging.getLogger(__name__)

_PROVIDER_REGISTERED = "GOOGLE_MCP"
_PROVIDER_DRAFT = "GOOGLE_MCP_DRAFT"
_DURATION_MINUTES = 30
_REMINDERS_MINUTES = [0, 24 * 60]  # 시작 정각과 하루 전


class CalendarPreparationService:
    """답례 준비 일정을 만들고, 토큰이 있으면 MCP 로 실제 등록까지 하는 서비스.

    동작이 두 가지로 갈립니다.

    - ``GOOGLE_ACCESS_TOKEN`` 이 있으면 MCP 서버를 통해 Google Calendar 에 실제로 등록하고
      ``eventId`` 와 ``htmlLink`` 를 payload 에 담습니다.
    - 토큰이 없으면 등록하지 않고 초안 JSON 만 만듭니다. 백엔드가 사용자 토큰으로
      직접 등록하거나, 사용자 승인 뒤에 등록할 때 이 초안을 그대로 쓰면 됩니다.

    어느 경우든 입력 ``GiftData`` 와 출력 ``PreparedData`` 계약은 같습니다.
    """

    async def prepare(
        self,
        gift_data: GiftData,
        workflow_id: str,
    ) -> PreparedData:
        """선물데이터로 캘린더 등록용 JSON을 준비합니다.

        Args:
            gift_data: 일정 제목과 날짜를 계산할 공통 선물 정보.
            workflow_id: 네 작업의 결과를 연결하는 요청 추적 ID.

        Returns:
            Google MCP 호출 결과 또는 등록용 초안을 담은 ``PreparedData``.
        """
        schedule = resolve_schedule(gift_data)
        person = gift_data.person_name or "상대방"
        draft = self._build_draft(gift_data, schedule, person, workflow_id)

        if not settings.google_access_token:
            logger.info(
                "캘린더 초안만 생성 workflow=%s date=%s (GOOGLE_ACCESS_TOKEN 미설정)",
                workflow_id,
                schedule.prepare_date,
            )
            return PreparedData(payload=draft)

        try:
            event = await calendar_mcp_client.create_event(
                access_token=settings.google_access_token,
                summary=draft["title"],
                description=draft["description"],
                start_date=draft["date"],
                start_time=draft["startTime"],
                duration_minutes=_DURATION_MINUTES,
                reminders_minutes=_REMINDERS_MINUTES,
            )
        except CalendarMcpError as exc:
            # 캘린더가 막혀도 나머지 세 작업 결과는 살아야 하므로 예외를 밖으로 던지지 않습니다.
            logger.warning("캘린더 등록 실패 workflow=%s: %s", workflow_id, exc)
            return PreparedData(payload={**draft, "registered": False, "registerError": str(exc)})

        logger.info(
            "캘린더 등록 완료 workflow=%s event=%s date=%s",
            workflow_id,
            event.get("event_id"),
            draft["date"],
        )
        return PreparedData(
            payload={
                **draft,
                "provider": _PROVIDER_REGISTERED,
                "registered": True,
                "eventId": event.get("event_id"),
                "htmlLink": event.get("html_link"),
            }
        )

    @staticmethod
    def _build_draft(
        gift_data: GiftData,
        schedule: ReciprocitySchedule,
        person: str,
        workflow_id: str,
    ) -> dict:
        """등록 여부와 무관하게 동일한 형태의 일정 초안을 만듭니다."""
        description_lines = [
            f"{person}님에게 받은 {gift_data.gift_name} ({gift_data.gift_price:,}원)에 대한 답례를 준비할 시간입니다.",
        ]
        if gift_data.received_at:
            description_lines.append(f"받은 날: {gift_data.received_at.isoformat()}")
        if gift_data.relationship:
            description_lines.append(f"관계: {gift_data.relationship}")
        description_lines.append(f"답례 예정일: {schedule.target_date.isoformat()}")

        return {
            "provider": _PROVIDER_DRAFT,
            "registered": False,
            "workflowId": workflow_id,
            "title": f"{person}님 답례 준비",
            "description": "\n".join(description_lines),
            "date": schedule.prepare_date.isoformat(),
            "startTime": schedule.calendar_start_time,
            "durationMinutes": _DURATION_MINUTES,
            "timezone": settings.default_timezone,
            "remindersMinutes": _REMINDERS_MINUTES,
            "calendarId": settings.google_calendar_id,
            "targetDate": schedule.target_date.isoformat(),
        }


calendar_preparation_service = CalendarPreparationService()
