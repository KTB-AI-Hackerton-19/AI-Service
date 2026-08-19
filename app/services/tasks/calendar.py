"""Google MCP 캘린더 등록 요청 데이터를 준비하는 작업."""

import logging

from app.core.config import settings
from app.schemas.agent import CalendarDraft, GiftData, PreparedData
from app.services import record_summary
from app.services.calendar_mcp_client import CalendarMcpError, calendar_mcp_client
from app.services.reciprocity_schedule import ReciprocitySchedule, resolve_schedule

logger = logging.getLogger(__name__)

PROVIDER_REGISTERED = "GOOGLE_MCP"
PROVIDER_DRAFT = "GOOGLE_MCP_DRAFT"
_DURATION_MINUTES = 30
_REMINDERS_MINUTES = [0, 24 * 60]  # 시작 정각과 하루 전


def build_draft(gift_data: GiftData, workflow_id: str) -> CalendarDraft:
    """선물데이터로 캘린더 일정 초안을 만듭니다.

    여러 건이면 일정을 여러 개 만들지 않고 하나로 묶습니다. 축의금 4건을 받았다고
    캘린더에 4개가 뜨면 오히려 방해가 되므로, 대상자 명단은 설명에 담습니다.

    Args:
        gift_data: 일정 제목과 날짜를 계산할 공통 선물 정보.
        workflow_id: 네 작업의 결과를 연결하는 요청 추적 ID.

    Returns:
        아직 등록되지 않은 일정 초안.
    """
    schedule = resolve_schedule(gift_data)
    who = record_summary.people_label(gift_data)

    lines = [f"{record_summary.headline(gift_data)}에 대한 답례를 준비할 시간입니다."]
    lines.extend(record_summary.detail_lines(gift_data))
    if gift_data.received_at and not record_summary.is_multi(gift_data):
        lines.append(f"받은 날: {gift_data.received_at.isoformat()}")
    if gift_data.relationship:
        lines.append(f"관계: {gift_data.relationship}")
    lines.append(f"답례 예정일: {schedule.target_date.isoformat()}")

    return CalendarDraft(
        provider=PROVIDER_DRAFT,
        registered=False,
        workflow_id=workflow_id,
        title=f"{who} 답례 준비",
        description="\n".join(lines),
        scheduled_date=schedule.prepare_date,
        start_time=schedule.calendar_start_time,
        duration_minutes=_DURATION_MINUTES,
        timezone=settings.default_timezone,
        reminders_minutes=list(_REMINDERS_MINUTES),
        calendar_id=settings.google_calendar_id,
        target_date=schedule.target_date,
    )


async def register(draft: CalendarDraft, access_token: str) -> CalendarDraft:
    """초안을 실제 Google Calendar 일정으로 등록합니다.

    실패해도 예외를 밖으로 던지지 않습니다. 캘린더가 막혀도 나머지 결과는 살아야 하고,
    사용자에게는 초안과 실패 사유를 함께 보여 주는 편이 낫기 때문입니다.

    Args:
        draft: 등록할 일정 초안.
        access_token: 사용자 Google OAuth access token.

    Returns:
        등록 결과가 반영된 초안. 성공하면 ``registered=True`` 와 ``event_id`` 가 채워집니다.
    """
    try:
        event = await calendar_mcp_client.create_event(
            access_token=access_token,
            summary=draft.title,
            description=draft.description,
            start_date=draft.scheduled_date.isoformat(),
            start_time=draft.start_time,
            duration_minutes=draft.duration_minutes,
            reminders_minutes=draft.reminders_minutes,
            calendar_id=draft.calendar_id,
        )
    except CalendarMcpError as exc:
        logger.warning("캘린더 등록 실패 workflow=%s: %s", draft.workflow_id, exc)
        return draft.model_copy(update={"registered": False, "register_error": str(exc)})

    logger.info(
        "캘린더 등록 완료 workflow=%s event=%s date=%s",
        draft.workflow_id,
        event.get("event_id"),
        draft.scheduled_date,
    )
    return draft.model_copy(
        update={
            "provider": PROVIDER_REGISTERED,
            "registered": True,
            "event_id": event.get("event_id"),
            "html_link": event.get("html_link"),
            "register_error": None,
        }
    )


class CalendarPreparationService:
    """답례 준비 일정을 만드는 서비스.

    기본 동작은 **초안 생성까지**입니다. 실제 등록은 사용자가 확인 화면에서 승인한 뒤
    ``/api/v1/agent/confirm`` 에서 일어납니다. 잘못 추출된 일정이 사용자 캘린더에
    바로 박히면 되돌리기 어렵기 때문입니다.

    승인 화면이 아직 없는 개발 단계에서는 ``CALENDAR_AUTO_REGISTER=true`` 로
    즉시 등록시켜 흐름을 확인할 수 있습니다.
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
            등록용 초안(또는 자동 등록 결과)을 담은 ``PreparedData``.
        """
        draft = build_draft(gift_data, workflow_id)

        if settings.calendar_auto_register and settings.google_access_token:
            draft = await register(draft, settings.google_access_token)
        else:
            logger.info(
                "캘린더 초안 생성 workflow=%s date=%s (승인 후 등록 대기)",
                workflow_id,
                draft.scheduled_date,
            )

        return PreparedData(payload=draft.to_payload())

    @staticmethod
    def build_draft(gift_data: GiftData, workflow_id: str) -> CalendarDraft:
        """확정 단계에서 초안을 다시 계산할 때 씁니다."""
        return build_draft(gift_data, workflow_id)


calendar_preparation_service = CalendarPreparationService()
