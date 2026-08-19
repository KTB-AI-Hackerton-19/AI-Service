"""사용자 승인 이후의 확정 처리를 담당합니다.

왜 별도 단계인가
- 잘못 추출된 일정이 사용자 캘린더에 바로 박히면 되돌리기 어렵습니다.
- 이미지 한 장에 여러 건이 있으면(거래내역 5건 등) 무엇을 저장할지 사람이 골라야 합니다.

왜 상태를 보관하지 않는가
- AI 서비스가 세션을 들고 있으면 재시작이나 인스턴스 증설에서 그대로 깨집니다.
- 확정에 필요한 데이터는 어차피 백엔드가 DB 에 저장할 것들입니다.
  백엔드가 ``/from-image`` 응답을 들고 있다가 사용자 수정본과 함께 되돌려주면 됩니다.
"""

import logging

from app.core.config import settings
from app.schemas.agent import (
    ConfirmRequest,
    ConfirmResponse,
    PreparedData,
    TaskStatus,
)
from app.services.tasks import calendar as calendar_task
from app.services.tasks.gift_record import build_payload as build_gift_record_payload
from app.services.tasks.notification import build_payload as build_notification_payload

logger = logging.getLogger(__name__)


class ConfirmationService:
    """사용자가 검토·수정한 결과를 확정하고 캘린더에 등록합니다."""

    async def confirm(self, request: ConfirmRequest) -> ConfirmResponse:
        """확정 요청을 처리합니다.

        Args:
            request: 사용자 수정본이 반영된 확정 요청.

        Returns:
            확정된 기록·캘린더·알림 결과.
        """
        gift_data = request.gift_data
        workflow_id = request.workflow_id

        # 사용자가 고친 값이 그대로 반영되도록 기록과 알림을 다시 계산합니다.
        gift_record = PreparedData(payload=build_gift_record_payload(gift_data, workflow_id))
        notification = PreparedData(payload=build_notification_payload(gift_data, workflow_id))

        if not request.approved:
            logger.info("확정 거부 workflow=%s — 아무것도 등록하지 않습니다", workflow_id)
            return ConfirmResponse(
                workflow_id=workflow_id,
                approved=False,
                gift_data=gift_record,
                calendar_info=PreparedData(
                    payload={"registered": False, "workflowId": workflow_id, "reason": "승인하지 않으셔서 캘린더에 등록하지 않았습니다."}
                ),
                noti_info=notification,
            )

        # 사용자가 일정을 수정해 보냈으면 그대로, 아니면 수정된 기록으로 다시 계산합니다.
        draft = request.calendar or calendar_task.build_draft(gift_data, workflow_id)
        if request.calendar_id:
            draft = draft.model_copy(update={"calendar_id": request.calendar_id})
        draft = draft.model_copy(update={"workflow_id": workflow_id})

        if not request.register_calendar:
            logger.info("캘린더 등록 생략 workflow=%s (register_calendar=false)", workflow_id)
            return ConfirmResponse(
                workflow_id=workflow_id,
                approved=True,
                gift_data=gift_record,
                calendar_info=PreparedData(payload=draft.to_payload()),
                noti_info=notification,
            )

        token = request.google_access_token or settings.google_access_token
        if not token:
            message = "Google access token 이 없어 캘린더에 등록하지 못했습니다."
            logger.warning("%s workflow=%s", message, workflow_id)
            registered = draft.model_copy(update={"registered": False, "register_error": message})
        else:
            registered = await calendar_task.register(draft, token)

        calendar_info = PreparedData(
            status=TaskStatus.SUCCESS,
            payload=registered.to_payload(),
        )
        return ConfirmResponse(
            workflow_id=workflow_id,
            approved=True,
            gift_data=gift_record,
            calendar_info=calendar_info,
            noti_info=notification,
        )


confirmation_service = ConfirmationService()
