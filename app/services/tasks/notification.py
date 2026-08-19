"""답례 알림 예약 요청 데이터를 준비하는 작업."""

import asyncio
from datetime import date, datetime, time, timedelta

from app.schemas.agent import HeartData, PreparedData


class NotificationPreparationService:
    """알림 담당자가 실제 구현으로 교체할 서비스."""

    async def prepare(
        self,
        heart_data: HeartData,
        workflow_id: str,
    ) -> PreparedData:
        """마음데이터로 알림 예약용 JSON을 준비합니다.

        Args:
            heart_data: 알림 내용과 시각을 계산할 공통 선물 정보.
            workflow_id: 네 작업의 결과를 연결하는 요청 추적 ID.

        Returns:
            알림 시스템에 전달할 JSON을 담은 ``PreparedData``.
        """
        # =====================================================================
        # IMPLEMENTATION POINT 4: 알림 담당자가 수정할 곳
        # ---------------------------------------------------------------------
        # 현재는 답례일 7일 전 오전 10시로 mock JSON을 만듭니다.
        # 실제 알림 DTO/예약 API에 맞춰 payload 내부만 변경하세요.
        # 함수 시그니처와 PreparedData 반환 타입은 유지하세요.
        # =====================================================================
        await asyncio.sleep(0)
        target_date = heart_data.target_date or date.today() + timedelta(days=30)
        scheduled_at = datetime.combine(target_date - timedelta(days=7), time(10))
        return PreparedData(
            payload={
                "title": "답례 선물을 준비할 시간이에요",
                "scheduledAt": scheduled_at.isoformat(),
                "workflowId": workflow_id,
            }
        )


notification_preparation_service = NotificationPreparationService()
