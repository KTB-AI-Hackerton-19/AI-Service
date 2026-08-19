"""Google MCP 캘린더 등록 요청 데이터를 준비하는 작업."""

import asyncio
from datetime import date, timedelta

from app.schemas.agent import GiftData, PreparedData


class CalendarPreparationService:
    """캘린더 담당자가 실제 Google MCP 구현으로 교체할 서비스."""

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
            Google MCP 호출 또는 백엔드 전달에 사용할 ``PreparedData``.
        """
        # =====================================================================
        # IMPLEMENTATION POINT 3: Google 캘린더 담당자가 수정할 곳
        # ---------------------------------------------------------------------
        # 현재는 mock JSON만 반환합니다. Google MCP를 연결할 경우에도
        # 입력 GiftData와 출력 PreparedData 계약은 유지하세요.
        # target_date가 없을 때의 기본 정책도 이 파일에서 관리합니다.
        # =====================================================================
        await asyncio.sleep(0)
        target_date = gift_data.target_date or date.today() + timedelta(days=30)
        person = gift_data.person_name or "상대방"
        return PreparedData(
            payload={
                "provider": "GOOGLE_MCP_MOCK",
                "title": f"{person}님 답례 준비",
                "date": target_date.isoformat(),
                "workflowId": workflow_id,
            }
        )


calendar_preparation_service = CalendarPreparationService()
