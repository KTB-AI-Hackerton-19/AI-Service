"""선물 기록 저장 요청 데이터를 준비하는 작업."""

import asyncio

from app.schemas.agent import GiftData, PreparedData


class GiftRecordPreparationService:
    """선물 기록 담당자가 실제 구현으로 교체할 서비스."""

    async def prepare(
        self,
        gift_data: GiftData,
        workflow_id: str,
    ) -> PreparedData:
        """공통 선물데이터를 백엔드 저장용 JSON으로 변환합니다.

        Args:
            gift_data: 선물명, 가격, 나이 등이 들어 있는 공통 입력.
            workflow_id: 네 작업의 결과를 연결하는 요청 추적 ID.

        Returns:
            백엔드가 저장 함수에 전달할 JSON을 담은 ``PreparedData``.
        """
        # =====================================================================
        # IMPLEMENTATION POINT 2: 선물데이터 준비 담당자가 수정할 곳
        # ---------------------------------------------------------------------
        # Spring Boot의 선물 기록 저장 DTO 계약에 맞춰 payload를 만드세요.
        # 이 함수는 DB에 직접 저장하지 않고 "저장할 JSON 준비"만 담당합니다.
        # 함수 시그니처와 PreparedData 반환 타입은 유지하세요.
        # =====================================================================
        await asyncio.sleep(0)
        return PreparedData(
            payload={
                **gift_data.model_dump(mode="json"),
                "workflowId": workflow_id,
            }
        )


gift_record_preparation_service = GiftRecordPreparationService()
