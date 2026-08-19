"""선물 기록 저장 요청 데이터를 준비하는 작업."""

import logging
from datetime import datetime

from app.schemas.agent import GiftData, PreparedData
from app.services import record_summary
from app.services.reciprocity_schedule import resolve_schedule

logger = logging.getLogger(__name__)

_CURRENCY = "KRW"


def build_payload(gift_data: GiftData, workflow_id: str) -> dict:
    """백엔드가 저장할 JSON 을 만듭니다.

    ``GiftData`` 원본 필드를 그대로 유지하고 파생 정보만 덧붙입니다.
    여러 건이면 ``records`` 에 전부 담기며, 백엔드는 그 배열을 저장하면 됩니다.
    ``records`` 가 비어 있으면 평면 필드가 유일한 기록입니다.
    """
    schedule = resolve_schedule(gift_data)
    # 저장 대상과 답례 대상은 다릅니다. 거래내역의 출금 건은 기록으로는 남기지만
    # 내가 받은 것이 아니므로 답례 대상 수와 금액 합계에서는 빠집니다.
    selected = gift_data.selected_records
    received = record_summary.received_records(gift_data)

    return {
        # 기존 계약을 읽는 쪽이 깨지지 않도록 원본 필드를 그대로 둡니다.
        **gift_data.model_dump(mode="json"),
        "workflowId": workflow_id,
        # 아래는 저장 편의를 위해 덧붙인 파생 정보입니다.
        # direction / record_type / price_basis 는 GiftData 원본 필드로 이미 들어갑니다.
        "currency": _CURRENCY,
        "summary": record_summary.headline(gift_data),
        "recordCount": len(selected) or 1,
        "receivedCount": len(received) or 1,
        "totalAmount": record_summary.total_amount(gift_data),
        "recordedAt": datetime.now().isoformat(timespec="seconds"),
        # target_date 가 비어 있을 때 규칙으로 계산한 답례일. 원본 target_date 는 건드리지 않습니다.
        "resolvedTargetDate": schedule.target_date.isoformat(),
        "targetDateEstimated": schedule.is_target_estimated,
    }


class GiftRecordPreparationService:
    """공통 선물데이터를 백엔드 저장용 JSON 으로 바꾸는 서비스.

    이 함수는 DB 에 직접 저장하지 않고 "저장할 JSON 준비"만 담당합니다.
    저장은 Spring Boot 가 합니다.
    """

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
        payload = build_payload(gift_data, workflow_id)
        logger.info(
            "선물 기록 준비 workflow=%s %d건 %s",
            workflow_id,
            payload["recordCount"],
            payload["summary"],
        )
        return PreparedData(payload=payload)


gift_record_preparation_service = GiftRecordPreparationService()
