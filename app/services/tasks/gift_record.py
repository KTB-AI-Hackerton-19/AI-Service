"""선물 기록 저장 요청 데이터를 준비하는 작업."""

import logging
from datetime import datetime

from app.schemas.agent import GiftData, PreparedData
from app.services.reciprocity_schedule import resolve_schedule

logger = logging.getLogger(__name__)

_CURRENCY = "KRW"
_DIRECTION_RECEIVED = "RECEIVED"


def _summarize(gift_data: GiftData) -> str:
    """사람이 읽을 한 줄 요약. 타임라인 화면에 그대로 쓸 수 있습니다."""
    person = gift_data.person_name or "이름 미상"
    return f"{person}님에게 받은 {gift_data.gift_name} ({gift_data.gift_price:,}원)"


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
        schedule = resolve_schedule(gift_data)

        payload = {
            # GiftData 원본 필드를 그대로 유지합니다. 기존 계약을 읽는 쪽이 깨지지 않게 하기 위함입니다.
            **gift_data.model_dump(mode="json"),
            "workflowId": workflow_id,
            # 아래는 저장 편의를 위해 덧붙인 파생 정보입니다.
            "direction": _DIRECTION_RECEIVED,
            "currency": _CURRENCY,
            "summary": _summarize(gift_data),
            "recordedAt": datetime.now().isoformat(timespec="seconds"),
            # target_date 가 비어 있을 때 규칙으로 계산한 답례일. 원본 target_date 는 건드리지 않습니다.
            "resolvedTargetDate": schedule.target_date.isoformat(),
            "targetDateEstimated": schedule.is_target_estimated,
        }

        logger.info("선물 기록 준비 workflow=%s %s", workflow_id, payload["summary"])
        return PreparedData(payload=payload)


gift_record_preparation_service = GiftRecordPreparationService()
