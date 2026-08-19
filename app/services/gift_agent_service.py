"""독립 작업의 실행 순서, 타임아웃, 결과 병합만 담당하는 오케스트레이터."""

import asyncio
import logging
from typing import Awaitable, TypeVar
from uuid import uuid4

from app.core.config import settings
from app.schemas.agent import (
    GiftAgentResponse,
    GiftRecommendationInfo,
    GiftData,
    InputCategory,
    PreparedData,
    RecordKind,
    TaskStatus,
)
from app.services import record_summary
from app.services.tasks.calendar import (
    CalendarPreparationService,
    calendar_preparation_service,
)
from app.services.tasks.gift_record import (
    GiftRecordPreparationService,
    gift_record_preparation_service,
)
from app.services.tasks.image_analysis import (
    ImageAnalysisService,
    image_analysis_service,
)
from app.services.tasks.notification import (
    NotificationPreparationService,
    notification_preparation_service,
)
from app.services.tasks.recommendation import (
    RecommendationPreparationService,
    recommendation_preparation_service,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")

# 답례 "선물" 추천이 의미 있는 기록 종류.
# 현금·부조금(money)과 영수증(receipt)은 추천 대상이 아닙니다. 축의금 명단에
# 대고 "8,000~12,000원 디저트"를 권하는 것은 사용자에게 의미가 없습니다.
# 청첩장(event_invitation)은 답례품이 아니라 축의금 적정 수준을 안내하므로 포함합니다.
RECOMMENDABLE_KINDS = frozenset({RecordKind.GIFT, RecordKind.EVENT_INVITATION})


class GiftInputAnalysisError(RuntimeError):
    """입력에서 유효한 선물데이터를 만들 수 없을 때 발생합니다."""


class ImageAnalysisError(RuntimeError):
    """외부 이미지 분석 함수가 실패하거나 시간 초과됐을 때 발생합니다."""


class GiftAgentService:
    """다섯 작업 서비스에 위임하고 결과만 합치는 가벼운 조정자.

    생성자 주입을 사용하므로 각 작업을 독립적으로 구현·테스트할 수 있습니다.
    이 클래스에는 S3, Google, 알림, Qwen의 세부 구현을 넣지 않습니다.
    """

    def __init__(
        self,
        image_analyzer: ImageAnalysisService,
        gift_record_preparer: GiftRecordPreparationService,
        calendar_preparer: CalendarPreparationService,
        notification_preparer: NotificationPreparationService,
        recommendation_preparer: RecommendationPreparationService,
    ) -> None:
        """각 역할별 서비스를 주입받아 오케스트레이터를 구성합니다."""
        self.image_analyzer = image_analyzer
        self.gift_record_preparer = gift_record_preparer
        self.calendar_preparer = calendar_preparer
        self.notification_preparer = notification_preparer
        self.recommendation_preparer = recommendation_preparer

    async def run_from_gift_data(self, gift_data: GiftData) -> GiftAgentResponse:
        """이미 준비된 선물데이터로 네 후속 작업을 실행합니다."""
        return await self._run_four_tasks(gift_data)

    async def run_from_image(
        self,
        image_url: str,
        category: InputCategory | None = None,
    ) -> GiftAgentResponse:
        """이미지를 선물데이터로 바꾼 뒤 네 후속 작업을 실행합니다.

        Args:
            image_url: 분석할 이미지 주소.
            category: 사용자가 업로드 화면에서 고른 종류. 지정되면 추천 실행 여부를
                이 값으로 판단합니다. 모델이 이미지를 잘못 분류해도 사용자의 선택이
                뒤집히지 않습니다.
        """
        try:
            gift_data = await self._with_timeout(
                self.image_analyzer.analyze(image_url)
            )
        except asyncio.TimeoutError as exc:
            raise ImageAnalysisError("이미지 분석 시간이 초과되었습니다.") from exc
        except GiftInputAnalysisError:
            raise
        except Exception as exc:
            logger.exception("이미지 분석 실패")
            raise ImageAnalysisError("이미지 분석에 실패했습니다.") from exc
        return await self._run_four_tasks(gift_data, category)

    async def _run_four_tasks(
        self,
        gift_data: GiftData,
        category: InputCategory | None = None,
    ) -> GiftAgentResponse:
        """네 독립 작업을 동시에 실행하고 부분 실패를 포함해 결과를 합칩니다."""
        workflow_id = str(uuid4())
        skip_reason = self._recommendation_skip_reason(gift_data, category)
        results = await asyncio.gather(
            self._with_timeout(
                self.gift_record_preparer.prepare(gift_data, workflow_id)
            ),
            self._with_timeout(
                self.calendar_preparer.prepare(gift_data, workflow_id)
            ),
            self._with_timeout(
                self.notification_preparer.prepare(gift_data, workflow_id)
            ),
            # 추천 대상이 아니면 모델을 아예 호출하지 않습니다. 지연과 비용을 아끼고,
            # 무엇보다 쓸 수 없는 추천을 사용자에게 보여 주지 않기 위함입니다.
            self._skipped_recommendation(skip_reason)
            if skip_reason
            else self._with_timeout(self.recommendation_preparer.prepare(gift_data)),
            return_exceptions=True,
        )
        calendar_info = self._prepared_result(results[1], "캘린더")
        return GiftAgentResponse(
            gift_data=self._prepared_result(results[0], "선물 기록"),
            calendar_info=calendar_info,
            noti_info=self._prepared_result(results[2], "알림"),
            recommend_gift_info=self._recommendation_result(results[3]),
            workflow_id=workflow_id,
            requires_confirmation=self._requires_confirmation(calendar_info),
        )

    @staticmethod
    def _recommendation_skip_reason(
        gift_data: GiftData,
        category: InputCategory | None = None,
    ) -> str | None:
        """답례 선물 추천을 건너뛸 이유. 추천해야 하면 ``None``.

        사용자가 업로드 화면에서 종류를 골랐다면 그 값이 우선입니다. 사람이 직접
        고른 값이 모델의 이미지 분류보다 정확하고, 손글씨 장부처럼 모델이 흔들리는
        입력에서도 결과가 일정해집니다.

        고르지 않았을 때만 받은 기록의 종류로 판단합니다. 대표 1건만 보면 여러 건이
        섞인 장부에서 엉뚱한 결론이 나오므로, 받은 기록 전체를 봅니다.
        """
        if category is InputCategory.GIFT:
            return None
        if category is InputCategory.OCCASION:
            return "사용자가 경조사로 선택해 답례 선물 추천 대신 금액 기준으로 안내하세요."

        received = record_summary.received_records(gift_data)
        kinds = {r.record_type for r in received} if received else {gift_data.record_type}
        if kinds & RECOMMENDABLE_KINDS:
            return None
        if kinds == {RecordKind.RECEIPT}:
            return "영수증은 답례 대상이 아니라 추천을 만들지 않았습니다."
        if kinds == {RecordKind.MONEY}:
            return "현금·부조금 기록이라 답례 선물 추천 대신 금액 기준으로 안내하세요."
        return "선물 기록이 아니라 답례 선물 추천을 만들지 않았습니다."

    @staticmethod
    async def _skipped_recommendation(reason: str) -> GiftRecommendationInfo:
        """추천을 건너뛰었다는 결과를 만듭니다. 실패가 아닙니다."""
        return GiftRecommendationInfo(status=TaskStatus.SKIPPED, reason=reason)

    @staticmethod
    def _requires_confirmation(calendar_info: PreparedData) -> bool:
        """캘린더가 아직 등록되지 않았으면 사용자 확인이 필요합니다."""
        payload = calendar_info.payload or {}
        return not payload.get("registered", False)

    @staticmethod
    async def _with_timeout(coroutine: Awaitable[T]) -> T:
        """한 작업이 전체 요청을 무한정 점유하지 않도록 제한 시간을 적용합니다."""
        return await asyncio.wait_for(
            coroutine,
            timeout=settings.request_timeout_seconds,
        )

    @staticmethod
    def _prepared_result(result: object, task_name: str) -> PreparedData:
        """일반 준비 작업의 예외를 외부에 노출하지 않는 ERROR 결과로 바꿉니다."""
        if isinstance(result, PreparedData):
            return result
        if isinstance(result, BaseException):
            logger.error("%s 준비 실패: %s", task_name, result)
        return PreparedData(
            status=TaskStatus.ERROR,
            error=f"{task_name} 준비 중 오류가 발생했습니다.",
        )

    @staticmethod
    def _recommendation_result(result: object) -> GiftRecommendationInfo:
        """추천 작업의 예외를 추천 전용 ERROR 결과로 바꿉니다."""
        if isinstance(result, GiftRecommendationInfo):
            return result
        if isinstance(result, BaseException):
            logger.error("추천 선물 준비 실패: %s", result)
        return GiftRecommendationInfo(
            status=TaskStatus.ERROR,
            error="추천 선물과 메시지 준비 중 오류가 발생했습니다.",
        )


gift_agent_service = GiftAgentService(
    image_analyzer=image_analysis_service,
    gift_record_preparer=gift_record_preparation_service,
    calendar_preparer=calendar_preparation_service,
    notification_preparer=notification_preparation_service,
    recommendation_preparer=recommendation_preparation_service,
)
