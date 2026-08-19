"""추천 상품과 감사 메시지를 실제로 생성하는 작업."""

import asyncio
import logging

from app.schemas.agent import GiftData, GiftRecommendationInfo, RecommendRequest
from app.schemas.recommendation import (
    ProductSuggestion,
    SimpleGiftRecommendationRequest,
    SimpleGiftRecommendationResponse,
)
from app.services import record_summary
from app.services import recommendation_rationale as rationale
from app.services.product_search import product_search
from app.services.qwen_service import qwen_service

logger = logging.getLogger(__name__)

# 검색 횟수 상한. 카테고리 수와 무관하게 이만큼은 검색해 후보를 확보합니다.
_SEARCH_QUERY_BUDGET = 3


def build_request(gift_data: GiftData) -> SimpleGiftRecommendationRequest:
    """공통 선물데이터를 추천 모델 입력으로 옮깁니다.

    여러 사람에게 받은 경우 각자의 금액과 이름을 함께 넘깁니다. 대표 1건만 넘기면
    5만원 준 사람에게 20만원짜리 답례를 권하는 결과가 나옵니다.
    """
    received = record_summary.received_records(gift_data)
    amounts = [r.price for r in received if r.price]
    people = [r.person_name for r in received if r.person_name]

    return SimpleGiftRecommendationRequest(
        gift_name=gift_data.gift_name,
        gift_price=gift_data.gift_price,
        age=gift_data.age,
        person_name=gift_data.person_name,
        relationship=gift_data.relationship,
        record_type=gift_data.record_type.value,
        event=gift_data.event,
        # 한 건뿐이면 비워 둡니다. 정책이 gift_price 하나만 보고 기존과 같이 동작합니다.
        received_amounts=amounts if len(amounts) > 1 else [],
        people=people if len(amounts) > 1 else [],
    )


def build_request_from_inputs(req: "RecommendRequest") -> SimpleGiftRecommendationRequest:
    """나이·가격대·카테고리·성별만으로도 추천 입력을 만듭니다.

    받은 선물 정보가 없어도 동작합니다. 사용자가 확인 화면에서 조건만 바꿔
    다시 추천받을 때 이미지 분석을 다시 돌릴 이유가 없기 때문입니다.
    """
    return SimpleGiftRecommendationRequest(
        gift_name=req.gift_name or "받은 선물",
        gift_price=req.gift_price or req.budget_max or req.budget_min or 30_000,
        age=req.age,
        gender=req.gender,
        person_name=req.person_name,
        relationship=req.relationship,
        event=req.event,
        budget_min=req.budget_min,
        budget_max=req.budget_max,
        preferred_categories=list(req.categories),
        interests=list(req.interests),
        dislikes=list(req.dislikes),
    )


class RecommendationPreparationService:
    """동기 추론을 비동기 워크플로에 연결하고, 실제 상품을 찾아 붙입니다."""

    async def prepare(self, gift_data: GiftData) -> GiftRecommendationInfo:
        """추천을 실행하고 실제 상품과 감사 메시지를 함께 반환합니다.

        모델이 카테고리와 가격 범위를 정하면, 그 조건으로 신뢰할 수 있는 국내
        거래 플랫폼을 검색해 구매 가능한 상품을 채웁니다. 검색이 실패해도
        카테고리 추천과 메시지는 그대로 나갑니다.

        Args:
            gift_data: 모델이 사용할 선물명, 가격, 선택적 나이와 기록 목록.

        Returns:
            추천 가격·카테고리, 실제 상품, 발송 메시지를 담은 결과.
        """
        recommendation = await asyncio.to_thread(
            qwen_service.recommend_simple,
            build_request(gift_data),
        )
        request = build_request(gift_data)
        recommendation.products = await self._find_products(recommendation)
        recommendation.rationale = rationale.build(
            request,
            [c.category for c in recommendation.categories],
            recommendation.products,
            recommendation.recommended_price_min,
            recommendation.recommended_price_max,
        )

        return GiftRecommendationInfo(
            recommend_gift=recommendation,
            message={
                "tone": "따뜻하고 구체적이며 부담 없는 말투",
                "content": recommendation.suggested_message,
                "generated_by": recommendation.source,
            },
        )

    async def recommend_only(self, req: "RecommendRequest") -> GiftRecommendationInfo:
        """추천만 단독으로 실행합니다. 기록·캘린더·알림은 건드리지 않습니다."""
        request = build_request_from_inputs(req)
        recommendation = await asyncio.to_thread(qwen_service.recommend_simple, request)
        recommendation.products = await self._find_products(recommendation)
        recommendation.rationale = rationale.build(
            request,
            [c.category for c in recommendation.categories],
            recommendation.products,
            recommendation.recommended_price_min,
            recommendation.recommended_price_max,
        )
        return GiftRecommendationInfo(
            recommend_gift=recommendation,
            message={
                "tone": "따뜻하고 구체적이며 부담 없는 말투",
                "content": recommendation.suggested_message,
                "generated_by": recommendation.source,
            },
        )

    @staticmethod
    async def _find_products(
        recommendation: SimpleGiftRecommendationResponse,
    ) -> list[ProductSuggestion]:
        """추천 카테고리와 가격 범위로 실제 상품을 찾습니다."""
        if not product_search.is_available:
            return []

        # 모델이 낸 상품 '유형'을 검색어 씨앗으로 씁니다. 카테고리명만으로는 검색이 잘 안 됩니다.
        # 카테고리가 적으면 한 카테고리의 유형을 여러 개 써서 검색 횟수를 채웁니다.
        # 검색이 한 번뿐이면 후보가 모자라 검색 결과 페이지까지 끌어다 쓰게 됩니다.
        targets: list[tuple[str, str | None]] = []
        examples_per_category = max(1, _SEARCH_QUERY_BUDGET // max(1, len(recommendation.categories)))
        for category in recommendation.categories:
            seeds = category.product_examples[:examples_per_category] or [None]
            targets.extend((category.category, seed) for seed in seeds)
        products = await product_search.search(
            targets,
            recommendation.recommended_price_min,
            recommendation.recommended_price_max,
        )
        if not products:
            logger.info("상품 검색 결과 없음. 카테고리 추천만 제공합니다.")
        return products


recommendation_preparation_service = RecommendationPreparationService()
