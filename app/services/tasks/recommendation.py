"""Qwen 추천 상품과 감사 메시지를 실제로 생성하는 작업."""

import asyncio

from app.schemas.agent import GiftRecommendationInfo, GiftData
from app.schemas.recommendation import SimpleGiftRecommendationRequest
from app.services.qwen_service import qwen_service
from app.services.product_search import product_search_service


class RecommendationPreparationService:
    """Qwen의 동기 추론을 비동기 워크플로에 연결합니다."""

    async def prepare(self, gift_data: GiftData) -> GiftRecommendationInfo:
        """실제 Qwen 추천을 실행하고 감사 메시지를 함께 반환합니다.

        Args:
            gift_data: 모델이 사용할 선물명, 가격, 선택적 나이.

        Returns:
            추천 가격·카테고리와 발송 메시지를 담은 결과.
        """
        recommendation = await asyncio.to_thread(
            qwen_service.recommend_simple,
            SimpleGiftRecommendationRequest(
                gift_name=gift_data.gift_name,
                gift_price=gift_data.gift_price,
                age=gift_data.age,
                person_name=gift_data.person_name,
                relationship=gift_data.relationship,
            ),
        )
        # Qwen이 만든 카테고리별 검색어를 외부 검색 도구에 전달합니다.
        # 검색은 병렬 실행되며, 검색 업체 장애 시 products만 빈 배열이 됩니다.
        searches = [
            product_search_service.search_safely(
                category.search_query,
                recommendation.recommended_price_min,
                recommendation.recommended_price_max,
            )
            for category in recommendation.categories
        ]
        product_groups = await asyncio.gather(*searches)
        for category, products in zip(recommendation.categories, product_groups):
            category.products = products

        return GiftRecommendationInfo(
            recommend_gift=recommendation,
            message={
                "tone": "따뜻하고 구체적이며 부담 없는 말투",
                "content": recommendation.suggested_message,
                "generated_by": recommendation.source,
            },
        )


recommendation_preparation_service = RecommendationPreparationService()
