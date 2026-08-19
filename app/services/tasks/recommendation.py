"""Qwen 추천 상품과 감사 메시지를 실제로 생성하는 작업."""

import asyncio

from app.schemas.agent import GiftRecommendationInfo, HeartData
from app.schemas.recommendation import SimpleGiftRecommendationRequest
from app.services.qwen_service import qwen_service


class RecommendationPreparationService:
    """Qwen의 동기 추론을 비동기 워크플로에 연결합니다."""

    async def prepare(self, heart_data: HeartData) -> GiftRecommendationInfo:
        """실제 Qwen 추천을 실행하고 감사 메시지를 함께 반환합니다.

        Args:
            heart_data: 모델이 사용할 선물명, 가격, 선택적 나이.

        Returns:
            추천 가격·카테고리와 발송 메시지를 담은 결과.
        """
        recommendation = await asyncio.to_thread(
            qwen_service.recommend_simple,
            SimpleGiftRecommendationRequest(
                gift_name=heart_data.gift_name,
                gift_price=heart_data.gift_price,
                age=heart_data.age,
            ),
        )
        person = heart_data.person_name or "상대방"
        return GiftRecommendationInfo(
            recommendations=recommendation,
            message={
                "tone": "따뜻하고 부담 없는 말투",
                "content": (
                    f"{person}님, 지난번에 챙겨주신 {heart_data.gift_name} 정말 고마웠어요. "
                    "저도 감사한 마음을 담아 준비했어요. 기분 좋게 받아주세요!"
                ),
            },
        )


recommendation_preparation_service = RecommendationPreparationService()
