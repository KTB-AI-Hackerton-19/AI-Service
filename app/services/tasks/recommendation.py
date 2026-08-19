"""추천 상품과 감사 메시지를 실제로 생성하는 작업."""

import asyncio

from app.schemas.agent import GiftData, GiftRecommendationInfo
from app.schemas.recommendation import SimpleGiftRecommendationRequest
from app.services import record_summary
from app.services.qwen_service import qwen_service


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


class RecommendationPreparationService:
    """동기 추론을 비동기 워크플로에 연결합니다."""

    async def prepare(self, gift_data: GiftData) -> GiftRecommendationInfo:
        """추천을 실행하고 감사 메시지를 함께 반환합니다.

        Args:
            gift_data: 모델이 사용할 선물명, 가격, 선택적 나이와 기록 목록.

        Returns:
            추천 가격·카테고리와 발송 메시지를 담은 결과.
        """
        recommendation = await asyncio.to_thread(
            qwen_service.recommend_simple,
            build_request(gift_data),
        )
        return GiftRecommendationInfo(
            recommend_gift=recommendation,
            message={
                "tone": "따뜻하고 구체적이며 부담 없는 말투",
                "content": recommendation.suggested_message,
                "generated_by": recommendation.source,
            },
        )


recommendation_preparation_service = RecommendationPreparationService()
