"""추천 상품과 감사 메시지를 실제로 생성하는 작업."""

import asyncio
import logging

from app.schemas.agent import (
    GiftData,
    GiftRecommendationInfo,
    RecommendRequest,
    ThankYouMessage,
)
from app.schemas.recommendation import (
    Gender,
    ProductSuggestion,
    SimpleGiftRecommendationRequest,
    SimpleGiftRecommendationResponse,
)
from app.services import record_summary
from app.services import recommendation_rationale as rationale
from app.services.product_search import SearchStats, product_search
from app.services.qwen_service import qwen_service
from app.services.recommendation_policy import (
    CATEGORY_ALIASES,
    SAFE_EXAMPLES,
    price_range,
    reconcile_summary,
)

logger = logging.getLogger(__name__)

# 검색 횟수 상한. 카테고리 수와 무관하게 이만큼은 검색해 후보를 확보합니다.
_SEARCH_QUERY_BUDGET = 3


def _search_targets(categories: list[tuple[str, list[str]]]) -> list[tuple[str, str | None]]:
    """(카테고리, 상품 유형 목록)에서 검색할 (카테고리, 씨앗) 짝을 고릅니다.

    모델이 낸 상품 '유형'을 검색어 씨앗으로 씁니다. 카테고리명만으로는 검색이 잘 안 됩니다.
    카테고리를 한 바퀴씩 돌며 유형을 하나씩 더해 검색 예산을 채웁니다. 카테고리 수로
    나누기만 하면 2개일 때 1개씩 2회에서 멈춰 예산이 남았습니다. 검색이 적으면 후보가
    모자라 예산에 맞는 상품이 하나도 안 남습니다.

    씨앗이 모자라 예산을 못 채우는 것은 정상입니다. Tavily Search 는 1회가 1크레딧이라
    없는 유형을 지어내면서까지 횟수를 채우지 않습니다.
    """
    targets: list[tuple[str, str | None]] = []
    for round_index in range(_SEARCH_QUERY_BUDGET):
        for name, examples in categories:
            if len(targets) >= _SEARCH_QUERY_BUDGET:
                return targets
            if round_index < len(examples):
                targets.append((name, examples[round_index]))
            elif round_index == 0:
                # 유형이 하나도 없는 카테고리도 이름만으로 한 번은 검색합니다.
                targets.append((name, None))
    return targets


def _preplanned_targets(
    request: SimpleGiftRecommendationRequest,
) -> list[tuple[str, str | None]] | None:
    """모델을 기다리지 않고 확정할 수 있는 검색 조건. 확정할 수 없으면 ``None``.

    사용자가 예산과 카테고리를 둘 다 지정한 경우입니다. 이때 가격 범위는 정책이 지정
    예산을 그대로 쓰고(``recommendation_policy.price_range``), 카테고리는 프롬프트가
    "이 안에서만 고르세요"로 제한한 뒤 정규화가 다시 한 번 좁힙니다. 씨앗도 모델 응답에
    실리는 값과 같은 ``SAFE_EXAMPLES`` 라, 모델을 기다려도 같은 검색어가 나옵니다.

    주의: 모델이 지정 카테고리를 하나도 고르지 않으면 ``normalize_recommendation`` 이
    모델 카테고리를 그대로 두므로 응답 카테고리와 검색 카테고리가 갈릴 수 있습니다.
    그래도 검색은 사용자가 고른 카테고리로 한 것이라 엉뚱한 결과는 아닙니다.
    """
    if request.budget_min is None and request.budget_max is None:
        return None
    names: list[str] = []
    for raw in request.preferred_categories:
        name = CATEGORY_ALIASES.get(raw, raw)
        if name in SAFE_EXAMPLES and name not in names:
            names.append(name)
    if not names:
        return None
    # 정규화가 카테고리를 3개로 자르므로 검색도 같은 수까지만 봅니다.
    return _search_targets([(name, list(SAFE_EXAMPLES[name])) for name in names[:3]])


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
        gift_price=gift_data.gift_price or 0,
        age=gift_data.age,
        gender=gift_data.gender or Gender.UNKNOWN,
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
        # 셋 다 없는 요청은 스키마에서 막습니다. 여기서 기본값을 채우면
        # 지어낸 금액이 그대로 추천 근거 문장에 실립니다.
        gift_price=req.gift_price or req.budget_max or req.budget_min or 0,
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
        request = build_request(gift_data)
        stats = SearchStats()
        recommendation = await asyncio.to_thread(qwen_service.recommend_simple, request)
        recommendation.products = await self._find_products(recommendation, stats)
        return self._finalize(request, recommendation, stats)

    async def recommend_only(self, req: "RecommendRequest") -> GiftRecommendationInfo:
        """추천만 단독으로 실행합니다. 기록·캘린더·알림은 건드리지 않습니다.

        사용자가 예산과 카테고리를 둘 다 지정했으면 검색 조건이 모델 없이 확정되므로
        (``recommendation_policy.price_range`` 가 지정 예산을 그대로 쓰고, 프롬프트와
        정규화가 지정 카테고리 안에서만 고르게 합니다) 검색을 추천 모델과 동시에
        시작합니다. 검색 갈래가 통째로 모델 시간 뒤에 숨습니다.
        """
        request = build_request_from_inputs(req)
        stats = SearchStats()
        targets = _preplanned_targets(request)
        if targets is None:
            recommendation = await asyncio.to_thread(qwen_service.recommend_simple, request)
            recommendation.products = await self._find_products(recommendation, stats)
        else:
            low, high = price_range(request)
            # return_exceptions=True 로 두 갈래를 모두 기다립니다. 그렇게 하지 않으면
            # 모델이 먼저 죽었을 때 검색 태스크가 주인 없이 남습니다.
            recommendation, products = await asyncio.gather(
                asyncio.to_thread(qwen_service.recommend_simple, request),
                self._search(targets, low, high, stats),
                return_exceptions=True,
            )
            if isinstance(recommendation, BaseException):
                raise recommendation
            if isinstance(products, BaseException):
                logger.warning("상품 검색 중 예외. 카테고리 추천만 제공합니다: %s", products)
                products = []
            recommendation.products = products
        return self._finalize(request, recommendation, stats)

    @staticmethod
    def _finalize(
        request: SimpleGiftRecommendationRequest,
        recommendation: SimpleGiftRecommendationResponse,
        stats: SearchStats,
    ) -> GiftRecommendationInfo:
        """상품이 확정된 뒤에만 만들 수 있는 문장을 채워 응답을 완성합니다.

        summary 는 모델이 검색 **전에** 씁니다. 그래서 "커피나 차 관련 제품으로
        답례하는 것을 추천합니다" 라고 써 놓고 화면에는 생활용품 하나가 나가는 일이
        실측에서 나왔습니다. 상품이 나온 뒤라야 그 어긋남을 알 수 있으므로,
        summary 와 rationale 을 여기서 한 번에 실제 결과에 맞춥니다.
        """
        categories = [c.category for c in recommendation.categories]
        recommendation.summary = reconcile_summary(
            recommendation.summary, categories, recommendation.products
        )
        recommendation.rationale = rationale.build(
            request,
            categories,
            recommendation.products,
            recommendation.recommended_price_min,
            recommendation.recommended_price_max,
            stats.examined,
        )
        return GiftRecommendationInfo(
            recommend_gift=recommendation,
            message=ThankYouMessage(
                tone="따뜻하고 구체적이며 부담 없는 말투",
                content=recommendation.suggested_message,
                # 추천 백엔드입니다. 메시지 문장의 출처가 아닙니다.
                generated_by=recommendation.source,
                # 문장의 출처는 이쪽입니다. 정책이 길이 미달로 메시지만 교체해도
                # generated_by 는 BEDROCK_CLAUDE 그대로라 여기서만 드러납니다.
                message_source=recommendation.message_source,
            ),
        )

    @classmethod
    async def _find_products(
        cls,
        recommendation: SimpleGiftRecommendationResponse,
        stats: SearchStats,
    ) -> list[ProductSuggestion]:
        """추천 카테고리와 가격 범위로 실제 상품을 찾습니다."""
        targets = _search_targets(
            [(c.category, list(c.product_examples)) for c in recommendation.categories]
        )
        return await cls._search(
            targets,
            recommendation.recommended_price_min,
            recommendation.recommended_price_max,
            stats,
        )

    @staticmethod
    async def _search(
        targets: list[tuple[str, str | None]], low: int, high: int, stats: SearchStats
    ) -> list[ProductSuggestion]:
        """검색을 실제로 부릅니다. 결과가 없어도 추천은 그대로 나갑니다.

        ``stats`` 는 상품 0건의 이유를 근거 문장이 구분해 말하는 데 씁니다. 가격을
        모르는 상품은 노출하지 않으므로 0건이 예전보다 자주 나오고, 그때 "검색
        결과가 없어" 와 "찾았지만 가격이 맞지 않아" 는 다른 말입니다.
        """
        if not product_search.is_available or not targets:
            return []
        products = await product_search.search(targets, low, high, stats=stats)
        if not products:
            logger.info(
                "상품 검색 결과 없음(심사 후보 %d건). 카테고리 추천만 제공합니다.", stats.examined
            )
        return products


recommendation_preparation_service = RecommendationPreparationService()
