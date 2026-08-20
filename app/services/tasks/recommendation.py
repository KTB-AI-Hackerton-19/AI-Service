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
from app.core.config import settings
from app.services import record_summary
from app.services import recommendation_rationale as rationale
from app.services.product_search import SearchStats, product_search
from app.services.qwen_service import qwen_service
from app.services.recommendation_policy import (
    CATEGORY_ALIASES,
    SAFE_EXAMPLES,
    normalize_recommendation,
    price_range,
    reconcile_summary,
)
from app.services.recommendation_stages import recommendation_stages

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


def _split_enabled() -> bool:
    """분할 호출을 쓸 수 있는 상태인지.

    Bedrock 에서만 씁니다. 분할은 ``recommendation_stages`` 의 비동기 Bedrock
    클라이언트에만 구현돼 있고, vLLM·MLX·mock 은 단일 호출 그대로입니다.
    """
    return settings.recommendation_split_calls and settings.model_backend == "bedrock"


def _langgraph_engine():
    """LangGraph 경로가 켜져 있고 쓸 수 있으면 그 서비스를, 아니면 ``None``.

    분할 경로와 같은 이유로 Bedrock 전용입니다. langgraph 는 선택 의존성이라
    미설치 환경에서 플래그만 켜져 있으면 조용히 죽는 대신 경고를 남기고 기존
    경로로 내려갑니다 — 추천이 안 나가는 것보다 그래프 없이 나가는 것이 낫습니다.
    """
    if not settings.recommendation_langgraph or settings.model_backend != "bedrock":
        return None
    try:
        from app.graph.recommendation_graph import graph_recommendation_service
    except ImportError as exc:
        logger.warning(
            "RECOMMENDATION_LANGGRAPH=true 지만 langgraph 를 임포트하지 못해 "
            "기존 경로로 실행합니다: %s", exc,
        )
        return None
    return graph_recommendation_service


def _merge_reasons(categories: list[dict], prose: dict) -> list[dict]:
    """2단계가 쓴 긴 이유를 1단계 카테고리에 붙입니다.

    **이름으로 대조**합니다. 순서로 붙이면 모델이 카테고리 순서를 바꿔 냈을 때
    이유가 엉뚱한 카테고리에 붙는데, 그건 화면에 그대로 나가는 문장입니다.

    1단계는 이유를 내지 않으므로(``prompt.build_plan_schema``), 2단계가 빠뜨린
    카테고리는 이유 없이 남습니다. 그 자리는 ``normalize_recommendation`` 이
    "관계와 가격대를 고려한 추천입니다" 로 채웁니다.
    """
    reasons: dict[str, str] = {}
    for item in prose.get("reasons") or []:
        if isinstance(item, dict) and item.get("category") and item.get("reason"):
            reasons[str(item["category"])] = str(item["reason"])
    return [
        {**item, "reason": reasons[str(item.get("category"))]}
        if str(item.get("category")) in reasons
        else item
        for item in categories
    ]


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
        engine = _langgraph_engine()
        if engine is not None:
            return await engine.prepare(gift_data)
        request = build_request(gift_data)
        stats = SearchStats()
        if _split_enabled():
            recommendation = await self._generate_split(request, stats, None)
        else:
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
        engine = _langgraph_engine()
        if engine is not None:
            return await engine.recommend_only(req)
        request = build_request_from_inputs(req)
        stats = SearchStats()
        targets = _preplanned_targets(request)
        if _split_enabled():
            # 분할 경로에서도 선계획은 살립니다. 계획 호출이 2초 안쪽이라 이득이
            # 줄기는 하지만, 검색을 0초에 출발시킬 수 있으면 그만큼 더 빠릅니다.
            return self._finalize(
                request, await self._generate_split(request, stats, targets), stats
            )
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

    async def _generate_split(
        self,
        request: SimpleGiftRecommendationRequest,
        stats: SearchStats,
        preplanned: list[tuple[str, str | None]] | None,
    ) -> SimpleGiftRecommendationResponse:
        """세 호출로 나눠 생성하고, 카테고리가 나오는 즉시 검색을 출발시킵니다.

        단일 호출 경로가 "생성 11초 → 검색 9초" 로 직렬이었던 자리입니다. 검색이
        기다리는 것은 카테고리 이름뿐인데 감사 메시지까지 다 쓰기를 기다렸습니다.

        Args:
            request: 추천 입력.
            preplanned: 모델 없이 확정할 수 있는 검색 조건. 있으면 계획 호출도
                기다리지 않고 0초에 검색을 출발시킵니다.

        Returns:
            상품까지 채워진 추천 결과. 세 단계는 각각 독립적으로 실패할 수 있고,
            빈 자리는 ``normalize_recommendation`` 의 기존 폴백이 채웁니다.
        """
        low, high = price_range(request)
        # 감사 메시지는 카테고리에 의존하지 않으므로 계획과 동시에 출발합니다.
        # 근거는 프롬프트 자신입니다 — "답례는 아직 고르는 중이니 선물을 준비한다거나
        # 주겠다는 말은 사용하지 말고" 라서 애초에 카테고리를 쓰면 안 되는 출력입니다.
        message_task = asyncio.create_task(recommendation_stages.message(request))
        search_task = (
            asyncio.create_task(self._search(preplanned, low, high, stats))
            if preplanned is not None
            else None
        )
        try:
            plan = await recommendation_stages.plan(request)
        except BaseException:
            # 계획이 죽으면 남은 태스크는 주인이 없습니다. 놔두면 "Task exception
            # was never retrieved" 로 흘러나옵니다.
            message_task.cancel()
            if search_task is not None:
                search_task.cancel()
            raise

        raw_categories = [c for c in (plan.get("categories") or []) if isinstance(c, dict)]
        if search_task is None:
            # 검색용 카테고리는 응답에 실릴 것과 **같아야** 합니다. 그래서 목록을
            # 손으로 다듬지 않고 최종 응답이 쓰는 함수를 한 번 더 돌립니다(순수
            # 함수라 부작용이 없습니다). 정렬·사용자 지정 카테고리 좁히기·
            # SAFE_EXAMPLES 채움이 여기서 그대로 적용됩니다.
            planned = normalize_recommendation(request, {"categories": raw_categories})
            targets = _search_targets(
                [(c["category"], list(c["product_examples"])) for c in planned["categories"]]
            )
            search_task = asyncio.create_task(self._search(targets, low, high, stats))

        prose, products, message = await asyncio.gather(
            recommendation_stages.prose(request, raw_categories),
            search_task,
            message_task,
            return_exceptions=True,
        )
        for name, value in (("이유·요약", prose), ("상품 검색", products), ("메시지", message)):
            if isinstance(value, BaseException):
                logger.warning("추천 %s 단계 예외. 폴백으로 대체합니다: %s", name, value)

        prose = prose if isinstance(prose, dict) else {}
        message = message if isinstance(message, dict) else {}

        parsed = {
            "categories": _merge_reasons(raw_categories, prose),
            "summary": str(prose.get("summary") or ""),
            "suggested_message": str(message.get("suggested_message") or ""),
        }
        recommendation = SimpleGiftRecommendationResponse(
            **normalize_recommendation(request, parsed),
            input_gift_name=request.gift_name,
            input_gift_price=request.gift_price,
            input_age=request.age,
            model=settings.bedrock_model_id,
            # 단일 호출과 같은 뜻으로 씁니다. 카테고리가 하나도 안 나온 실행만
            # 폴백이고, 이유나 메시지 한 단계가 빈 것은 그 필드의 문제입니다
            # (그건 message_source 가 따로 말합니다).
            source="BEDROCK_CLAUDE" if raw_categories else "BEDROCK_CLAUDE_FALLBACK",
        )
        recommendation.products = products if isinstance(products, list) else []
        return recommendation

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
