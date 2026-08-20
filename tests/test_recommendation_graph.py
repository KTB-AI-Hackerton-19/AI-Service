"""LangGraph 추천 경로가 기존 분할 경로와 같은 답을 같은 속도 구조로 내는지 확인합니다.

이 파일이 지키는 것은 셋입니다.

1. **동일성**: 같은 모델 출력·같은 검색 결과에서 두 경로의 최종 응답이 같아야
   합니다. 그래프는 오케스트레이션만 바꾸는 계층이라, 여기서 출력이 갈리면
   버그입니다(프롬프트·정규화·근거 생성을 전부 기존 함수로 부릅니다).

2. **동시성 구조**: LangGraph 는 superstep 배리어가 있어 노드를 잘못 배치하면
   검색이 감사 메시지를 기다립니다(시뮬레이션 실측 +1.8초). 가지 캡슐화가 그
   회귀를 막는지 — 검색이 메시지 종료 전에 출발하는지 — 시간으로 검증합니다.

3. **재검색 루프의 경계**: 상품 0건에서만, 남은 씨앗으로만, 한 번만 돕니다.
   경계가 없으면 크레딧과 지연이 비결정적으로 늘어 분할 경로보다 나빠집니다.
"""

import asyncio
import time

import pytest

pytest.importorskip("langgraph")

from app.core.config import settings
from app.schemas.agent import GiftData, RecommendRequest
from app.schemas.recommendation import ProductSuggestion
from app.services.recommendation_stages import recommendation_stages
from app.services.tasks.recommendation import (
    RecommendationPreparationService,
    recommendation_preparation_service,
)
from app.graph import recommendation_graph
from app.graph.recommendation_graph import graph_recommendation_service

PLAN = {
    "categories": [
        {"category": "커피·차", "score": 85},
        {"category": "식품·디저트", "score": 70},
        {"category": "생활용품", "score": 60},
    ]
}
PROSE = {
    "reasons": [
        {"category": "커피·차", "reason": "부담 없이 나누기 좋은 답례입니다"},
        {"category": "식품·디저트", "reason": "취향을 덜 타는 무난한 선택입니다"},
    ],
    "summary": "커피와 디저트 계열의 답례를 권합니다.",
}
MESSAGE = {
    "suggested_message": (
        "이지수님, 지난번 베스킨라빈스 쿠폰 정말 잘 받았어요. "
        "덕분에 하루가 달콤했고 마음 써 주신 게 느껴져서 큰 힘이 됐어요. "
        "저도 보답하고 싶어 답례를 고르고 있으니 조만간 얼굴 보고 인사드릴게요."
    )
}
PRODUCT = ProductSuggestion(
    title="스페셜티 드립백 세트",
    url="https://gift.kakao.com/product/1",
    source="카카오 선물하기",
    category="커피·차",
    price=21_000,
    price_verified=True,
)


def normal_request() -> RecommendRequest:
    return RecommendRequest(
        gift_name="베스킨라빈스 쿠폰",
        gift_price=20_000,
        age=31,
        person_name="이지수",
        relationship="친구",
    )


def preplanned_request() -> RecommendRequest:
    """예산과 카테고리를 둘 다 지정해 선계획 경로를 태우는 입력."""
    return RecommendRequest(
        gift_name="핸드크림 세트",
        gift_price=18_000,
        budget_min=18_000,
        budget_max=30_000,
        categories=["식품·디저트", "상품권"],
        person_name="이지수",
    )


@pytest.fixture
def bedrock(monkeypatch):
    """두 경로 모두 Bedrock 분기를 타게 하고, 외부 호출은 전부 막습니다."""
    monkeypatch.setattr(settings, "model_backend", "bedrock")
    monkeypatch.setattr(settings, "recommendation_split_calls", True)
    monkeypatch.setattr(settings, "recommendation_langgraph", False)


@pytest.fixture
def stages(monkeypatch, bedrock):
    """세 단계와 상품 검색을 고정 출력으로 바꿉니다. 두 경로가 같은 싱글턴을 쓰므로
    한 번의 monkeypatch 가 양쪽 모두에 걸립니다."""
    search_calls: list[list] = []

    def stage(value, delay=0.0):
        async def fn(*_a, **_k):
            if delay:
                await asyncio.sleep(delay)
            return dict(value)
        return fn

    async def fake_search(targets, low, high, stats):
        search_calls.append(list(targets))
        stats.examined = 4
        return [PRODUCT.model_copy(deep=True)]

    monkeypatch.setattr(recommendation_stages, "plan", stage(PLAN))
    monkeypatch.setattr(recommendation_stages, "prose", stage(PROSE))
    monkeypatch.setattr(recommendation_stages, "message", stage(MESSAGE))
    monkeypatch.setattr(
        RecommendationPreparationService, "_search", staticmethod(fake_search)
    )
    return search_calls


async def run_both(request: RecommendRequest):
    """같은 입력을 기존 분할 경로와 그래프 경로로 실행해 나란히 돌려줍니다."""
    legacy = await recommendation_preparation_service.recommend_only(request)
    graph = await graph_recommendation_service.recommend_only(request)
    return legacy, graph


class TestTheGraphMatchesTheSplitPipeline:
    """오케스트레이션만 바뀌었으므로 응답은 필드 하나까지 같아야 합니다."""

    async def test_normal_mode_returns_the_same_response(self, stages):
        legacy, graph = await run_both(normal_request())
        assert graph == legacy

    async def test_preplanned_mode_returns_the_same_response(self, stages):
        legacy, graph = await run_both(preplanned_request())
        assert graph == legacy

    async def test_preplanned_mode_searches_with_the_same_targets(self, stages):
        await recommendation_preparation_service.recommend_only(preplanned_request())
        legacy_targets = stages[0]
        stages.clear()
        await graph_recommendation_service.recommend_only(preplanned_request())
        assert stages[0] == legacy_targets

    async def test_gift_data_flow_returns_the_same_response(self, stages):
        gift_data = GiftData(
            gift_name="스타벅스 케이크",
            gift_price=35_000,
            person_name="김민수",
            relationship="회사 선배",
            received_at="2026-08-19",
        )
        legacy = await recommendation_preparation_service.prepare(gift_data)
        graph = await graph_recommendation_service.prepare(gift_data)
        assert graph == legacy

    async def test_empty_stages_fall_back_identically(self, stages, monkeypatch):
        """모델이 전부 죽어도 두 경로가 같은 폴백(상품권·템플릿)으로 떨어져야 합니다."""

        async def empty(*_a, **_k):
            return {}

        async def no_products(targets, low, high, stats):
            return []

        for name in ("plan", "prose", "message"):
            monkeypatch.setattr(recommendation_stages, name, empty)
        monkeypatch.setattr(
            RecommendationPreparationService, "_search", staticmethod(no_products)
        )
        # 재검색까지 검증에 섞지 않습니다. 여기서 보는 것은 폴백 동일성뿐입니다.
        monkeypatch.setattr(settings, "langgraph_search_retry", False)

        legacy, graph = await run_both(normal_request())
        assert graph == legacy
        assert graph.recommend_gift.source == "BEDROCK_CLAUDE_FALLBACK"
        assert graph.recommend_gift.categories[0].category == "상품권"
        assert graph.message.message_source != "MODEL"


class TestTheGraphKeepsTheConcurrencyShape:
    """superstep 배리어가 임계 경로에 끼어들지 않아야 지연이 기존과 같습니다."""

    @pytest.fixture
    def timeline(self, monkeypatch, bedrock):
        """단계마다 다른 지연을 줘 시작·종료 시각을 기록합니다.

        message(0.30초)가 plan(0.05초)보다 훨씬 길게 잡은 이유: 배리어가 있다면
        검색 출발이 message 종료 뒤로 밀려 시간 순서가 뒤집히고, 없다면 plan 직후
        출발합니다. 실측 비율(계획 2~4초 vs 메시지 4~7초)을 축소한 값입니다.
        """
        marks: dict[str, tuple[float, float]] = {}
        t0 = time.perf_counter()

        def stage(name, value, delay):
            async def fn(*_a, **_k):
                start = time.perf_counter() - t0
                await asyncio.sleep(delay)
                marks[name] = (start, time.perf_counter() - t0)
                return dict(value)
            return fn

        async def fake_search(targets, low, high, stats):
            start = time.perf_counter() - t0
            await asyncio.sleep(0.05)
            marks["search"] = (start, time.perf_counter() - t0)
            stats.examined = 1
            return [PRODUCT.model_copy(deep=True)]

        monkeypatch.setattr(recommendation_stages, "plan", stage("plan", PLAN, 0.05))
        monkeypatch.setattr(recommendation_stages, "prose", stage("prose", PROSE, 0.05))
        monkeypatch.setattr(
            recommendation_stages, "message", stage("message", MESSAGE, 0.30)
        )
        monkeypatch.setattr(
            RecommendationPreparationService, "_search", staticmethod(fake_search)
        )
        return marks

    async def test_search_does_not_wait_for_the_message(self, timeline):
        await graph_recommendation_service.recommend_only(normal_request())
        assert timeline["search"][0] < timeline["message"][1], (
            "검색이 감사 메시지 종료를 기다렸습니다. superstep 배리어가 임계 경로에 "
            "끼어든 것으로, 가지 캡슐화가 깨졌다는 뜻입니다."
        )

    async def test_prose_does_not_wait_for_the_message_either(self, timeline):
        await graph_recommendation_service.recommend_only(normal_request())
        assert timeline["prose"][0] < timeline["message"][1]

    async def test_preplanned_search_departs_before_the_plan_returns(self, timeline):
        await graph_recommendation_service.recommend_only(preplanned_request())
        assert timeline["search"][0] < timeline["plan"][1], (
            "선계획 검색이 계획 호출을 기다렸습니다. 검색 조건이 모델 없이 확정되는 "
            "입력에서는 t=0 에 출발해야 합니다."
        )


class TestTheSearchRetryIsBounded:
    """재검색은 관찰된 실패(0건)에서만, 남은 씨앗으로만, 한 번만 돕니다."""

    @pytest.fixture
    def empty_then_found(self, monkeypatch, bedrock):
        """첫 검색은 0건, 두 번째부터 상품이 나오는 검색."""
        calls: list[list] = []

        async def fake_search(targets, low, high, stats):
            calls.append(list(targets))
            stats.examined = 0
            if len(calls) == 1:
                return []
            return [PRODUCT.model_copy(deep=True)]

        def stage(value):
            async def fn(*_a, **_k):
                return dict(value)
            return fn

        monkeypatch.setattr(recommendation_stages, "plan", stage(PLAN))
        monkeypatch.setattr(recommendation_stages, "prose", stage(PROSE))
        monkeypatch.setattr(recommendation_stages, "message", stage(MESSAGE))
        monkeypatch.setattr(
            RecommendationPreparationService, "_search", staticmethod(fake_search)
        )
        return calls

    async def test_zero_products_trigger_one_more_search_with_fresh_seeds(
        self, empty_then_found
    ):
        info = await graph_recommendation_service.recommend_only(normal_request())

        assert len(empty_then_found) == 2
        first, second = map(set, (map(tuple, c) for c in empty_then_found))
        assert not first & second, "재검색이 이미 쓴 씨앗을 다시 썼습니다. 크레딧 낭비입니다."
        assert info.recommend_gift.products, "재검색이 찾은 상품이 응답에 실려야 합니다."

    async def test_two_empty_rounds_stop_the_loop(self, monkeypatch, bedrock):
        """상품이 끝내 없어도 두 바퀴에서 멈춰야 합니다. 상한이 없으면 씨앗이 남는 한 돕니다."""
        calls = []

        async def always_empty(targets, low, high, stats):
            calls.append(list(targets))
            return []

        def stage(value):
            async def fn(*_a, **_k):
                return dict(value)
            return fn

        monkeypatch.setattr(recommendation_stages, "plan", stage(PLAN))
        monkeypatch.setattr(recommendation_stages, "prose", stage(PROSE))
        monkeypatch.setattr(recommendation_stages, "message", stage(MESSAGE))
        monkeypatch.setattr(
            RecommendationPreparationService, "_search", staticmethod(always_empty)
        )

        info = await graph_recommendation_service.recommend_only(normal_request())
        assert len(calls) == 2
        assert info.recommend_gift.products == []

    async def test_products_on_the_first_round_skip_the_retry(self, stages):
        await graph_recommendation_service.recommend_only(normal_request())
        assert len(stages) == 1

    async def test_the_flag_turns_the_retry_off(self, empty_then_found, monkeypatch):
        monkeypatch.setattr(settings, "langgraph_search_retry", False)
        await graph_recommendation_service.recommend_only(normal_request())
        assert len(empty_then_found) == 1

    async def test_an_exhausted_time_budget_skips_the_retry(
        self, empty_then_found, monkeypatch
    ):
        """남은 시간이 검색 한 바퀴를 못 감당하면 0건이라도 시작하지 않습니다."""
        monkeypatch.setattr(settings, "task_timeout_seconds", 0.0)
        await graph_recommendation_service.recommend_only(normal_request())
        assert len(empty_then_found) == 1


class TestTheEngineDispatch:
    """플래그가 켜졌을 때만, Bedrock 에서만 그래프 경로로 갑니다."""

    async def test_the_flag_routes_recommend_only_to_the_graph(
        self, stages, monkeypatch
    ):
        monkeypatch.setattr(settings, "recommendation_langgraph", True)
        called = False

        async def spy(request):
            nonlocal called
            called = True
            return await GraphRecommendationServiceOriginal(request)

        GraphRecommendationServiceOriginal = graph_recommendation_service.recommend_only
        monkeypatch.setattr(graph_recommendation_service, "recommend_only", spy)

        await recommendation_preparation_service.recommend_only(normal_request())
        assert called is True

    async def test_non_bedrock_backends_stay_on_the_existing_path(
        self, stages, monkeypatch
    ):
        """그래프 경로는 Bedrock 전용입니다. mock 백엔드는 기존 단일 호출로 돕니다."""
        monkeypatch.setattr(settings, "recommendation_langgraph", True)
        monkeypatch.setattr(settings, "model_backend", "mock")

        async def fail(*_a, **_k):
            raise AssertionError("mock 백엔드가 그래프 경로를 탔습니다.")

        monkeypatch.setattr(graph_recommendation_service, "recommend_only", fail)
        info = await recommendation_preparation_service.recommend_only(normal_request())
        assert info.recommend_gift is not None

    async def test_the_default_config_keeps_the_graph_off(self):
        """기본값이 꺼져 있어야 기존 배포가 이 실험의 영향을 받지 않습니다."""
        assert type(settings).model_fields["recommendation_langgraph"].default is False
