"""LangGraph 로 조립한 답례 선물 추천 에이전트(실험 경로).

무엇이 다른가
    기존 분할 경로(``tasks/recommendation.py`` 의 ``_generate_split``)와 **같은 세
    Bedrock 호출, 같은 프롬프트, 같은 정규화**를 쓰되, 실행 순서를 손으로 짠
    ``asyncio.gather`` 대신 상태 그래프로 선언합니다. 프롬프트·정책·검색 함수를
    전부 기존 모듈에서 임포트하므로 **출력 품질은 구성상 동일**하고, 이 모듈이
    바꾸는 것은 오케스트레이션 계층 하나입니다.

지연을 지키는 설계 (이 파일에서 가장 중요한 결정)
    LangGraph 는 Pregel 식 superstep 으로 돕니다. 한 superstep 의 노드가 전부
    끝나야 다음 superstep 이 출발하므로, 노드를 한 층에 늘어놓으면 **가장 느린
    노드가 배리어**가 됩니다. 실측(시뮬레이션 지연, scripts/benchmark_graph.py):

        순진한 단층 배치   search 가 message(4.0초)를 기다림 → 총 9.1초 (+1.8초 회귀)
        가지 캡슐화        asyncio 구현과 동일한 7.4초 (오버헤드 +11ms)

    그래서 **서로 독립인 사슬을 서브그래프로 캡슐화**합니다. 최상위 층에는 서로
    기다릴 이유가 없는 가지만 늘어놓고, 의존 관계(plan → prose/search)는 가지
    안에 숨깁니다. 이것이 기존 asyncio 코드와 같은 임계 경로를 만듭니다.

        일반 모드                              선계획 모드(예산+카테고리 지정)
        START ─┬─ message                      START ─┬─ message
               └─ [plan ─┬─ prose   ]                 ├─ [plan ── prose]
                         └─ search ↺]                 └─ search ↺
               → finalize(defer)                      → finalize(defer)

    ``finalize`` 는 ``defer=True`` 라 활성 가지가 전부 끝난 뒤 한 번만 돕니다.

무엇이 "에이전트"인가
    ``search ↺`` 의 자기 순환입니다. 상품이 0건이면 결과를 관찰하고(0건 + 남은
    예산), 아직 쓰지 않은 검색 씨앗으로 **한 번** 재검색을 결정합니다. 기존
    파이프라인은 0건이면 그대로 내보냈습니다. 재시도는 실패 경로에서만 돌므로
    정상 경로의 지연은 그대로이고, 시간 예산 가드가 재시도의 상한을 겁니다.

    반대로 모델에게 도구 선택을 맡기는 ReAct 루프는 **넣지 않습니다**. 도구 선택
    호출마다 왕복(고정비 약 1.2초 + 토큰)이 붙고 호출 횟수가 비결정적이라, 지연
    예측이 무너집니다. 검색 여부를 파이프라인이 결정론적으로 부른다는 기존 결정
    (config.py 의 tavily 주석)을 그래프에서도 그대로 따릅니다.

체크포인터는 쓰지 않습니다. 이 서비스는 상태를 보관하지 않는다는 계약이라
(README: 백엔드가 직전 응답을 들고 있다가 되돌려줌) 저장할 상태가 없고,
체크포인트 직렬화 비용만 생깁니다.
"""

import logging
import operator
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.schemas.agent import GiftData, GiftRecommendationInfo, RecommendRequest
from app.schemas.recommendation import (
    ProductSuggestion,
    SimpleGiftRecommendationRequest,
    SimpleGiftRecommendationResponse,
)
from app.services.product_search import SearchStats
from app.services.recommendation_policy import (
    SAFE_EXAMPLES,
    normalize_recommendation,
    price_range,
)
from app.services.recommendation_stages import recommendation_stages

# 기존 파이프라인의 헬퍼를 그대로 씁니다. 여기 복사해 두면 두 경로가 반드시
# 어긋납니다(_merge_reasons 의 이름 대조, _search_targets 의 예산 채우기 규칙).
# 순환 임포트가 아닙니다 — tasks/recommendation.py 는 이 모듈을 함수 안에서만
# 지연 임포트합니다.
from app.services.tasks.recommendation import (
    _SEARCH_QUERY_BUDGET,
    RecommendationPreparationService,
    _merge_reasons,
    _preplanned_targets,
    _search_targets,
    build_request,
    build_request_from_inputs,
)

logger = logging.getLogger(__name__)

# 재검색을 허용하는 경과 시간 상한(작업 예산 대비 비율). 이 시점을 넘겨 재검색을
# 시작하면 남은 시간이 검색 최악 지연보다 짧아져, 재검색이 타임아웃으로 통째로
# 버려집니다. task_timeout_seconds 30초 기준 15초 — 검색 한 바퀴(실측 6~9초)를
# 안전하게 마칠 수 있는 마지막 출발선입니다.
_RETRY_BUDGET_RATIO = 0.5


class RecommendationState(TypedDict, total=False):
    """그래프를 흐르는 상태. 병렬 가지는 서로 다른 키에만 씁니다.

    ``search_round``·``examined``·``searched`` 는 재검색 루프가 회차를 거듭하며
    **누적**해야 하는 값이라 add 리듀서를 답니다. 나머지는 쓰는 노드가 하나뿐이라
    기본(마지막 값)으로 충분합니다.
    """

    request: SimpleGiftRecommendationRequest
    # 모델 없이 확정된 검색 조건. None 이면 일반 모드(계획을 기다려 검색)입니다.
    preplanned: list[tuple[str, str | None]] | None
    # 재검색 시간 예산 판단용 시작 시각(time.monotonic).
    started: float
    plan: dict[str, Any]
    prose: dict[str, Any]
    message: dict[str, Any]
    products: list[ProductSuggestion]
    examined: Annotated[int, operator.add]
    search_round: Annotated[int, operator.add]
    searched: Annotated[list, operator.add]
    # 재검색이 남은 씨앗을 고를 때 쓰는 (카테고리, 예시 목록). 1차 검색이 채웁니다.
    search_categories: list[tuple[str, list[str]]]
    result: GiftRecommendationInfo


def _raw_categories(state: RecommendationState) -> list[dict]:
    """plan 결과에서 dict 인 카테고리만 골라냅니다. 기존 경로와 같은 식입니다."""
    plan = state.get("plan") or {}
    return [c for c in (plan.get("categories") or []) if isinstance(c, dict)]


async def _plan_node(state: RecommendationState) -> dict:
    """1단계: 카테고리와 점수. 예외를 삼키지 않습니다.

    ``recommendation_stages.plan`` 은 호출 실패를 빈 dict 로 돌려주는 계약이라
    여기 예외가 오면 프로그래밍 오류입니다. 기존 경로도 이 경우 전체를 중단하므로
    (``_generate_split`` 의 재-raise) 같은 의미를 유지합니다 — 그래프에서 노드
    예외는 실행 전체를 실패시킵니다.
    """
    return {"plan": await recommendation_stages.plan(state["request"])}


async def _prose_node(state: RecommendationState) -> dict:
    """2단계: 카테고리별 이유와 요약. 실패하면 빈 dict — 폴백이 채웁니다."""
    try:
        return {"prose": await recommendation_stages.prose(state["request"], _raw_categories(state))}
    except Exception as exc:
        # 기존 경로의 gather(return_exceptions=True) 와 같은 의미입니다. 한 단계의
        # 실패가 추천 전체를 죽이면 나눈 뜻이 없습니다.
        logger.warning("추천 이유·요약 단계 예외. 폴백으로 대체합니다: %s", exc)
        return {"prose": {}}


async def _message_node(state: RecommendationState) -> dict:
    """감사 메시지. 카테고리에 의존하지 않으므로 최상위 가지로 t=0 에 출발합니다."""
    try:
        return {"message": await recommendation_stages.message(state["request"])}
    except Exception as exc:
        logger.warning("추천 메시지 단계 예외. 폴백으로 대체합니다: %s", exc)
        return {"message": {}}


def _categories_for_search(state: RecommendationState) -> list[tuple[str, list[str]]]:
    """이번 실행에서 검색이 쓸 (카테고리, 씨앗 목록)을 정합니다.

    일반 모드는 응답에 실릴 것과 **같은** 카테고리로 검색해야 하므로, 최종 응답이
    쓰는 ``normalize_recommendation`` 을 한 번 더 돌립니다(순수 함수, 기존 경로와
    같은 코드). 선계획 모드는 씨앗이 이미 ``SAFE_EXAMPLES`` 에서 나왔으므로 그
    이름으로 예시 목록을 되찾습니다.
    """
    preplanned = state.get("preplanned")
    if preplanned is not None:
        names = list(dict.fromkeys(name for name, _ in preplanned))
        return [(name, list(SAFE_EXAMPLES[name])) for name in names if name in SAFE_EXAMPLES]
    planned = normalize_recommendation(state["request"], {"categories": _raw_categories(state)})
    return [(c["category"], list(c["product_examples"])) for c in planned["categories"]]


def _rotation_targets(
    categories: list[tuple[str, list[str]]],
    searched: list[tuple[str, str | None]],
) -> list[tuple[str, str | None]]:
    """1차에 쓰지 않은 씨앗으로 재검색 조건을 만듭니다. 남은 씨앗이 없으면 빈 목록.

    ``_search_targets`` 와 같은 라운드 로빈 순서로 전체 후보를 늘어놓고, 이미 쓴
    짝을 뺀 앞쪽만 취합니다. 없는 씨앗을 지어내면서까지 재검색하지 않는 것은 1차와
    같은 원칙입니다(검색 1회 = 1크레딧).
    """
    used = {tuple(t) for t in searched}
    remaining: list[tuple[str, str | None]] = []
    max_rounds = max((len(examples) for _, examples in categories), default=0)
    for round_index in range(max_rounds):
        for name, examples in categories:
            if round_index < len(examples) and (name, examples[round_index]) not in used:
                remaining.append((name, examples[round_index]))
    return remaining[:_SEARCH_QUERY_BUDGET]


async def _search_node(state: RecommendationState) -> dict:
    """상품 검색. 1차는 계획(또는 선계획)의 씨앗을, 재검색은 남은 씨앗을 씁니다.

    실패해도 추천은 그대로 나가야 하므로 예외를 빈 목록으로 바꿉니다. 기존 경로의
    gather(return_exceptions=True) 와 같은 계약입니다.
    """
    request = state["request"]
    low, high = price_range(request)
    if state.get("search_round", 0) == 0:
        categories = _categories_for_search(state)
        targets = (
            state["preplanned"]
            if state.get("preplanned") is not None
            else _search_targets(categories)
        )
    else:
        categories = state.get("search_categories") or []
        targets = _rotation_targets(categories, state.get("searched") or [])
        logger.info("상품 0건. 남은 씨앗으로 재검색합니다: %s", targets)

    stats = SearchStats()
    try:
        products = await RecommendationPreparationService._search(targets, low, high, stats)
    except Exception as exc:
        logger.warning("상품 검색 중 예외. 카테고리 추천만 제공합니다: %s", exc)
        products = []
    return {
        "products": products,
        "examined": stats.examined,
        "search_round": 1,
        "searched": [tuple(t) for t in targets],
        "search_categories": categories,
    }


def _should_retry_search(state: RecommendationState) -> bool:
    """재검색 여부. 관찰(0건) → 판단(예산·남은 씨앗) → 행동(한 번 더)의 자리입니다.

    네 조건을 모두 만족할 때만 참:
      1. 기능이 켜져 있고,
      2. 1차 검색이 끝났으며(무한 루프 방지 — 재검색은 한 번),
      3. 상품이 0건이고,
      4. 남은 시간이 검색 한 바퀴를 감당하며, 아직 안 쓴 씨앗이 남아 있음.
    """
    if not settings.langgraph_search_retry:
        return False
    if state.get("search_round", 0) != 1 or state.get("products"):
        return False
    elapsed = time.monotonic() - state.get("started", 0.0)
    if elapsed > settings.task_timeout_seconds * _RETRY_BUDGET_RATIO:
        logger.info("상품 0건이지만 경과 %.1f초라 재검색을 건너뜁니다.", elapsed)
        return False
    return bool(
        _rotation_targets(state.get("search_categories") or [], state.get("searched") or [])
    )


def _after_search(state: RecommendationState) -> str:
    """검색 뒤의 조건 분기. 그래프마다 path_map 으로 목적지를 다르게 답니다."""
    return "retry" if _should_retry_search(state) else "done"


def _finalize_node(state: RecommendationState) -> dict:
    """세 단계 결과와 상품을 기존 경로와 같은 규칙으로 병합합니다.

    ``_merge_reasons`` → ``normalize_recommendation`` → ``_finalize``(요약 보정과
    근거 생성)를 전부 기존 함수로 부르므로, 같은 입력이면 기존 분할 경로와 같은
    응답이 나옵니다(tests/test_recommendation_graph.py 가 동일성을 지킵니다).
    """
    request = state["request"]
    raw_categories = _raw_categories(state)
    prose = state.get("prose") or {}
    message = state.get("message") or {}

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
        # 기존 분할 경로와 같은 뜻입니다. 카테고리가 하나도 안 나온 실행만 폴백입니다.
        source="BEDROCK_CLAUDE" if raw_categories else "BEDROCK_CLAUDE_FALLBACK",
    )
    products = state.get("products")
    recommendation.products = products if isinstance(products, list) else []
    stats = SearchStats(examined=state.get("examined", 0))
    return {
        "result": RecommendationPreparationService._finalize(request, recommendation, stats)
    }


def _build_normal_branch():
    """일반 모드의 추천 사슬: plan → (prose ∥ search↺).

    서브그래프로 캡슐화하는 이유가 이 모듈 최상단 설계 설명입니다 — 최상위 층에
    plan 을 두면 superstep 배리어 때문에 search 가 message 를 기다립니다(+1.8초).
    prose 와 search 는 어차피 병합이 둘 다를 기다리므로, 서브그래프 안의 배리어는
    임계 경로를 바꾸지 않습니다.
    """
    graph = StateGraph(RecommendationState)
    graph.add_node("plan", _plan_node)
    graph.add_node("prose", _prose_node)
    graph.add_node("search", _search_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "prose")
    graph.add_edge("plan", "search")
    graph.add_edge("prose", END)
    graph.add_conditional_edges("search", _after_search, {"retry": "search", "done": END})
    return graph.compile()


def _build_preplanned_branch():
    """선계획 모드의 생성 사슬: plan → prose. 검색은 최상위에서 t=0 에 따로 돕니다.

    이 모드에서 search 를 이 사슬에 남기면 prose(plan 의존)가 search 와 같은
    superstep 에 묶여 서로를 기다립니다. 검색이 계획보다 오래 걸리므로(실측 6~9초
    vs 2~4초) prose 출발이 그만큼 밀립니다. 갈라 두면 임계 경로가 기존 코드의
    ``max(plan+prose, search, message)`` 그대로입니다.
    """
    graph = StateGraph(RecommendationState)
    graph.add_node("plan", _plan_node)
    graph.add_node("prose", _prose_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "prose")
    graph.add_edge("prose", END)
    return graph.compile()


def _route(state: RecommendationState) -> list[str]:
    """입장 라우터. 검색 조건이 모델 없이 확정되면 검색을 t=0 에 출발시킵니다."""
    if state.get("preplanned") is not None:
        return ["message", "plan_prose", "search"]
    return ["message", "recommend"]


def build_graph():
    """전체 추천 그래프를 조립합니다. 프로세스당 한 번 컴파일해 재사용합니다."""
    graph = StateGraph(RecommendationState)
    graph.add_node("message", _message_node)
    graph.add_node("recommend", _build_normal_branch())
    graph.add_node("plan_prose", _build_preplanned_branch())
    graph.add_node("search", _search_node)
    # defer=True: 활성화된 가지가 전부 끝난 뒤 한 번만 돕니다. 모드에 따라 도는
    # 가지가 다르므로(list-edge 는 고정 목록을 전부 기다려 교착) 이걸 써야 합니다.
    graph.add_node("finalize", _finalize_node, defer=True)
    graph.add_conditional_edges(START, _route, ["message", "recommend", "plan_prose", "search"])
    graph.add_edge("message", "finalize")
    graph.add_edge("recommend", "finalize")
    graph.add_edge("plan_prose", "finalize")
    graph.add_conditional_edges("search", _after_search, {"retry": "search", "done": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile()


class GraphRecommendationService:
    """기존 ``RecommendationPreparationService`` 와 같은 얼굴의 그래프 실행기.

    호출 측(라우터·gift_agent_service)의 타임아웃과 예외 처리를 그대로 쓰기 위해
    같은 시그니처를 유지합니다. 이 클래스가 갈아 끼우는 것은 오케스트레이션뿐입니다.
    """

    def __init__(self) -> None:
        self.graph = build_graph()

    async def prepare(self, gift_data: GiftData) -> GiftRecommendationInfo:
        """이미지·직접 입력 흐름의 추천 작업. 선계획 없이 계획을 기다려 검색합니다."""
        return await self._run(build_request(gift_data), preplanned=None)

    async def recommend_only(self, request: RecommendRequest) -> GiftRecommendationInfo:
        """추천 단독 실행. 예산·카테고리가 확정이면 검색이 t=0 에 출발합니다."""
        simple = build_request_from_inputs(request)
        return await self._run(simple, preplanned=_preplanned_targets(simple))

    async def _run(
        self,
        request: SimpleGiftRecommendationRequest,
        preplanned: list[tuple[str, str | None]] | None,
    ) -> GiftRecommendationInfo:
        state: RecommendationState = {
            "request": request,
            "preplanned": preplanned,
            "started": time.monotonic(),
        }
        final = await self.graph.ainvoke(state)
        logger.info(
            "LangGraph 추천 완료 mode=%s search_rounds=%d products=%d",
            "preplanned" if preplanned is not None else "normal",
            final.get("search_round", 0),
            len(final.get("products") or []),
        )
        return final["result"]


graph_recommendation_service = GraphRecommendationService()
