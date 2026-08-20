"""기존 분할 경로 vs LangGraph 경로의 오케스트레이션 지연을 비교합니다.

무엇을 재는가
    두 경로는 **같은 세 Bedrock 호출과 같은 상품 검색**을 부르므로, 지연 차이는
    오케스트레이션 계층(asyncio.gather vs LangGraph superstep)에서만 나옵니다.
    기본 모드는 그 계층만 분리해 잽니다 — 단계 함수를 실측 비율의 모의 지연으로
    바꿔 끼우고 **실제 서비스 코드 경로**(recommend_only)를 통째로 돌립니다.
    Bedrock 토큰과 Tavily 크레딧을 전혀 쓰지 않고 몇 초 안에 끝납니다.

    모의 지연의 근거(코드 주석의 실측): 분할 경로 중앙값 7.6초에서
    plan 2.6 / prose 4.5 / message 5.0 / search 4.0 초 비율을 따고, 기본 1/10
    스케일로 돌립니다. 오케스트레이션 오버헤드는 절대량(ms 단위)이라 스케일을
    줄일수록 오히려 도드라집니다.

    --real 을 주면 모의 대신 실제 Bedrock 호출로 두 경로를 번갈아 돌립니다.
    벤치마크 방법론은 benchmark_split.py 와 같습니다(번갈아 실행해 네트워크
    상태가 한쪽에 몰리지 않게 함). --search 를 함께 주면 Tavily 도 실제로 부릅니다
    (검색 1회 = 1크레딧, 실행당 최대 3회 × 반복 수).

사용법
    python scripts/benchmark_graph.py                  # 모의 지연, 5회씩 (외부 호출 0)
    python scripts/benchmark_graph.py --runs 10
    python scripts/benchmark_graph.py --real --runs 2  # 실제 Bedrock (토큰 소모)
    python scripts/benchmark_graph.py --mermaid        # 그래프 구조만 출력
"""

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.schemas.agent import RecommendRequest  # noqa: E402
from app.schemas.recommendation import ProductSuggestion  # noqa: E402

# 실측 비율(분할 경로 중앙값 7.6초 기준)을 따른 단계별 모의 지연(초).
STAGE_SECONDS = {"plan": 2.6, "prose": 4.5, "message": 5.0, "search": 4.0}

CASES: list[tuple[str, RecommendRequest]] = [
    (
        "일반(계획을 기다려 검색)",
        RecommendRequest(
            gift_name="베스킨라빈스 쿠폰",
            gift_price=20_000,
            age=31,
            person_name="이지수",
            relationship="친구",
        ),
    ),
    (
        "선계획(예산·카테고리 지정)",
        RecommendRequest(
            gift_name="핸드크림 세트",
            gift_price=18_000,
            budget_min=18_000,
            budget_max=30_000,
            categories=["디저트", "상품권", "생활용품"],
            person_name="이지수",
            relationship="친구",
        ),
    ),
]

PLAN = {
    "categories": [
        {"category": "디저트", "score": 85},
        {"category": "꽃·식물", "score": 70},
        {"category": "생활용품", "score": 60},
    ]
}
PROSE = {
    "reasons": [{"category": "디저트", "reason": "부담 없이 나누기 좋은 답례입니다"}],
    "summary": "커피와 디저트 계열의 답례를 권합니다.",
}
MESSAGE = {
    "suggested_message": (
        "이지수님, 지난번에 챙겨 주신 마음 정말 잘 받았어요. 덕분에 하루가 "
        "내내 따뜻했고, 저도 보답하고 싶어 답례를 고르고 있어요. 조만간 얼굴 "
        "보고 직접 인사드릴게요."
    )
}
PRODUCT = ProductSuggestion(
    title="스페셜티 드립백 세트",
    url="https://gift.kakao.com/product/1",
    source="카카오 선물하기",
    category="디저트",
    price=21_000,
    price_verified=True,
)


def install_simulated_stages(scale: float) -> dict[str, tuple[float, float]]:
    """실제 단계 함수를 모의 지연으로 바꿔 끼우고, 시작·종료 기록 dict 를 돌려줍니다."""
    from app.services.recommendation_stages import recommendation_stages
    from app.services.tasks.recommendation import RecommendationPreparationService

    marks: dict[str, tuple[float, float]] = {}
    clock = {"t0": 0.0}

    def stage(name: str, value: dict):
        async def fn(*_a, **_k):
            start = time.perf_counter() - clock["t0"]
            await asyncio.sleep(STAGE_SECONDS[name] * scale)
            marks[name] = (start, time.perf_counter() - clock["t0"])
            return dict(value)
        return fn

    async def fake_search(targets, low, high, stats):
        start = time.perf_counter() - clock["t0"]
        await asyncio.sleep(STAGE_SECONDS["search"] * scale)
        marks["search"] = (start, time.perf_counter() - clock["t0"])
        stats.examined = 4
        return [PRODUCT.model_copy(deep=True)]

    recommendation_stages.plan = stage("plan", PLAN)
    recommendation_stages.prose = stage("prose", PROSE)
    recommendation_stages.message = stage("message", MESSAGE)
    RecommendationPreparationService._search = staticmethod(fake_search)
    marks["_clock"] = clock  # type: ignore[assignment]
    return marks


async def run_engine(engine: str, request: RecommendRequest) -> float:
    """한 엔진으로 recommend_only 를 한 번 돌리고 벽시계 시간을 돌려줍니다."""
    from app.services.tasks.recommendation import recommendation_preparation_service

    settings.recommendation_langgraph = engine == "langgraph"
    settings.recommendation_split_calls = True
    started = time.perf_counter()
    await recommendation_preparation_service.recommend_only(request)
    return time.perf_counter() - started


async def simulated(runs: int, scale: float) -> None:
    settings.model_backend = "bedrock"
    settings.langgraph_search_retry = True
    marks = install_simulated_stages(scale)
    clock = marks.pop("_clock")

    print(f"모의 지연 {scale}배 스케일, 케이스당 {runs}회씩 번갈아 실행")
    print(f"단계 지연(스케일 전): {STAGE_SECONDS}")
    for name, request in CASES:
        totals: dict[str, list[float]] = {"split": [], "langgraph": []}
        timelines: dict[str, dict] = {}
        for _ in range(runs):
            for engine in ("split", "langgraph"):
                marks.clear()
                clock["t0"] = time.perf_counter()
                totals[engine].append(await run_engine(engine, request))
                timelines[engine] = {
                    k: (round(v[0] / scale, 2), round(v[1] / scale, 2))
                    for k, v in sorted(marks.items(), key=lambda kv: kv[1][0])
                }
        split_med = statistics.median(totals["split"])
        graph_med = statistics.median(totals["langgraph"])
        print(f"\n[{name}]")
        for engine in ("split", "langgraph"):
            med = statistics.median(totals[engine])
            print(f"  {engine:9s} 중앙값 {med:6.3f}초 (환산 {med / scale:5.1f}초)")
            for stage_name, (s, e) in timelines[engine].items():
                print(f"            {stage_name:8s} 시작 {s:5.1f} → 종료 {e:5.1f} (환산 초)")
        delta_ms = (graph_med - split_med) * 1000
        print(f"  오케스트레이션 차이: {delta_ms:+.0f}ms (스케일과 무관한 절대 오버헤드)")


async def real(runs: int, search: bool) -> None:
    if not search:
        settings.tavily_enabled = False
    settings.model_backend = "bedrock"
    print(f"실제 Bedrock 호출로 케이스당 {runs}회씩 번갈아 실행 (검색 {'포함' if search else '제외'})")
    for name, request in CASES:
        totals: dict[str, list[float]] = {"split": [], "langgraph": []}
        for index in range(runs):
            for engine in ("split", "langgraph"):
                elapsed = await run_engine(engine, request)
                totals[engine].append(elapsed)
                print(f"  [{name}] {engine:9s} #{index + 1} {elapsed:5.2f}초")
        for engine in ("split", "langgraph"):
            print(
                f"[{name}] {engine:9s} 중앙값 {statistics.median(totals[engine]):5.2f}초 "
                f"최대 {max(totals[engine]):5.2f}초"
            )


def mermaid() -> None:
    from app.graph.recommendation_graph import graph_recommendation_service

    print(graph_recommendation_service.graph.get_graph(xray=1).draw_mermaid())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="케이스·엔진당 반복 수")
    parser.add_argument("--scale", type=float, default=0.1, help="모의 지연 스케일(기본 1/10)")
    parser.add_argument("--real", action="store_true", help="실제 Bedrock 호출로 비교")
    parser.add_argument("--search", action="store_true", help="--real 에서 Tavily 도 실제 호출")
    parser.add_argument("--mermaid", action="store_true", help="그래프 구조(Mermaid)만 출력")
    args = parser.parse_args()

    if args.mermaid:
        mermaid()
        return
    if args.real:
        asyncio.run(real(args.runs, args.search))
        return
    asyncio.run(simulated(args.runs, args.scale))


if __name__ == "__main__":
    main()
