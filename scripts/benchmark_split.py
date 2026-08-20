"""추천 단일 호출 vs 분할 호출을 같은 입력으로 번갈아 돌려 비교합니다.

무엇을 재는가
    지연  단계별 벽시계 시간과 총 시간. 같은 케이스를 두 모드로 번갈아 돌려
          네트워크 상태 차이가 한쪽에만 몰리지 않게 합니다.
    품질  같은 입력에 대한 두 모드의 출력을 나란히 놓고, 규칙으로 판정할 수
          있는 것만 지표로 셉니다. 문장이 더 좋아졌는지는 사람이 읽어야 하므로
          출력 전문도 함께 남깁니다.

품질 지표를 규칙으로만 세는 이유: 모델에게 모델 출력을 채점시키면 채점자가
흔들리는 만큼 결과도 흔들리고, 여기서 알고 싶은 것은 "분할이 무언가를
망가뜨렸는가" 라는 회귀 여부입니다. 회귀는 규칙으로 잡힙니다 — 템플릿으로
떨어졌는지, 이름을 빠뜨렸는지, 쓰면 안 되는 카테고리·금액을 썼는지.

주의: 실호출입니다. Bedrock 토큰을 쓰고, --search 를 켜면 Tavily 크레딧도
씁니다(검색 1회 = 1크레딧, 실행당 최대 3회).

사용법
    python scripts/benchmark_split.py                      # 모델만, 3회씩 (Tavily 0원)
    python scripts/benchmark_split.py --runs 2 --search    # 검색 포함 종단
    python scripts/benchmark_split.py --json out.json
"""

import argparse
import asyncio
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_latency import Recorder, instrument, pad, probe, width  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.schemas.agent import RecordDirection, RecordKind  # noqa: E402
from app.schemas.recommendation import (  # noqa: E402
    Gender,
    MessageSource,
    SimpleGiftRecommendationRequest,
)
from app.services.recommendation_policy import DEFAULT_SUMMARY  # noqa: E402

# 케이스는 프롬프트가 갈라지는 지점을 하나씩 덮습니다. 분할이 깨진다면 여기서
# 깨집니다 — 안내문(``prompt._notes``)이 단계마다 제대로 실리는지가 이 목록의 요점입니다.
#
# 진입점이 둘인 이유: ``RecommendRequest`` 에는 record_type·여러 명 필드가 없습니다.
# 그 두 케이스는 ``/from-gift-data`` 가 타는 ``prepare()`` 로만 재현됩니다. 예산·
# 카테고리 지정은 반대로 ``/recommend`` 에만 있는 입력이고, 선계획 경로
# (``_preplanned_targets``)를 타는 유일한 케이스라 그쪽으로 넣습니다.
CASES: list[tuple[str, str, dict]] = [
    ("친구·반말", "gift",
     dict(gift_name="베스킨라빈스 쿠폰", gift_price=20_000, age=31,
          gender=Gender.MALE, person_name="이지수", relationship="친구")),
    ("직장 선배·존댓말", "gift",
     dict(gift_name="스타벅스 기프티콘 케이크", gift_price=35_000, age=29,
          gender=Gender.FEMALE, person_name="김민수", relationship="회사 선배")),
    ("예산·카테고리 지정", "recommend",
     dict(gift_name="핸드크림 세트", gift_price=18_000, age=31,
          gender=Gender.MALE, person_name="이지수", relationship="친구",
          budget_min=18_000, budget_max=30_000,
          categories=["식품·디저트", "상품권", "생활용품"])),
    ("청첩장", "gift",
     dict(gift_name="결혼식 청첩장", gift_price=100_000, age=31,
          gender=Gender.MALE, person_name="이서연", relationship="회사 선배",
          record_type=RecordKind.EVENT_INVITATION, event="결혼")),
    ("조의", "gift",
     dict(gift_name="조의금", gift_price=50_000, age=45,
          gender=Gender.MALE, person_name="박정호", relationship="직장 동료",
          event="조의")),
    ("여러 명", "gift",
     dict(gift_name="집들이 선물", gift_price=100_000, age=34,
          gender=Gender.FEMALE, person_name="박정호", relationship="대학 친구",
          records=[
              dict(record_id="1", person_name="김민수", price=30_000,
                   gift_name="집들이 선물", direction=RecordDirection.RECEIVED),
              dict(record_id="2", person_name="이서연", price=50_000,
                   gift_name="집들이 선물", direction=RecordDirection.RECEIVED),
              dict(record_id="3", person_name="박정호", price=100_000,
                   gift_name="집들이 선물", direction=RecordDirection.RECEIVED),
          ])),
]

SINGLE_STAGES = ["추천 모델", "상품 검색"]
SPLIT_STAGES = ["plan", "prose", "message", "상품 검색"]

_MONEY = re.compile(r"\d[\d,]*\s*[만천억]?\s*원")
_SENTENCE = re.compile(r"[^.!?…\n]+[.!?…]?")
# 문장이 존댓말로 끝나는지. 반말 판정은 "존댓말이 아님" 으로 둡니다. 한국어
# 반말 어미를 전부 열거하는 것보다 오판이 적습니다.
_POLITE_END = re.compile(r"(요|죠|다|까|오|봐요|네요)[.!?…]*$")


def install_probes() -> None:
    """두 경로의 단계를 각각 감쌉니다. 앱 코드에는 타이머를 넣지 않습니다."""
    from app.services.product_search import product_search
    from app.services.qwen_service import qwen_service
    from app.services.recommendation_stages import recommendation_stages

    instrument(qwen_service, "recommend_simple", "추천 모델")
    instrument(recommendation_stages, "plan", "plan")
    instrument(recommendation_stages, "prose", "prose")
    instrument(recommendation_stages, "message", "message")
    instrument(product_search, "search", "상품 검색")


# ─────────────────────────── 품질 지표 ───────────────────────────


def polite_ratio(text: str) -> float | None:
    """존댓말로 끝난 문장의 비율. 문장이 없으면 ``None``."""
    sentences = [s.strip() for s in _SENTENCE.findall(text) if s.strip()]
    if not sentences:
        return None
    return sum(bool(_POLITE_END.search(s)) for s in sentences) / len(sentences)


def grade(request: SimpleGiftRecommendationRequest, info) -> dict:
    """규칙으로 판정할 수 있는 것만 셉니다.

    ``category_leak`` 과 ``money_in_message`` 는 프롬프트가 **금지한** 것이라
    0 이 정상입니다. 분할이 감사 메시지에서 카테고리 맥락을 빼는 것이 이 설계의
    핵심 가정이므로, 단일 호출 쪽 ``category_leak`` 도 함께 셉니다 — 원래 0 이라야
    "빼도 된다" 는 말이 성립합니다.
    """
    result = info.recommend_gift
    message = info.message.content
    categories = [c.category for c in result.categories]
    reasons = [c.reason for c in result.categories]
    return {
        "message_source": str(info.message.message_source),
        "from_model": info.message.message_source == MessageSource.MODEL,
        "source": result.source,
        "categories": categories,
        "category_count": len(categories),
        "message_len": len(message),
        "reason_len_mean": round(statistics.fmean(len(r) for r in reasons), 1) if reasons else 0,
        "summary_len": len(result.summary),
        "summary_is_default": result.summary.startswith(DEFAULT_SUMMARY),
        "name_in_message": bool(request.person_name) and request.person_name in message,
        # 프롬프트가 금지한 것들. 0 이 정상입니다.
        "category_leak": sum(c in message for c in categories),
        "money_in_message": bool(_MONEY.search(message)),
        "money_in_summary": bool(_MONEY.search(result.summary)),
        "polite_ratio": polite_ratio(message),
        "product_count": len(result.products),
        "message": message,
        "summary": result.summary,
        "reasons": [f"{c}: {r}" for c, r in zip(categories, reasons)],
    }


# ─────────────────────────── 실행 ───────────────────────────


async def run_one(label: str, entry: str, kwargs: dict, split: bool) -> dict:
    """한 케이스를 한 모드로 1회 실행합니다.

    ``entry`` 가 진입점을 고릅니다. gift 는 ``/from-gift-data`` 가 타는
    ``prepare()``, recommend 는 ``/recommend`` 가 타는 ``recommend_only()`` 입니다.
    둘 다 분할 경로를 타지만 선계획 여부가 달라 지연 구성이 다릅니다.
    """
    from app.schemas.agent import GiftData, RecommendRequest
    from app.services.tasks.recommendation import (
        build_request,
        build_request_from_inputs,
        recommendation_preparation_service,
    )

    settings.recommendation_split_calls = split
    service = recommendation_preparation_service
    if entry == "gift":
        gift_data = GiftData(**kwargs)
        request = build_request(gift_data)
        call = service.prepare(gift_data)
    else:
        req = RecommendRequest(**kwargs)
        request = build_request_from_inputs(req)
        call = service.recommend_only(req)

    recorder: Recorder = probe.start()
    begin = time.perf_counter()
    try:
        info = await call
        total = time.perf_counter() - begin
        quality = grade(request, info)
        error = None
    except Exception as exc:
        total = time.perf_counter() - begin
        quality, error = {}, f"{type(exc).__name__}: {exc}"

    stages = SPLIT_STAGES if split else SINGLE_STAGES
    return {
        "case": label,
        "mode": "split" if split else "single",
        "total": total,
        "stages": {name: recorder.wall(name) for name in stages},
        "error": error,
        **quality,
    }


async def run_all(runs: int, cases: list[tuple[str, str, dict]]) -> list[dict]:
    """케이스마다 두 모드를 번갈아 돌립니다.

    번갈아 도는 것이 중요합니다. 한 모드를 몰아서 돌리면 그 시간대의 Bedrock
    혼잡이 통째로 한쪽 결과가 됩니다.
    """
    rows = []
    for label, entry, kwargs in cases:
        for index in range(1, runs + 1):
            for split in (False, True):
                row = await run_one(label, entry, kwargs, split)
                rows.append(row)
                parts = "  ".join(
                    f"{n} {v:.1f}s" for n, v in row["stages"].items() if v is not None
                )
                mark = "!" if row["error"] else " "
                print(f"  {mark}[{pad(label, 20)} {index}/{runs} {row['mode']:>6}] "
                      f"총 {row['total']:6.2f}s   {parts}")
                if row["error"]:
                    print(f"      → {row['error']}")
    print()
    return rows


# ─────────────────────────── 보고 ───────────────────────────


def latency_table(rows: list[dict], runs: int) -> None:
    print("── 지연 (초) " + "─" * 50)
    print(f"  {pad('케이스', 22)}{'단일':>8}{'분할':>8}{'차이':>9}{'':>3}")
    singles, splits = [], []
    for label in dict.fromkeys(r["case"] for r in rows):
        a = [r["total"] for r in rows if r["case"] == label and r["mode"] == "single"]
        b = [r["total"] for r in rows if r["case"] == label and r["mode"] == "split"]
        if not a or not b:
            continue
        singles += a
        splits += b
        ma, mb = statistics.median(a), statistics.median(b)
        print(f"  {pad(label, 22)}{ma:>8.2f}{mb:>8.2f}{mb - ma:>+9.2f}"
              f"{'':>2}{(mb - ma) / ma * 100:>+6.0f}%")
    if singles and splits:
        ma, mb = statistics.median(singles), statistics.median(splits)
        print(f"  {pad('── 전체 중앙값', 22)}{ma:>8.2f}{mb:>8.2f}{mb - ma:>+9.2f}"
              f"{'':>2}{(mb - ma) / ma * 100:>+6.0f}%")
        print(f"  {pad('── 전체 최대', 22)}{max(singles):>8.2f}{max(splits):>8.2f}"
              f"{max(splits) - max(singles):>+9.2f}")
    print()


def stage_table(rows: list[dict]) -> None:
    print("── 단계별 (초, 중앙값) " + "─" * 40)
    for mode, stages in (("single", SINGLE_STAGES), ("split", SPLIT_STAGES)):
        subset = [r for r in rows if r["mode"] == mode]
        if not subset:
            continue
        parts = []
        for name in stages:
            values = [r["stages"][name] for r in subset if r["stages"].get(name) is not None]
            if values:
                parts.append(f"{name} {statistics.median(values):.2f}")
        print(f"  {pad(mode, 8)}{'  '.join(parts)}")
    print()


QUALITY_ROWS = [
    ("모델이 쓴 메시지", "from_model", "ratio"),
    ("카테고리 폴백 아님", "source", "not_fallback"),
    ("메시지에 상대 이름", "name_in_message", "ratio"),
    ("메시지 길이(자)", "message_len", "median"),
    ("이유 길이(자)", "reason_len_mean", "median"),
    ("요약 길이(자)", "summary_len", "median"),
    ("카테고리 수", "category_count", "median"),
    ("존댓말 문장 비율", "polite_ratio", "median"),
    ("상품 건수", "product_count", "median"),
    ("[금지] 메시지에 카테고리", "category_leak", "sum"),
    ("[금지] 메시지에 금액", "money_in_message", "count"),
    ("[폴백] 요약이 기본문구", "summary_is_default", "count"),
]


def aggregate(values: list, how: str) -> str:
    clean = [v for v in values if v is not None]
    if not clean:
        return "—"
    if how == "ratio":
        return f"{sum(bool(v) for v in clean)}/{len(clean)}"
    if how == "not_fallback":
        return f"{sum(v == 'BEDROCK_CLAUDE' for v in clean)}/{len(clean)}"
    if how == "count":
        return str(sum(bool(v) for v in clean))
    if how == "sum":
        return str(sum(clean))
    return f"{statistics.median(clean):.2f}".rstrip("0").rstrip(".")


def agreement(rows: list[dict]) -> None:
    """같은 입력에서 두 모드가 고른 카테고리가 얼마나 겹치는지.

    분할은 1단계에서 이유 없이 카테고리만 고릅니다. 단일 호출은 2·3순위를 고를 때
    1순위의 이유를 보고 고르므로, 그 조건을 뺀 것이 선택을 바꾸는지 여기서 봅니다.
    1순위는 단일 호출에서도 어떤 이유보다 먼저 생성되므로 원래 같아야 합니다.
    """
    print("── 카테고리 일치 (같은 입력, 단일 vs 분할) " + "─" * 20)
    top1 = both = 0
    total = 0
    for label in dict.fromkeys(r["case"] for r in rows):
        pairs = []
        for index in range(len([r for r in rows if r["case"] == label and r["mode"] == "single"])):
            a = [r for r in rows if r["case"] == label and r["mode"] == "single"]
            b = [r for r in rows if r["case"] == label and r["mode"] == "split"]
            if index < len(a) and index < len(b) and not a[index]["error"] and not b[index]["error"]:
                pairs.append((a[index]["categories"], b[index]["categories"]))
        if not pairs:
            continue
        hits1 = sum(x[:1] == y[:1] for x, y in pairs)
        overlap = statistics.fmean(
            len(set(x) & set(y)) / max(len(set(x) | set(y)), 1) for x, y in pairs
        )
        top1 += hits1
        both += overlap * len(pairs)
        total += len(pairs)
        print(f"  {pad(label, 22)}1순위 {hits1}/{len(pairs)}   집합 겹침 {overlap:.0%}")
    if total:
        print(f"  {pad('── 전체', 22)}1순위 {top1}/{total}   집합 겹침 {both / total:.0%}")
    print()


def quality_table(rows: list[dict]) -> None:
    ok = [r for r in rows if not r["error"]]
    print("── 품질 (같은 입력, 규칙 판정) " + "─" * 32)
    print(f"  {pad('지표', 26)}{'단일':>10}{'분할':>10}")
    for title, key, how in QUALITY_ROWS:
        a = aggregate([r.get(key) for r in ok if r["mode"] == "single"], how)
        b = aggregate([r.get(key) for r in ok if r["mode"] == "split"], how)
        flag = "  ←" if a != b and title.startswith(("[금지]", "[폴백]")) else ""
        print(f"  {pad(title, 26)}{a:>10}{b:>10}{flag}")
    print()


def dump_outputs(rows: list[dict], limit: int) -> None:
    """문장은 사람이 읽어야 합니다. 케이스마다 첫 실행의 출력을 나란히 놓습니다."""
    print("── 출력 대조 " + "─" * 50)
    for label in dict.fromkeys(r["case"] for r in rows):
        pair = {}
        for mode in ("single", "split"):
            found = [r for r in rows if r["case"] == label and r["mode"] == mode and not r["error"]]
            if found:
                pair[mode] = found[0]
        if len(pair) < 2:
            continue
        print(f"\n  ▶ {label}")
        for mode, row in pair.items():
            print(f"    [{mode}] {' / '.join(row['categories'])}")
            print(f"      메시지: {row['message']}")
            print(f"      요약  : {row['summary']}")
            for reason in row["reasons"][:limit]:
                print(f"      이유  : {reason}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="추천 단일 호출 vs 분할 호출 A/B")
    parser.add_argument("--runs", type=int, default=3, help="케이스당 모드별 반복 (기본 3)")
    parser.add_argument("--search", action="store_true",
                        help="Tavily 검색 포함 종단 측정. 켜면 크레딧을 씁니다")
    parser.add_argument("--cases", type=int, default=0, help="앞에서 N개만 (0=전부)")
    parser.add_argument("--json", help="원자료 저장 경로")
    parser.add_argument("--reasons", type=int, default=1, help="출력 대조에 실을 이유 개수")
    args = parser.parse_args()

    if not args.search:
        settings.tavily_enabled = False
        settings.product_price_lookup_enabled = False

    cases = CASES[: args.cases] if args.cases else CASES
    print(f"backend  : {settings.model_backend} ({settings.bedrock_model_id})")
    print(f"검색     : {'포함 (Tavily 크레딧 사용)' if args.search else '제외 (모델 시간만)'}")
    print(f"케이스   : {len(cases)}종 × {args.runs}회 × 2모드 = "
          f"{len(cases) * args.runs * 2}회 호출\n")

    install_probes()
    rows = asyncio.run(run_all(args.runs, cases))

    print("=" * 64)
    latency_table(rows, args.runs)
    stage_table(rows)
    quality_table(rows)
    agreement(rows)
    dump_outputs(rows, args.reasons)

    if args.json:
        import json

        Path(args.json).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  원자료를 {args.json} 에 저장했습니다.\n")


if __name__ == "__main__":
    main()
