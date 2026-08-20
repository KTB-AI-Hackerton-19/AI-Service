"""이미지 정보 추출과 답례 추천의 지연 시간을 실제 호출로 측정합니다.

앱이 실제로 타는 경로(``image_analysis_service.analyze`` /
``recommendation_preparation_service.prepare``)를 그대로 여러 번 돌리고,
각 단계(모델 호출, Tavily 검색, 이미지 다운로드)가 몇 초를 썼는지 나눠서 보여 줍니다.
어디를 줄여야 하는지는 총 시간이 아니라 이 구성비에서만 보입니다.

주의: 실호출이므로 Bedrock 토큰과 Tavily 크레딧(검색 1회 = 1크레딧)을 씁니다.
기본값은 3회 실행 / 추천 케이스 1건으로 잡아 두었습니다.

사용법
    python scripts/benchmark_latency.py                   # 이미지 분석 + 추천, 3회씩
    python scripts/benchmark_latency.py --runs 5
    python scripts/benchmark_latency.py --vision --image ./gift.png
    python scripts/benchmark_latency.py --vision --url "https://...s3...png"
    python scripts/benchmark_latency.py --recommend --all-cases
    python scripts/benchmark_latency.py --no-search      # 모델 시간만 (Tavily 미사용)
    python scripts/benchmark_latency.py --json out.json  # 원자료 저장
"""

import argparse
import asyncio
import inspect
import io
import json
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import settings
from app.schemas.agent import GiftData, InputCategory

# 추천 지연은 입력에 따라 달라집니다(여러 명이면 프롬프트가 길어지고, 청첩장은
# 카테고리가 달라 검색어도 달라집니다). verify_bedrock.py 와 같은 케이스를 씁니다.
CASES: list[tuple[str, dict]] = [
    ("단건", dict(gift_name="스타벅스 기프티콘 케이크", gift_price=35_000, age=29,
                 person_name="김민수", relationship="직장 동료")),
    ("나이·성별 없음", dict(gift_name="핸드크림 세트", gift_price=18_000)),
    ("여러 명", dict(gift_name="집들이 선물", gift_price=50_000, age=34,
                   person_name="김민수", relationship="대학 친구")),
    ("청첩장", dict(gift_name="결혼식 청첩장", gift_price=100_000, age=31,
                  person_name="이서연", relationship="대학 친구",
                  record_type="event_invitation")),
]


# ─────────────────────────── 계측 도구 ───────────────────────────


@dataclass
class Span:
    """한 번의 호출이 차지한 구간. 동시에 도는 호출을 겹쳐 보려고 끝점을 둘 다 둡니다."""

    stage: str
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass
class Recorder:
    """한 번의 실행에서 나온 모든 구간."""

    spans: list[Span] = field(default_factory=list)

    def wall(self, stage: str) -> float | None:
        """그 단계가 실제로 붙잡은 벽시계 시간.

        판매가 검색처럼 여러 호출이 ``asyncio.gather`` 로 겹쳐 도는 단계는 합이 아니라
        가장 이른 시작부터 가장 늦은 끝까지가 실제 지연입니다.
        """
        spans = [s for s in self.spans if s.stage == stage]
        if not spans:
            return None
        return max(s.end for s in spans) - min(s.start for s in spans)

    def count(self, stage: str) -> int:
        return sum(1 for s in self.spans if s.stage == stage)


class Probe:
    """현재 실행 중인 Recorder 를 가리키는 홀더. 패치는 한 번, 수집은 실행마다."""

    def __init__(self) -> None:
        self.current: Recorder | None = None

    def start(self) -> Recorder:
        self.current = Recorder()
        return self.current

    def record(self, stage: str, start: float, end: float) -> None:
        if self.current is not None:
            self.current.spans.append(Span(stage, start, end))


probe = Probe()


def instrument(owner: object, attr: str, stage: str) -> None:
    """``owner.attr`` 호출 시간을 ``stage`` 이름으로 기록하도록 감쌉니다.

    앱 코드에는 손대지 않습니다. 측정을 위해 서비스에 타이머를 심으면 그 타이머가
    운영 코드에 남고, 지금 재고 싶은 경계와 나중에 재고 싶은 경계는 다릅니다.
    """
    original = getattr(owner, attr)

    if inspect.iscoroutinefunction(original):
        async def wrapper(*args, **kwargs):
            begin = time.perf_counter()
            try:
                return await original(*args, **kwargs)
            finally:
                probe.record(stage, begin, time.perf_counter())
    else:
        def wrapper(*args, **kwargs):
            begin = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                probe.record(stage, begin, time.perf_counter())

    setattr(owner, attr, wrapper)


VISION_STAGES = ["이미지 다운로드", "VLM 추출", "판매가 검색"]
RECOMMEND_STAGES = ["추천 모델", "상품 검색"]


def install_probes() -> None:
    """측정 지점을 심습니다. 앱이 실제로 부르는 그 객체를 감쌉니다."""
    from app.services import product_search as product_search_module
    from app.services.image_loader import image_loader
    from app.services.product_search import product_search
    from app.services.qwen_service import qwen_service
    from app.services.vlm_service import vlm_extraction_service

    instrument(image_loader, "load", "이미지 다운로드")
    instrument(vlm_extraction_service, "extract", "VLM 추출")
    # image_analysis 는 모듈 속성으로 부르므로 모듈 쪽을 감쌉니다.
    instrument(product_search_module, "lookup_price", "판매가 검색")
    instrument(qwen_service, "recommend_simple", "추천 모델")
    instrument(product_search, "search", "상품 검색")


# ─────────────────────────── 통계 ───────────────────────────


def percentile(values: list[float], ratio: float) -> float:
    """정렬 후 최근접 순위법. 실행 횟수가 적어 보간은 의미가 없습니다."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(ratio * len(ordered) + 0.5) - 1))
    return ordered[index]


def summarize(label: str, samples: list[float]) -> dict:
    return {
        "stage": label,
        "n": len(samples),
        "min": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "p95": percentile(samples, 0.95),
        "max": max(samples),
    }


def width(text: str) -> int:
    """한글은 터미널에서 두 칸을 차지합니다. len 으로 맞추면 표가 어긋납니다."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, size: int) -> str:
    return text + " " * max(0, size - width(text))


def print_table(title: str, rows: list[dict], total_mean: float | None) -> None:
    print(f"── {title} " + "─" * max(0, 58 - width(title)))
    header = f"  {pad('단계', 16)}{'n':>3} {'최소':>7}{'중앙':>7}{'평균':>7}{'p95':>8}{'최대':>7}"
    print(header + ("    비중" if total_mean else ""))
    for row in rows:
        line = (f"  {pad(row['stage'], 16)}{row['n']:>3}"
                f"{row['min']:>8.2f}{row['median']:>7.2f}{row['mean']:>7.2f}"
                f"{row['p95']:>8.2f}{row['max']:>7.2f}")
        if total_mean:
            line += f"   {row['mean'] / total_mean * 100:5.1f}%"
        print(line)
    print()


# ─────────────────────────── 이미지 분석 ───────────────────────────


def sample_image_bytes() -> bytes:
    """검증 스크립트와 같은 카카오톡 선물하기 스타일 샘플을 씁니다."""
    from verify_bedrock import sample_image

    return sample_image()


def use_local_image(data: bytes) -> tuple[int, int, int]:
    """로컬 파일을 쓸 때 다운로드 단계만 고정 결과로 바꿉니다.

    S3 presigned URL 이 없어도 VLM 지연을 잴 수 있어야 합니다. 대신 이 실행의
    '이미지 다운로드' 는 0 이 되므로 표에서 빠집니다.
    """
    from PIL import Image

    from app.services import image_loader as loader_module
    from app.services.image_loader import LoadedImage

    with Image.open(io.BytesIO(data)) as opened:
        width, height = opened.size
        mime = f"image/{(opened.format or 'PNG').lower()}"
    loaded = LoadedImage(
        data=data, mime=mime, width=width, height=height, downloaded_bytes=len(data)
    )

    async def fake_load(image_url: str) -> LoadedImage:
        return loaded

    loader_module.image_loader.load = fake_load
    return width, height, len(data)


async def measure_vision(runs: int, url: str, category: InputCategory) -> list[dict]:
    """``/from-image`` 앞단(이미지 → 선물데이터)을 runs 회 실행합니다."""
    from app.services.tasks.image_analysis import image_analysis_service

    measurements = []
    for index in range(1, runs + 1):
        recorder = probe.start()
        begin = time.perf_counter()
        try:
            gift_data = await image_analysis_service.analyze(url, category)
            total = time.perf_counter() - begin
            detail = f"{gift_data.gift_name[:20]} / {gift_data.gift_price or '금액미상'}"
        except Exception as exc:  # 실패한 실행도 지연은 지연입니다. 남기고 계속합니다.
            total = time.perf_counter() - begin
            detail = f"실패: {type(exc).__name__}: {exc}"
        stages = {name: recorder.wall(name) for name in VISION_STAGES}
        counts = {name: recorder.count(name) for name in VISION_STAGES}
        measurements.append({"run": index, "total": total, "stages": stages,
                             "counts": counts, "detail": detail})
        parts = "  ".join(
            f"{name} {value:.2f}s" + (f"×{counts[name]}" if counts[name] > 1 else "")
            for name, value in stages.items() if value is not None
        )
        print(f"  [{index}/{runs}] 총 {total:6.2f}s   {parts}")
        print(f"           → {detail}")
    print()
    return measurements


# ─────────────────────────── 추천 ───────────────────────────


async def measure_recommend(runs: int, cases: list[tuple[str, dict]]) -> list[dict]:
    """``/from-image``·``/from-gift-data`` 의 추천 작업을 runs 회 실행합니다."""
    from app.services.tasks.recommendation import recommendation_preparation_service

    measurements = []
    for label, kwargs in cases:
        gift_data = GiftData(**kwargs)
        for index in range(1, runs + 1):
            recorder = probe.start()
            begin = time.perf_counter()
            try:
                info = await recommendation_preparation_service.prepare(gift_data)
                total = time.perf_counter() - begin
                result = info.recommend_gift
                detail = (f"{result.recommended_price_min:,}~{result.recommended_price_max:,}원  "
                          f"상품 {len(result.products)}건  source={result.source}")
            except Exception as exc:
                total = time.perf_counter() - begin
                detail = f"실패: {type(exc).__name__}: {exc}"
            stages = {name: recorder.wall(name) for name in RECOMMEND_STAGES}
            counts = {name: recorder.count(name) for name in RECOMMEND_STAGES}
            measurements.append({"case": label, "run": index, "total": total,
                                 "stages": stages, "counts": counts, "detail": detail})
            parts = "  ".join(
                f"{name} {value:.2f}s" for name, value in stages.items() if value is not None
            )
            print(f"  [{label} {index}/{runs}] 총 {total:6.2f}s   {parts}")
            print(f"           → {detail}")
    print()
    return measurements


# ─────────────────────────── 보고 ───────────────────────────


def report(title: str, measurements: list[dict], stage_names: list[str],
           budget: float | None) -> dict | None:
    if not measurements:
        return None
    totals = [m["total"] for m in measurements]
    total_row = summarize("전체", totals)
    rows = [total_row]
    for name in stage_names:
        samples = [m["stages"][name] for m in measurements if m["stages"].get(name) is not None]
        if samples:
            row = summarize(name, samples)
            row["n"] = len(samples)
            rows.append(row)
    print_table(title, rows, total_row["mean"])
    if budget is not None:
        over = [t for t in totals if t > budget]
        verdict = "초과 없음" if not over else f"{len(over)}/{len(totals)}회 초과"
        print(f"  타임아웃 예산 {budget:.0f}s 대비: 최대 {max(totals):.2f}s — {verdict}\n")
    return {"rows": rows, "totals": totals}


def banner(args) -> None:
    print(f"backend  : {settings.model_backend}  ({settings.bedrock_model_id})")
    print(f"region   : {settings.bedrock_region}")
    print(f"tavily   : {'on' if settings.tavily_enabled else 'off'}"
          f"  (판매가 검색 {'on' if settings.product_price_lookup_enabled else 'off'})")
    print(f"runs     : {args.runs}회\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="이미지 추출·답례 추천 지연 시간 측정")
    parser.add_argument("--runs", type=int, default=3, help="측정 반복 횟수 (기본 3)")
    parser.add_argument("--vision", action="store_true", help="이미지 분석만")
    parser.add_argument("--recommend", action="store_true", help="추천만")
    parser.add_argument("--image", help="이미지 분석에 쓸 로컬 파일. 없으면 샘플을 만듭니다.")
    parser.add_argument("--url", help="실제 presigned URL. 다운로드 시간까지 포함해 잽니다.")
    parser.add_argument("--category", choices=["gift", "occasion"], default="gift")
    parser.add_argument("--all-cases", action="store_true", help="추천 케이스 4종 전부")
    parser.add_argument("--no-search", action="store_true", help="Tavily 없이 모델 시간만")
    parser.add_argument("--json", help="측정 원자료를 저장할 경로")
    args = parser.parse_args()

    if args.no_search:
        settings.tavily_enabled = False
        settings.product_price_lookup_enabled = False

    selected = args.vision or args.recommend
    run_vision = not selected or args.vision
    run_recommend = not selected or args.recommend

    banner(args)
    install_probes()

    vision_runs: list[dict] = []
    recommend_runs: list[dict] = []

    if run_vision:
        if args.url:
            print("── 이미지 분석 (다운로드 포함) " + "─" * 30)
            url = args.url
        else:
            data = Path(args.image).read_bytes() if args.image else sample_image_bytes()
            width, height, size = use_local_image(data)
            print("── 이미지 분석 " + "─" * 45)
            print(f"  입력: {args.image or '내장 샘플'} ({width}x{height}, {size:,} bytes)"
                  "  — 다운로드 단계 제외")
            url = "https://benchmark.local/sample.png"
        vision_runs = asyncio.run(measure_vision(args.runs, url, InputCategory(args.category)))

    if run_recommend:
        cases = CASES if args.all_cases else CASES[:1]
        print("── 답례 추천 " + "─" * 47)
        recommend_runs = asyncio.run(measure_recommend(args.runs, cases))

    print("=" * 62)
    print("요약 (초)\n")
    vision_summary = report("이미지 정보 추출", vision_runs, VISION_STAGES,
                            settings.image_analysis_timeout_seconds)
    recommend_summary = report("답례 추천", recommend_runs, RECOMMEND_STAGES,
                               settings.task_timeout_seconds)

    if vision_summary and recommend_summary:
        # /from-image 는 이미지 분석이 끝난 뒤 네 작업이 동시에 돕니다. 넷 중 추천이
        # 압도적으로 느리므로 사용자 체감 지연은 사실상 이 둘의 합입니다.
        mean = vision_summary["rows"][0]["mean"] + recommend_summary["rows"][0]["mean"]
        worst = max(vision_summary["totals"]) + max(recommend_summary["totals"])
        print(f"  /from-image 예상 응답 시간: 평균 {mean:.2f}s, 최악 {worst:.2f}s")
        print(f"  (예산 {settings.image_analysis_timeout_seconds:.0f}s + "
              f"{settings.task_timeout_seconds:.0f}s = "
              f"{settings.image_analysis_timeout_seconds + settings.task_timeout_seconds:.0f}s)\n")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"vision": vision_runs, "recommend": recommend_runs},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  원자료를 {args.json} 에 저장했습니다.\n")


if __name__ == "__main__":
    main()
