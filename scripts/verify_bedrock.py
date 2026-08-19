"""Bedrock 백엔드를 실제 호출로 검증합니다.

앱의 실제 경로(``qwen_service`` / ``vlm_extraction_service``)를 그대로 통과시키므로,
여기서 통과하면 API 도 통과합니다. ``MODEL_BACKEND`` 설정을 따르기 때문에 bedrock
이외의 백엔드에도 그대로 씁니다.

사용법
    # .env 에 인증 정보를 넣은 뒤
    python scripts/verify_bedrock.py              # 전체
    python scripts/verify_bedrock.py --preflight  # 연결·권한만
    python scripts/verify_bedrock.py --recommend  # 추천만
    python scripts/verify_bedrock.py --vision     # 이미지 분석만
    python scripts/verify_bedrock.py --vision --image ./gift.png
"""

import argparse
import asyncio
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.schemas.recommendation import MessageSource, SimpleGiftRecommendationRequest
from app.services.recommendation_policy import ALLOWED_CATEGORIES

CASES = [
    ("단건", dict(gift_name="스타벅스 기프티콘 케이크", gift_price=35_000, age=29,
                 person_name="김민수", relationship="직장 동료")),
    ("나이·성별 없음", dict(gift_name="핸드크림 세트", gift_price=18_000)),
    ("여러 명", dict(gift_name="집들이 선물", gift_price=50_000, age=34,
                   received_amounts=[30_000, 50_000, 100_000],
                   received_people=["김민수", "이서연", "박준호"])),
    ("청첩장", dict(gift_name="결혼식 청첩장", gift_price=100_000, age=31,
                  person_name="이서연", relationship="대학 친구",
                  record_type="event_invitation")),
]


def banner() -> None:
    key = settings.bedrock_api_key
    print(f"backend : {settings.model_backend}")
    print(f"style   : {settings.bedrock_api_style}")
    print(f"region  : {settings.bedrock_region}")
    print(f"model   : {settings.bedrock_model_id}")
    print(f"auth    : {'API key(' + key[:4] + '****)' if key else 'SigV4 (AWS credential chain)'}")
    print(f"max_tok : {settings.bedrock_max_tokens}\n")


def preflight() -> bool:
    """인증·리전·모델 접근을 가장 짧은 호출로 확인합니다."""
    import anthropic

    from app.services import bedrock_client

    print("── 프리플라이트 " + "─" * 50)
    try:
        response = bedrock_client.get_client().messages.create(
            model=settings.bedrock_model_id,
            max_tokens=32,
            messages=[{"role": "user", "content": "한 문장으로 인사해줘."}],
        )
    except (anthropic.AnthropicError, bedrock_client.BedrockClientError) as exc:
        message = (
            bedrock_client.describe_failure(exc)
            if isinstance(exc, anthropic.AnthropicError)
            else str(exc)
        )
        print(f"  실패: {message}\n")
        print("  힌트: 403 이면 BEDROCK_API_STYLE(invoke/mantle) 과 콘솔의 Model access 를,")
        print("        'use case details' 가 보이면 Bedrock 콘솔에서 Anthropic 사용 사례")
        print("        양식 제출이 필요합니다(제출 후 약 15분).")
        return False
    print(f"  성공: {bedrock_client.extract_text(response).strip()[:60]}")
    print(f"  토큰: in={response.usage.input_tokens} out={response.usage.output_tokens}\n")
    return True


def recommend() -> bool:
    """추천 엔진을 앱과 동일한 경로로 호출합니다."""
    from app.services.qwen_service import RecommendationGenerationError, qwen_service

    print("── 추천 " + "─" * 57)
    ok = True
    for label, kwargs in CASES:
        request = SimpleGiftRecommendationRequest(**kwargs)
        start = time.time()
        try:
            result = qwen_service.recommend_simple(request)
        except RecommendationGenerationError as exc:
            print(f"  [{label}] 실패: {exc}")
            ok = False
            continue

        elapsed = time.time() - start
        fallback = result.source.endswith("_FALLBACK")
        categories = [c.category for c in result.categories]
        outside = [c for c in categories if c not in ALLOWED_CATEGORIES]
        # 정책이 직접 알려 줍니다. 예전에는 기본 문구와 문자열을 맞춰 봤는데,
        # 그 비교는 이름 교정(fix_person_name)이 끼면 어긋납니다.
        templated = result.message_source is not MessageSource.MODEL

        print(f"  [{label}] {elapsed:.1f}s  source={result.source}")
        print(f"    가격      : {result.recommended_price_min:,} ~ {result.recommended_price_max:,}원")
        print(f"    카테고리  : {categories}{'  <- 목록 밖 ' + str(outside) if outside else ''}")
        print(f"    메시지    : {len(result.suggested_message)}자"
              f"{'  (기본 문구로 대체됨: ' + result.message_source + ')' if templated else '  (모델 문장 사용)'}")
        if fallback:
            print("    경고      : JSON 파싱 실패로 안전 추천으로 대체되었습니다.")
        ok = ok and not fallback and not outside
    print()
    return ok


def sample_image() -> bytes:
    """검증용 카카오톡 선물하기 스타일 이미지를 만듭니다."""
    from PIL import Image, ImageDraw, ImageFont

    for path in ("/System/Library/Fonts/AppleSDGothicNeo.ttc",
                 "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            font_path = path
            break
    else:
        raise SystemExit("한글 폰트를 찾지 못했습니다. --image 로 직접 넣어 주세요.")

    image = Image.new("RGB", (720, 1000), "#b2c7d9")
    draw = ImageDraw.Draw(image)
    big = ImageFont.truetype(font_path, 34)
    mid = ImageFont.truetype(font_path, 27)
    small = ImageFont.truetype(font_path, 22)
    draw.rectangle([40, 60, 680, 120], fill="#ffffff")
    draw.text((60, 75), "김민수", font=mid, fill="#000000")
    draw.rounded_rectangle([60, 160, 660, 560], 20, fill="#ffffff")
    draw.text((90, 190), "선물이 도착했어요!", font=big, fill="#191919")
    draw.rounded_rectangle([90, 250, 630, 430], 12, fill="#f5f0e8")
    draw.text((120, 300), "스타벅스 카페 아메리카노 T", font=mid, fill="#191919")
    draw.text((120, 350), "+ 뉴욕 치즈케이크", font=mid, fill="#191919")
    draw.text((90, 460), "12,500원", font=big, fill="#d32f2f")
    draw.text((90, 510), "2026. 8. 15.  유효기간 90일", font=small, fill="#767676")
    draw.rounded_rectangle([60, 600, 560, 720], 20, fill="#ffffff")
    draw.text((90, 630), "생일 축하해! 늘 고마워 :)", font=mid, fill="#191919")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def vision(image_path: str | None) -> bool:
    """이미지 분석을 앱과 동일한 경로로 호출합니다."""
    from PIL import Image

    from app.services.image_loader import LoadedImage
    from app.services.vlm_service import VisionAnalysisError, vlm_extraction_service

    print("── 이미지 분석 " + "─" * 51)
    data = Path(image_path).read_bytes() if image_path else sample_image()
    with Image.open(io.BytesIO(data)) as opened:
        width, height = opened.size
        mime = f"image/{(opened.format or 'PNG').lower()}"
    image = LoadedImage(
        data=data, mime=mime, width=width, height=height, downloaded_bytes=len(data)
    )
    print(f"  입력: {image_path or '내장 샘플'} ({width}x{height}, {len(data):,} bytes)")

    start = time.time()
    try:
        result = asyncio.run(vlm_extraction_service.extract(image))
    except VisionAnalysisError as exc:
        print(f"  실패: {exc}\n")
        return False

    print(f"  {time.time() - start:.1f}s  "
          f"in={result.prompt_tokens} out={result.completion_tokens}토큰")
    for warning in result.warnings:
        print(f"  경고: {warning}")
    print(json.dumps(result.payload, ensure_ascii=False, indent=2))
    print()
    return not result.warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Bedrock 백엔드 실호출 검증")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--recommend", action="store_true")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--image", help="이미지 분석에 쓸 파일. 없으면 샘플을 만듭니다.")
    args = parser.parse_args()

    selected = args.preflight or args.recommend or args.vision
    banner()
    results = []
    if not selected or args.preflight:
        results.append(("프리플라이트", preflight()))
        if not results[-1][1] and not selected:
            print("프리플라이트가 실패해 나머지를 건너뜁니다.")
            sys.exit(1)
    if not selected or args.recommend:
        results.append(("추천", recommend()))
    if not selected or args.vision:
        results.append(("이미지 분석", vision(args.image)))

    print("─" * 66)
    for name, passed in results:
        print(f"  {name:12s} {'통과' if passed else '문제 있음'}")
    sys.exit(0 if all(p for _, p in results) else 1)


if __name__ == "__main__":
    main()
