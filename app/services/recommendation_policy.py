"""Qwen 출력을 Giftie의 가격·카테고리 안전 정책에 맞게 보정합니다."""

from typing import Any

from app.schemas.recommendation import SimpleGiftRecommendationRequest

CATEGORY_ALIASES = {
    "식품/음료": "식품·디저트",
    "음식": "식품·디저트",
    "식품": "식품·디저트",
    "디저트": "식품·디저트",
    "커피": "커피·차",
    "디지털 기기": "디지털 액세서리",
    "전자기기": "디지털 액세서리",
    "패션": "패션·잡화",
    "문화": "문화·취미",
    "취미": "문화·취미",
}
SAFE_EXAMPLES = {
    "식품·디저트": ["프리미엄 디저트 세트", "제철 과일 세트"],
    "커피·차": ["스페셜티 드립백 세트", "프리미엄 티 세트"],
    "생활용품": ["고급 타월 세트", "보온·보냉 텀블러"],
    "패션·잡화": ["카드지갑", "파우치·에코백"],
    "문화·취미": ["도서·문구 세트", "전시·공연 관람권"],
    "건강·웰니스": ["건강 간식 세트", "마사지·스트레칭 소품"],
    "꽃·식물": ["미니 꽃다발", "관리하기 쉬운 화분"],
    "상품권": ["외식 상품권", "문화생활 상품권"],
    "디지털 액세서리": ["휴대폰 거치대", "충전 케이블 세트"],
    "유아·아동": ["연령별 그림책", "창의 놀이 세트"],
}


def normalize_recommendation(
    request: SimpleGiftRecommendationRequest,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    """가격을 80~120%로 고정하고 허용된 카테고리와 예시만 반환합니다."""
    minimum = max(int(request.gift_price * 0.8 / 1000) * 1000, 1_000)
    maximum = max(int(request.gift_price * 1.2 / 1000) * 1000, 1_000)
    categories: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_categories = parsed.get("categories", [])
    if not isinstance(raw_categories, list):
        raw_categories = []

    for item in raw_categories:
        if not isinstance(item, dict):
            continue
        raw_category = str(item.get("category", "")).strip()
        category = CATEGORY_ALIASES.get(raw_category, raw_category)
        if category not in SAFE_EXAMPLES or category in seen:
            continue
        seen.add(category)
        try:
            score = int(item.get("score", 50))
        except (TypeError, ValueError):
            score = 50
        categories.append(
            {
                "category": category,
                "score": min(max(score, 0), 100),
                "reason": str(
                    item.get("reason", "관계와 가격대를 고려한 추천입니다.")
                )[:300],
                "product_examples": SAFE_EXAMPLES[category],
            }
        )

    if not categories:
        categories.append(
            {
                "category": "상품권",
                "score": 70,
                "reason": "취향 정보가 부족할 때 선택 실패 가능성이 낮습니다.",
                "product_examples": SAFE_EXAMPLES["상품권"],
            }
        )
    suggested_message = str(parsed.get("suggested_message", "")).strip()
    # 소형 모델이 지나치게 짧거나 문맥이 빈약한 문장을 만들면 안정적인
    # 장문 템플릿으로 교체해 사용자에게 항상 충분한 메시지를 제공합니다.
    if len(suggested_message) < 120:
        suggested_message = _default_message(request)

    return {
        "recommended_price_min": minimum,
        "recommended_price_max": maximum,
        "categories": categories[:3],
        "summary": str(
            parsed.get("summary", "받은 선물과 가격대를 고려한 답례 추천입니다.")
        )[:500],
        "suggested_message": suggested_message[:500],
    }


def _default_message(request: SimpleGiftRecommendationRequest) -> str:
    """모델 메시지가 없거나 너무 짧을 때 사용할 충분히 구체적인 기본 문구."""
    greeting = f"{request.person_name}님, " if request.person_name else ""
    relationship_context = (
        f"늘 {request.relationship}로서 따뜻하게 챙겨주시는 마음이 느껴져서"
        if request.relationship
        else "세심하게 챙겨주신 마음이 느껴져서"
    )
    return (
        f"{greeting}지난번에 선물해 주신 {request.gift_name} 정말 고마웠어요. "
        f"{relationship_context} 선물을 받을 때부터 기분이 참 좋았어요. "
        "덕분에 잘 사용하고 있고, 볼 때마다 감사한 마음이 들어요. "
        "저도 그 마음을 기억하고 작은 정성을 준비했으니 부담 없이 기쁘게 받아주세요!"
    )
