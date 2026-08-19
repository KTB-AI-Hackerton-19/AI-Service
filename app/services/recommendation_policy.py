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
    return {
        "recommended_price_min": minimum,
        "recommended_price_max": maximum,
        "categories": categories[:3],
        "summary": str(
            parsed.get("summary", "받은 선물과 가격대를 고려한 답례 추천입니다.")
        )[:500],
    }
