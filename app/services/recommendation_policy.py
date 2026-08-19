"""Qwen 출력을 Giftie의 가격·카테고리 안전 정책에 맞게 보정합니다."""

from typing import Any

from app.schemas.recommendation import SimpleGiftRecommendationRequest
from app.services.price_policy import calculate_recommended_price_range

CATEGORY_ALIASES = {
    "식품/음료": "식품·디저트",
    "음식": "식품·디저트",
    "식품": "식품·디저트",
    "디저트": "식품·디저트",
    "커피": "커피·차",
    "디지털 기기": "디지털 액세서리",
    "전자기기": "디지털 액세서리",
    "패션": "패션·잡화",
    "화장품": "뷰티·화장품",
    "화장품·스킨케어": "뷰티·화장품",
    "스킨케어": "뷰티·화장품",
    "뷰티": "뷰티·화장품",
    "향수": "뷰티·화장품",
    "문화": "문화·취미",
    "취미": "문화·취미",
}
SAFE_EXAMPLES = {
    "식품·디저트": ["프리미엄 디저트 세트", "제철 과일 세트"],
    "커피·차": ["스페셜티 드립백 세트", "프리미엄 티 세트"],
    "생활용품": ["고급 타월 세트", "보온·보냉 텀블러"],
    "뷰티·화장품": ["핸드크림·립밤 세트", "향수 미니어처 세트"],
    "패션·잡화": ["카드지갑", "파우치·에코백"],
    "문화·취미": ["도서·문구 세트", "전시·공연 관람권"],
    "건강·웰니스": ["건강 간식 세트", "마사지·스트레칭 소품"],
    "꽃·식물": ["미니 꽃다발", "관리하기 쉬운 화분"],
    "상품권": ["외식 상품권", "문화생활 상품권"],
    "디지털 액세서리": ["휴대폰 거치대", "충전 케이블 세트"],
    "유아·아동": ["연령별 그림책", "창의 놀이 세트"],
}


ALLOWED_CATEGORIES = tuple(SAFE_EXAMPLES)
"""추천에 허용된 카테고리. 프롬프트와 구조화 출력 스키마가 이 목록 하나를 공유합니다."""

_PRICE_FLOOR_RATIO = 0.8
_PRICE_CEILING_RATIO = 1.2
_MIN_PRICE = 1_000


def price_range(request: SimpleGiftRecommendationRequest) -> tuple[int, int]:
    """답례 가격 범위를 정합니다.

    한 건이면 받은 금액의 80~120% 입니다. 여러 사람에게 받았다면 각 금액의
    최저 80% 부터 최고 120% 까지로 넓힙니다. 축의금을 5만원 준 사람과 20만원 준 사람에게
    같은 가격대를 권하면 한쪽에는 과하고 다른 쪽에는 모자라기 때문입니다.
    """
    # 사용자가 예산을 직접 지정했으면 그대로 씁니다. 받은 금액에서 유추할 이유가 없습니다.
    if request.budget_min is not None or request.budget_max is not None:
        minimum = max(request.budget_min or _MIN_PRICE, _MIN_PRICE)
        maximum = max(request.budget_max or minimum, minimum)
        return minimum, maximum

    amounts = [a for a in request.received_amounts if a > 0] or [request.gift_price]
    minimum = max(int(min(amounts) * _PRICE_FLOOR_RATIO / 1000) * 1000, _MIN_PRICE)
    maximum = max(int(max(amounts) * _PRICE_CEILING_RATIO / 1000) * 1000, _MIN_PRICE)
    return minimum, max(minimum, maximum)


def normalize_recommendation(
    request: SimpleGiftRecommendationRequest,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    """가격을 안전 범위로 고정하고 허용된 카테고리와 예시만 반환합니다."""
    minimum, maximum = price_range(request)
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
                "search_query": str(
                    item.get(
                        "search_query",
                        f"{category} 답례 선물 {minimum}원 {maximum}원",
                    )
                )[:200],
            }
        )

    allowed = {CATEGORY_ALIASES.get(c, c) for c in request.preferred_categories}
    if allowed:
        narrowed = [c for c in categories if c["category"] in allowed]
        if narrowed:
            categories = narrowed

    if not categories:
        categories.append(
            {
                "category": "상품권",
                "score": 70,
                "reason": "취향 정보가 부족할 때 선택 실패 가능성이 낮습니다.",
                "product_examples": SAFE_EXAMPLES["상품권"],
                "search_query": f"답례 상품권 {minimum}원 {maximum}원",
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
    """모델 메시지가 없거나 너무 짧을 때 사용할 기본 문구.

    받은 것의 종류에 따라 문장이 달라야 합니다. 청첩장에 "선물해 주신 청첩장 고마웠어요"
    라고 쓰면 어색하고, 여러 사람에게 받았는데 한 사람 이름을 넣으면 나머지에게는 못 씁니다.
    """
    if request.record_type == "event_invitation":
        return _invitation_message(request)
    if len(request.received_amounts) > 1:
        return _group_message(request)
    return _single_gift_message(request)


def _single_gift_message(request: SimpleGiftRecommendationRequest) -> str:
    """한 사람에게 선물을 받은 기본 경우."""
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


def _invitation_message(request: SimpleGiftRecommendationRequest) -> str:
    """청첩장·초대장을 받은 경우. 사용자는 주인공이 아니라 하객입니다."""
    greeting = f"{request.person_name}님, " if request.person_name else ""
    occasion = request.event or "좋은 소식"
    return (
        f"{greeting}{occasion} 소식 전해 주셔서 정말 기뻤어요. "
        "정성스럽게 준비하신 초대장 잘 받았고, 소중한 자리에 함께할 수 있어 영광입니다. "
        "그날까지 준비하시느라 바쁘시겠지만 건강 꼭 챙기시고, "
        "좋은 모습으로 뵙겠습니다. 진심으로 축하드려요!"
    )


def _group_message(request: SimpleGiftRecommendationRequest) -> str:
    """여러 사람에게 받은 경우. 특정 이름 없이 두루 쓸 수 있어야 합니다."""
    occasion_context = f"{request.event}에 " if request.event else ""
    return (
        f"{occasion_context}보내주신 따뜻한 마음 덕분에 정말 큰 힘을 얻었습니다. "
        "바쁘신 중에도 이렇게 챙겨 주셔서 진심으로 감사드려요. "
        "덕분에 잘 지내고 있고, 그 마음 오래 기억하겠습니다. "
        "작은 정성을 준비했으니 부담 없이 기쁘게 받아주세요!"
    )
