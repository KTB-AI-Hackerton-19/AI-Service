"""추천 근거를 조립합니다.

카테고리별 이유는 모델이 쓰지만, 여기 값들은 규칙에서 결정론적으로 나옵니다.
모델이 지어낸 설명이 아니라 실제로 적용된 계산이라 사용자에게 그대로 보여 줘도 됩니다.
"""

from app.schemas.recommendation import (
    Gender,
    ProductSuggestion,
    RecommendationRationale,
    SimpleGiftRecommendationRequest,
)

_GENDER_LABEL = {Gender.MALE: "남성", Gender.FEMALE: "여성"}
_RECORD_LABEL = {
    "gift": "받은 선물",
    "money": "받은 현금·부조금",
    "event_invitation": "받은 초대장",
    "receipt": "구매 영수증",
}


def price_range_basis(request: SimpleGiftRecommendationRequest, low: int, high: int) -> str:
    """가격 범위를 그렇게 잡은 근거를 문장으로 만듭니다."""
    if request.budget_min is not None or request.budget_max is not None:
        return f"사용자가 지정한 예산({low:,}원 ~ {high:,}원)을 그대로 따랐습니다."

    amounts = [a for a in request.received_amounts if a > 0]
    if len(amounts) > 1:
        return (
            f"{len(amounts)}명에게 {min(amounts):,}원 ~ {max(amounts):,}원을 받아, "
            f"가장 적게 준 분의 80%부터 가장 많이 준 분의 120%까지로 잡았습니다. "
            f"모두에게 같은 가격대를 권하면 한쪽에는 과하고 다른 쪽에는 모자랍니다."
        )
    return f"받은 금액 {request.gift_price:,}원의 80% ~ 120% 범위로 잡았습니다."


def inputs_used(request: SimpleGiftRecommendationRequest) -> list[str]:
    """추천에 실제로 반영된 입력만 나열합니다.

    반영되지 않은 것은 넣지 않습니다. 없는 근거를 있는 것처럼 보이면 안 됩니다.
    """
    used: list[str] = []
    if request.age is not None:
        used.append(f"나이 {request.age}세")
    if request.gender is not Gender.UNKNOWN:
        used.append(f"성별 {_GENDER_LABEL.get(request.gender, request.gender)}")
    if request.relationship:
        used.append(f"관계 {request.relationship}")
    if request.event:
        used.append(f"계기 {request.event}")
    if request.record_type in _RECORD_LABEL:
        used.append(_RECORD_LABEL[request.record_type])
    if request.budget_min is not None or request.budget_max is not None:
        used.append("사용자 지정 예산")
    elif request.received_amounts:
        used.append(f"받은 금액 {len(request.received_amounts)}건")
    else:
        used.append(f"받은 금액 {request.gift_price:,}원")
    if request.preferred_categories:
        used.append("사용자 지정 카테고리 " + ", ".join(request.preferred_categories))
    if request.interests:
        used.append("관심사 " + ", ".join(request.interests))
    if request.dislikes:
        used.append("기피 " + ", ".join(request.dislikes))
    return used


def category_basis(request: SimpleGiftRecommendationRequest, categories: list[str]) -> str:
    """카테고리를 그렇게 좁힌 근거."""
    listed = ", ".join(categories) or "없음"
    if request.preferred_categories:
        return f"사용자가 고른 카테고리 안에서만 골랐습니다: {listed}"
    hints = []
    if request.age is not None:
        hints.append("연령대")
    if request.gender is not Gender.UNKNOWN:
        hints.append("성별")
    if request.relationship:
        hints.append("관계")
    lead = "·".join(hints) or "받은 선물의 성격"
    return f"{lead}을(를) 고려해 {listed}을(를) 골랐습니다."


def product_basis(products: list[ProductSuggestion], low: int, high: int) -> str:
    """상품을 그렇게 고른 근거."""
    if not products:
        return "상품 검색 결과가 없어 카테고리와 가격대만 제안했습니다."

    verified = sum(1 for p in products if p.price_verified)
    in_range = sum(1 for p in products if p.price is not None and low <= p.price <= high)
    sources = sorted({p.source for p in products})
    return (
        f"{', '.join(sources)}에서 찾았습니다. "
        f"{len(products)}개 중 {verified}개는 상품 페이지에서 판매가를 확인했고, "
        f"{in_range}개가 {low:,}원 ~ {high:,}원 안에 듭니다."
    )


def warnings(products: list[ProductSuggestion], low: int, high: int) -> list[str]:
    """사용자가 알아야 할 한계를 모읍니다."""
    notes: list[str] = []
    unverified = [p for p in products if p.price is not None and not p.price_verified]
    if unverified:
        notes.append(
            f"{len(unverified)}개는 판매가를 상품 페이지에서 확인하지 못해 참고용 금액입니다."
        )
    no_price = [p for p in products if p.price is None]
    if no_price:
        notes.append(f"{len(no_price)}개는 가격을 확인하지 못했습니다. 링크에서 확인해 주세요.")
    over = [p for p in products if p.price is not None and not (low <= p.price <= high)]
    if over:
        notes.append(f"{len(over)}개는 제안 가격대를 벗어납니다.")
    return notes


def build(
    request: SimpleGiftRecommendationRequest,
    categories: list[str],
    products: list[ProductSuggestion],
    low: int,
    high: int,
) -> RecommendationRationale:
    """추천 근거 전체를 조립합니다."""
    return RecommendationRationale(
        price_range_basis=price_range_basis(request, low, high),
        inputs_used=inputs_used(request),
        category_basis=category_basis(request, categories),
        product_basis=product_basis(products, low, high),
        warnings=warnings(products, low, high),
    )
