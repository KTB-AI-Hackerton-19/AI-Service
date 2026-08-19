"""추천 근거를 조립합니다.

카테고리별 이유는 모델이 쓰지만, 여기 값들은 규칙에서 결정론적으로 나옵니다.
모델이 지어낸 설명이 아니라 실제로 적용된 계산이라 사용자에게 그대로 보여 줘도 됩니다.

그래서 여기 문장은 실제 결과와 반드시 일치해야 합니다. 좁히기에 실패했는데
"고른 카테고리 안에서만 골랐습니다" 라고 쓰면 계산이 아니라 거짓말이 됩니다.
"""

from app.schemas.recommendation import (
    Gender,
    ProductSuggestion,
    RecommendationRationale,
    SimpleGiftRecommendationRequest,
)
from app.services.price_policy import PRICE_CEILING_RATIO, PRICE_FLOOR_RATIO
from app.services.recommendation_policy import (
    CATEGORY_ALIASES,
    object_particle,
    shipped_categories,
)

_GENDER_LABEL = {Gender.MALE: "남성", Gender.FEMALE: "여성"}
_RECORD_LABEL = {
    "gift": "받은 선물",
    "money": "받은 현금·부조금",
    "event_invitation": "받은 초대장",
    "receipt": "구매 영수증",
}

_FLOOR_PERCENT = round(PRICE_FLOOR_RATIO * 100)
_CEILING_PERCENT = round(PRICE_CEILING_RATIO * 100)


def price_range_basis(request: SimpleGiftRecommendationRequest, low: int, high: int) -> str:
    """가격 범위를 그렇게 잡은 근거를 문장으로 만듭니다.

    비율만 말하고 끝내면 안 됩니다. 실제 값은 금액대에 맞는 단위로 넓혀 떨어뜨리므로,
    80%·120% 원값과 최종 값을 함께 보여 줘야 화면의 숫자와 설명이 맞습니다.
    """
    if request.budget_min is not None or request.budget_max is not None:
        if request.budget_min is not None and low > request.budget_min:
            return (
                f"사용자가 지정한 예산 하한 {request.budget_min:,}원으로는 답례를 고르기 어려워 "
                f"{low:,}원까지 올려 {low:,}원 ~ {high:,}원으로 잡았습니다."
            )
        if request.budget_min is None:
            return (
                f"사용자가 지정한 예산 상한 {high:,}원에 맞추고, "
                f"하한은 지정이 없어 {low:,}원으로 뒀습니다."
            )
        return f"사용자가 지정한 예산({low:,}원 ~ {high:,}원)을 그대로 따랐습니다."

    amounts = [a for a in request.received_amounts if a > 0] or [request.gift_price]
    if len(amounts) > 1:
        return (
            f"{len(amounts)}명에게 {min(amounts):,}원 ~ {max(amounts):,}원을 받아, "
            f"가장 적게 준 분의 {_FLOOR_PERCENT}%({int(min(amounts) * PRICE_FLOOR_RATIO):,}원)부터 "
            f"가장 많이 준 분의 {_CEILING_PERCENT}%({int(max(amounts) * PRICE_CEILING_RATIO):,}원)까지를 "
            f"고르기 쉬운 단위로 넓혀 {low:,}원 ~ {high:,}원으로 잡았습니다."
        )
    amount = amounts[0]
    return (
        f"받은 금액 {amount:,}원의 {_FLOOR_PERCENT}%({int(amount * PRICE_FLOOR_RATIO):,}원) ~ "
        f"{_CEILING_PERCENT}%({int(amount * PRICE_CEILING_RATIO):,}원)를 고르기 쉬운 단위로 넓혀 "
        f"{low:,}원 ~ {high:,}원으로 잡았습니다."
    )


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


def category_basis(
    request: SimpleGiftRecommendationRequest,
    categories: list[str],
    products: list[ProductSuggestion] | tuple[()] = (),
) -> str:
    """카테고리를 그렇게 좁힌 근거.

    지정 카테고리로 좁히다가 결과가 비면 정책이 모델 카테고리를 그대로 내보냅니다
    (``recommendation_policy.normalize_recommendation``). 그때도 "고른 카테고리
    안에서만 골랐습니다" 라고 쓰면 화면에 없는 카테고리를 있다고 말하는 셈입니다.

    "셋을 골랐습니다" 도 같은 종류의 과장입니다. 실측에서 커피·차·식품·디저트·
    생활용품 셋을 골랐다고 써 놓고 화면에는 생활용품 한 건만 나갔습니다. 상품이
    실제로 나온 카테고리가 고른 것과 다르면 그 사실까지 적습니다.
    """
    listed = ", ".join(categories) or "없음"
    if request.preferred_categories:
        wanted = {CATEGORY_ALIASES.get(c, c) for c in request.preferred_categories}
        if categories and all(c in wanted for c in categories):
            base = f"사용자가 고른 카테고리 안에서만 골랐습니다: {listed}"
        else:
            base = (
                f"사용자가 고른 카테고리({', '.join(request.preferred_categories)})에서는 "
                f"추천할 만한 것을 찾지 못해 받은 선물의 성격에 맞춰 골랐습니다: {listed}"
            )
    else:
        hints = []
        if request.age is not None:
            hints.append("연령대")
        if request.gender is not Gender.UNKNOWN:
            hints.append("성별")
        if request.relationship:
            hints.append("관계")
        lead = "·".join(hints) or "받은 선물의 성격"
        base = (
            f"{lead}{object_particle(lead)} 고려해 {listed}{object_particle(listed)} 골랐습니다."
        )
    return _with_shipped_categories(base, categories, products)


def _with_shipped_categories(
    base: str,
    categories: list[str],
    products: list[ProductSuggestion] | tuple[()],
) -> str:
    """고른 카테고리와 상품이 나온 카테고리가 다르면 그 사실을 덧붙입니다.

    상품이 하나도 없을 때는 붙이지 않습니다. ``product_basis`` 가 "검색 결과가
    없어 카테고리와 가격대만 제안했습니다" 라고 이미 말합니다.
    """
    shipped = shipped_categories(products)
    if not shipped or set(shipped) == set(categories):
        return base
    separator = " " if base.endswith(".") else ". "
    return f"{base}{separator}이 가격대에서 상품이 나온 것은 {', '.join(shipped)}입니다."


def product_basis(
    products: list[ProductSuggestion], low: int, high: int, examined: int = 0
) -> str:
    """상품을 그렇게 고른 근거.

    Args:
        examined: 가격 심사까지 간 후보 수(``product_search.SearchStats.examined``).
            상품 0건의 이유가 "검색이 비었다" 인지 "찾았지만 가격이 맞지 않았다"
            인지가 이 값으로 갈립니다. 둘을 같은 문장으로 말하면 실측 gift 처럼
            후보 9건을 찾아 놓고 "검색 결과가 없어" 라고 쓰게 됩니다.
    """
    if not products:
        if examined:
            return (
                f"상품 후보 {examined}개를 찾았지만 {low:,}원 ~ {high:,}원에 맞는 판매가를 "
                "확인하지 못해 카테고리와 가격대만 제안했습니다."
            )
        return "상품 검색 결과가 없어 카테고리와 가격대만 제안했습니다."

    verified = sum(1 for p in products if p.price_verified)
    in_range = sum(1 for p in products if p.price is not None and low <= p.price <= high)
    sources = sorted({p.source for p in products})
    return (
        f"{', '.join(sources)}에서 찾았습니다. "
        f"{len(products)}개 중 {verified}개는 상품 페이지에서 판매가를 확인했고, "
        f"{in_range}개가 {low:,}원 ~ {high:,}원 안에 듭니다."
    )


def warnings(
    products: list[ProductSuggestion], low: int, high: int, examined: int = 0
) -> list[str]:
    """사용자가 알아야 할 한계를 모읍니다."""
    notes: list[str] = []
    # 상품 0건은 화면에서 가장 설명이 필요한 상태입니다. product_basis 가 계산을
    # 말한다면 여기서는 사용자가 다음에 무엇을 하면 되는지를 말합니다.
    if not products:
        notes.append(
            "제안 가격대에 맞는 상품을 확인하지 못해 이번에는 상품을 보여 드리지 못했습니다. "
            "카테고리와 가격대를 참고해 직접 골라 주세요."
            if examined
            else "상품을 찾지 못했습니다. 카테고리와 가격대를 참고해 직접 골라 주세요."
        )
        return notes
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
    examined: int = 0,
) -> RecommendationRationale:
    """추천 근거 전체를 조립합니다."""
    return RecommendationRationale(
        price_range_basis=price_range_basis(request, low, high),
        inputs_used=inputs_used(request),
        category_basis=category_basis(request, categories, products),
        product_basis=product_basis(products, low, high, examined),
        warnings=warnings(products, low, high, examined),
    )
