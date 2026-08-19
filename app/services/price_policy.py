"""받은 선물 가격으로 자연스러운 답례 가격 범위를 계산합니다.

답례 가격대는 받은 금액의 80~120% 라는 규칙 하나에서 나옵니다.
추천 정책(``recommendation_policy.price_range``)과 mock 추천이 모두 이 모듈을
거치므로, 근거 문장(``recommendation_rationale.price_range_basis``)이 말하는
비율과 실제 계산이 어긋나지 않습니다.
"""

from math import ceil, floor

PRICE_FLOOR_RATIO = 0.8
PRICE_CEILING_RATIO = 1.2
"""답례 가격대의 하한·상한 비율. 근거 문장도 이 값을 그대로 읽어 씁니다."""

_ABSOLUTE_MIN = 100


def rounding_unit(amount: int) -> int:
    """금액대에 맞는 반올림 단위.

    저가 선물에 1,000원 단위를 일괄 적용하면 최저·최고 금액이 같아지는
    문제가 있어, 받은 금액에 따라 100/1,000/10,000원 단위를 씁니다.
    """
    if amount < 10_000:
        return 100
    if amount < 100_000:
        return 1_000
    return 10_000


def floor_price(amount: int) -> int:
    """80% 하한. 단위 아래로 내려 80%를 잘라먹지 않습니다."""
    unit = rounding_unit(amount)
    return max(floor(amount * PRICE_FLOOR_RATIO / unit) * unit, _ABSOLUTE_MIN)


def ceil_price(amount: int) -> int:
    """120% 상한. 단위 위로 올려 120%를 잘라먹지 않습니다.

    상한도 내림하면 12,300원을 받았을 때 상한이 14,000원(113.8%)이 되어
    "120% 범위"라는 근거 문장이 거짓말이 됩니다.
    """
    unit = rounding_unit(amount)
    return max(ceil(amount * PRICE_CEILING_RATIO / unit) * unit, unit)


def calculate_recommended_price_range(gift_price: int) -> tuple[int, int]:
    """받은 금액 하나에 대한 80~120% 범위를 금액대에 맞는 단위로 넓혀 돌려줍니다."""
    minimum = floor_price(gift_price)
    return minimum, max(minimum, ceil_price(gift_price))
