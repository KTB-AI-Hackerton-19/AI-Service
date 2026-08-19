"""받은 선물 가격으로 자연스러운 답례 가격 범위를 계산합니다."""

from math import ceil, floor


def calculate_recommended_price_range(gift_price: int) -> tuple[int, int]:
    """80~120% 범위를 금액대에 맞는 단위로 바깥 방향 반올림합니다.

    저가 선물에 1,000원 단위를 일괄 적용하면 최저·최고 금액이 같아지는
    문제가 있어, 받은 선물 가격에 따라 100/1,000/10,000원 단위를 씁니다.
    최저가는 내리고 최고가는 올려 80~120% 범위를 좁히지 않습니다.
    """
    if gift_price < 10_000:
        unit = 100
    elif gift_price < 100_000:
        unit = 1_000
    else:
        unit = 10_000

    minimum = floor(gift_price * 0.8 / unit) * unit
    maximum = ceil(gift_price * 1.2 / unit) * unit
    return max(minimum, 100), max(maximum, unit)
