"""Qwen 선물 추천 함수가 사용하는 입력·출력 데이터 모델."""

from pydantic import BaseModel, Field, model_validator


class SimpleGiftRecommendationRequest(BaseModel):
    """추천 모델 입력.

    앞쪽 다섯 필드는 기존 계약 그대로이며 **대표 1건**을 담습니다.
    뒤쪽은 전부 기본값이 있는 선택 항목으로, 이미지 한 장에 여러 건이 있거나
    받은 것이 선물이 아닌 경우(청첩장 등)의 맥락을 전달합니다.

    Attributes:
        gift_name: 사용자가 받은 선물 이름.
        gift_price: 받은 선물의 추정 가격(원).
        age: 선물을 다시 받을 상대의 나이. 모르면 ``None``.
    """

    gift_name: str = Field(min_length=1, max_length=200)
    gift_price: int = Field(gt=0, le=100_000_000)
    age: int | None = Field(default=None, ge=0, le=120)
    person_name: str | None = Field(default=None, max_length=50)
    relationship: str | None = Field(default=None, max_length=50)

    # ── 이하 확장 필드. 전부 선택이며 기존 사용처에 영향을 주지 않습니다. ──
    record_type: str = Field(
        default="gift",
        description="gift | money | event_invitation | receipt | unknown. 받은 것의 종류",
    )
    event: str | None = Field(default=None, max_length=50, description="생일 / 결혼 / 조의 등 계기")
    received_amounts: list[int] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "여러 사람에게 받았을 때 각각의 금액. 비어 있으면 gift_price 하나만 있는 것으로 봅니다. "
            "가격 범위를 이 값들의 최소~최대로 잡아, 적게 준 사람에게 과한 답례를 권하지 않습니다."
        ),
    )
    people: list[str] = Field(
        default_factory=list, max_length=20, description="받은 사람들의 이름. 여러 명일 때만 채웁니다."
    )


class CategoryRecommendation(BaseModel):
    """Qwen이 제안한 하나의 선물 카테고리."""

    category: str = Field(min_length=1, max_length=50)
    score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=300)
    product_examples: list[str] = Field(default_factory=list, max_length=3)


class ProductSuggestion(BaseModel):
    """국내 거래 플랫폼에서 실제로 찾은 상품.

    ``CategoryRecommendation.product_examples`` 는 상품 '유형'이라 링크가 없지만,
    이쪽은 검색으로 찾은 실제 페이지라 사용자가 바로 구매로 이동할 수 있습니다.
    """

    title: str = Field(min_length=1, max_length=200)
    url: str = Field(min_length=1, max_length=1000)
    source: str = Field(max_length=50, description="쿠팡 / 카카오 선물하기 / 네이버 쇼핑 등")
    category: str | None = Field(default=None, max_length=50)
    price: int | None = Field(
        default=None, ge=0, description="검색 결과에서 읽어 낸 가격. 못 읽으면 None"
    )
    kind: str = Field(
        default="product",
        description="product=개별 상품 페이지 / listing=검색·목록 페이지",
    )
    snippet: str | None = Field(default=None, max_length=200)


class SimpleGiftRecommendationResponse(BaseModel):
    """추천 모델 출력과 추론 출처를 포함한 최종 추천 결과."""

    input_gift_name: str
    input_gift_price: int
    input_age: int | None
    recommended_price_min: int = Field(ge=0)
    recommended_price_max: int = Field(ge=0)
    categories: list[CategoryRecommendation] = Field(min_length=1, max_length=3)
    products: list[ProductSuggestion] = Field(
        default_factory=list,
        max_length=10,
        description="실제 구매 가능한 상품. 검색이 비활성이거나 실패하면 빈 배열입니다.",
    )
    summary: str = Field(min_length=1, max_length=500)
    # Qwen 서비스와 메시지 준비 작업 사이에서만 사용하는 내부 전달값입니다.
    # 최종 HTTP 응답에서는 message.content와 중복되므로 직렬화하지 않습니다.
    suggested_message: str = Field(min_length=1, max_length=500, exclude=True)
    model: str
    source: str

    @model_validator(mode="after")
    def validate_price_range(self) -> "SimpleGiftRecommendationResponse":
        """최저 추천 금액이 최고 추천 금액보다 큰 잘못된 응답을 차단합니다."""
        if self.recommended_price_min > self.recommended_price_max:
            raise ValueError("추천 최저 금액은 최고 금액보다 클 수 없습니다.")
        return self
