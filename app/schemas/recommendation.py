"""선물 추천 함수가 사용하는 입력·출력 데이터 모델."""

from enum import StrEnum

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Gender(StrEnum):
    """상대방 성별. 추천 카테고리에 영향을 줍니다."""

    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


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

    gift_name: str = Field(default="받은 선물", min_length=1, max_length=200)
    gift_price: int = Field(default=30_000, gt=0, le=100_000_000)
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

    # ── 사용자가 직접 조절하는 입력. 나이·가격대·카테고리·성별만으로도 추천이 가능합니다. ──
    gender: Gender = Field(default=Gender.UNKNOWN, description="상대방 성별. 모르면 unknown")
    budget_min: int | None = Field(
        default=None, ge=0, le=100_000_000, description="사용자가 지정한 예산 하한. 있으면 받은 금액보다 우선합니다."
    )
    budget_max: int | None = Field(
        default=None, ge=0, le=100_000_000, description="사용자가 지정한 예산 상한. 있으면 받은 금액보다 우선합니다."
    )
    preferred_categories: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="사용자가 고른 카테고리. 지정하면 모델이 이 안에서만 고릅니다.",
    )
    interests: list[str] = Field(
        default_factory=list, max_length=5, description="상대방 관심사"
    )
    dislikes: list[str] = Field(
        default_factory=list, max_length=5, description="상대방이 싫어하는 것"
    )

    @field_validator("age", mode="before")
    @classmethod
    def normalize_optional_age(cls, value: Any) -> Any:
        """0, 빈 문자열, null은 나이 정보가 없는 것으로 통일합니다."""
        if value is None or value == 0:
            return None
        if isinstance(value, str) and value.strip() in {"", "0"}:
            return None
        return value

    @field_validator("person_name", "relationship", "event", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: Any) -> Any:
        """선택 문자열의 공백을 제거하고 빈 값은 ``None``으로 바꿉니다."""
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip() or None
        return value


class CategoryRecommendation(BaseModel):
    """Qwen이 제안한 하나의 선물 카테고리."""

    category: str = Field(min_length=1, max_length=50)
    score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=300)
    product_examples: list[str] = Field(default_factory=list, max_length=3)
    search_query: str = Field(default="", max_length=200)


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
        default=None, ge=0, description="상품 페이지에서 확인한 판매가. 확인하지 못하면 None"
    )
    price_verified: bool = Field(
        default=False,
        description="상품 페이지 본문의 '판매가' 표기에서 읽었는지. false 면 가격을 신뢰할 수 없습니다.",
    )
    kind: str = Field(
        default="product",
        description="product=개별 상품 페이지 / listing=검색·목록 페이지",
    )
    reason: str = Field(
        default="", max_length=200, description="이 상품을 고른 이유. 화면에 그대로 보여 줄 수 있습니다."
    )
    snippet: str | None = Field(default=None, max_length=200)


class RecommendationRationale(BaseModel):
    """추천이 이렇게 나온 근거.

    카테고리별 이유(``CategoryRecommendation.reason``)는 모델이 쓰지만, 여기 값들은
    규칙에서 결정론적으로 나옵니다. 모델이 지어낸 설명이 아니라 실제로 적용된 계산이라
    사용자에게 그대로 보여 줘도 됩니다.
    """

    price_range_basis: str = Field(
        default="", max_length=300, description="가격 범위를 그렇게 잡은 근거"
    )
    inputs_used: list[str] = Field(
        default_factory=list,
        max_length=15,
        description="추천에 실제로 반영된 입력. 반영되지 않은 것은 넣지 않습니다.",
    )
    category_basis: str = Field(
        default="", max_length=300, description="카테고리를 그렇게 좁힌 근거"
    )
    product_basis: str = Field(
        default="", max_length=300, description="상품을 그렇게 고른 근거"
    )
    warnings: list[str] = Field(
        default_factory=list, max_length=10, description="사용자가 알아야 할 한계"
    )


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
    rationale: RecommendationRationale = Field(
        default_factory=RecommendationRationale, description="추천 근거"
    )
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
