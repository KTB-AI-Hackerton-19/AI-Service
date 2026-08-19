"""Qwen 선물 추천 함수가 사용하는 입력·출력 데이터 모델."""

from pydantic import BaseModel, Field, model_validator


class SimpleGiftRecommendationRequest(BaseModel):
    """추천 모델 입력.

    Attributes:
        gift_name: 사용자가 받은 선물 이름.
        gift_price: 받은 선물의 추정 가격(원).
        age: 선물을 다시 받을 상대의 나이. 모르면 ``None``.
    """

    gift_name: str = Field(min_length=1, max_length=200)
    gift_price: int = Field(gt=0, le=100_000_000)
    age: int | None = Field(default=None, ge=0, le=120)


class CategoryRecommendation(BaseModel):
    """Qwen이 제안한 하나의 선물 카테고리."""

    category: str = Field(min_length=1, max_length=50)
    score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=300)
    product_examples: list[str] = Field(default_factory=list, max_length=3)


class SimpleGiftRecommendationResponse(BaseModel):
    """추천 모델 출력과 추론 출처를 포함한 최종 추천 결과."""

    input_gift_name: str
    input_gift_price: int
    input_age: int | None
    recommended_price_min: int = Field(ge=0)
    recommended_price_max: int = Field(ge=0)
    categories: list[CategoryRecommendation] = Field(min_length=1, max_length=3)
    summary: str = Field(min_length=1, max_length=500)
    model: str
    source: str

    @model_validator(mode="after")
    def validate_price_range(self) -> "SimpleGiftRecommendationResponse":
        """최저 추천 금액이 최고 추천 금액보다 큰 잘못된 응답을 차단합니다."""
        if self.recommended_price_min > self.recommended_price_max:
            raise ValueError("추천 최저 금액은 최고 금액보다 클 수 없습니다.")
        return self
