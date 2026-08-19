"""선물 추천 함수가 사용하는 입력·출력 데이터 모델."""

from enum import StrEnum

from typing import Any

from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator


class Gender(StrEnum):
    """상대방 성별. 추천 카테고리에 영향을 줍니다."""

    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class MessageSource(StrEnum):
    """감사 메시지 문장을 **누가 썼는지**.

    ``SimpleGiftRecommendationResponse.source`` 로는 이것을 알 수 없습니다. 그 값은
    추천 백엔드와 JSON 파싱 성공 여부만 반영하므로, 파싱이 성공한 뒤 길이 미달로
    메시지만 통째로 템플릿에 교체돼도 그대로 ``BEDROCK_CLAUDE`` 입니다. 3차 실측에서
    4건 중 2건이 그 경우였는데 응답만 보고는 구분할 수 없어 폴백 문구가 모델 출력으로
    오독됐습니다.

    판정은 ``MODEL`` 하나만 보면 됩니다. **``MODEL`` 이 아닌 값은 전부 템플릿**이고,
    나머지 값은 왜 템플릿으로 떨어졌는지를 나눠 볼 때만 씁니다.
    """

    MODEL = "MODEL"
    """모델이 쓴 문장이 그대로 나갔습니다(이름·조사 교정만 적용)."""

    TEMPLATE_TOO_SHORT = "TEMPLATE_TOO_SHORT"
    """모델이 문장을 쓰긴 했지만 ``MIN_MESSAGE_LENGTH`` 에 못 미쳐 폐기했습니다.

    이 값이 잦으면 프롬프트가 요구하는 길이(``TARGET_MESSAGE_LENGTH``)를 올릴 자리입니다.
    """

    TEMPLATE_NO_OUTPUT = "TEMPLATE_NO_OUTPUT"
    """모델 문장이 아예 없었습니다. JSON 파싱 실패, 필드 누락, mock 백엔드가 여기입니다.

    이 값이 잦으면 프롬프트 형식이나 ``bedrock_temperature`` 를 볼 자리입니다.
    """


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
    snippet: str | None = Field(
        default=None,
        max_length=200,
        description="내부 진단용. 응답에는 실리지 않습니다(_hide_snippet 참고).",
    )

    @field_serializer("snippet")
    def _hide_snippet(self, value: str | None) -> str | None:
        """검색 스니펫은 응답에 싣지 않습니다.

        이 값은 우리가 쓴 문장이 아니라 판매자·페이지에서 긁어 온 텍스트인데,
        추천 카드 안에 놓이면 우리가 그 상품에 대해 한 말처럼 읽힙니다. 4·5차
        실측에서 화면까지 나간 값은 셋뿐이었고 그중 하나가 결제 화면 부스러기
        였습니다("감성 엽서 증정 무료배송. 장바구니 담기 사용안함,위시, 담은
        수4.5만, 스위치."). 나머지 둘은 판매자 홍보 문구였습니다.

        길이 제한으로는 못 막습니다. 그 부스러기는 47자로, 정상으로 통과한 마케팅
        문구(178자)보다 **짧습니다**. 길이는 판별 신호가 아닙니다. ``clean_snippet``
        은 라운드마다 새로운 잔재 계열을 하나씩 놓쳤고(2차 판매가 표, 3차 상품명
        표, 4차 장바구니 잔재), 실제 페이지를 받아 보지 않는 한 다음 계열도 같은
        방식으로 새어 나갑니다.

        카드에는 이미 ``title`` · ``price`` · ``url`` · ``reason`` 이 있고 그중
        ``reason`` 은 우리가 근거를 적어 만든 문장입니다. 검증 못 하는 남의 문구가
        더 보태 줄 것이 없습니다.

        필드 자체는 남깁니다. 백엔드가 이 스펙으로 Java 클라이언트를 생성하므로
        (``scripts/export_openapi.py``) 속성을 지우면 게터가 사라집니다. 값을
        채우는 쪽(``product_search.search_one``)과 다듬는 쪽(``clean_snippet``)도
        그대로 두어, 다시 내보내기로 정하면 이 함수 하나만 지우면 됩니다.

        라우터가 ``response_model_exclude_none=True`` 라 실제 응답에서는 키가
        통째로 빠집니다. 5차 실측에서도 상품 3건 중 2건은 이미 키가 없었으므로
        나가는 JSON 의 모양은 달라지지 않습니다.
        """
        return None


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
    # 위와 같은 이유로 직렬화하지 않습니다. 최종 응답에서는
    # recommend_gift_info.message.message_source 로 나갑니다.
    # 기본값을 두지 않습니다. 여기서 아무 값이나 채우면 "모델이 썼다"는 거짓말이
    # 조용히 섞이는데, 그 오보를 없애려고 만든 필드입니다.
    message_source: MessageSource = Field(exclude=True)
    model: str
    source: str

    @model_validator(mode="after")
    def validate_price_range(self) -> "SimpleGiftRecommendationResponse":
        """최저 추천 금액이 최고 추천 금액보다 큰 잘못된 응답을 차단합니다."""
        if self.recommended_price_min > self.recommended_price_max:
            raise ValueError("추천 최저 금액은 최고 금액보다 클 수 없습니다.")
        return self
