"""추천 근거 조립 테스트.

근거는 모델이 쓰는 문장이 아니라 규칙에서 나옵니다.
그래서 없는 근거를 있는 것처럼 말하지 않는지가 핵심 검증입니다.
"""

from app.schemas.recommendation import (
    Gender,
    ProductSuggestion,
    SimpleGiftRecommendationRequest,
)
from app.services import recommendation_rationale as rationale


def product(price: int | None, verified: bool = True, kind: str = "product") -> ProductSuggestion:
    return ProductSuggestion(
        title="상품",
        url=f"https://www.coupang.com/vp/products/{price}",
        source="쿠팡",
        category="식품·디저트",
        price=price,
        price_verified=verified,
        kind=kind,
    )


class TestPriceRangeBasis:
    def test_user_budget_is_stated_as_such(self):
        req = SimpleGiftRecommendationRequest(budget_min=30000, budget_max=50000)
        text = rationale.price_range_basis(req, 30000, 50000)
        assert "사용자가 지정한 예산" in text

    def test_single_gift_explains_80_120(self):
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=35000)
        text = rationale.price_range_basis(req, 28000, 42000)
        assert "35,000원" in text and "80%" in text

    def test_multiple_people_explain_span(self):
        req = SimpleGiftRecommendationRequest(
            gift_name="축의금", gift_price=200000, received_amounts=[50000, 100000, 200000]
        )
        text = rationale.price_range_basis(req, 40000, 240000)
        assert "3명" in text
        assert "50,000원" in text and "200,000원" in text


class TestInputsUsed:
    def test_only_provided_inputs_are_listed(self):
        """없는 근거를 있는 것처럼 보이면 안 됩니다."""
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=35000)
        used = rationale.inputs_used(req)

        assert not any("나이" in u for u in used)
        assert not any("성별" in u for u in used)
        assert not any("관계" in u for u in used)
        assert any("35,000원" in u for u in used)

    def test_lists_age_gender_relationship(self):
        req = SimpleGiftRecommendationRequest(
            gift_name="케이크",
            gift_price=35000,
            age=29,
            gender=Gender.FEMALE,
            relationship="대학 동기",
            event="생일",
        )
        used = rationale.inputs_used(req)

        assert "나이 29세" in used
        assert "성별 여성" in used
        assert "관계 대학 동기" in used
        assert "계기 생일" in used

    def test_lists_interests_and_dislikes(self):
        req = SimpleGiftRecommendationRequest(
            gift_name="케이크", gift_price=35000, interests=["커피"], dislikes=["향수"]
        )
        used = rationale.inputs_used(req)
        assert any("커피" in u for u in used)
        assert any("향수" in u for u in used)


class TestCategoryBasis:
    def test_user_selection_is_stated(self):
        req = SimpleGiftRecommendationRequest(preferred_categories=["식품·디저트"])
        assert "사용자가 고른 카테고리" in rationale.category_basis(req, ["식품·디저트"])

    def test_derived_selection_names_the_signals(self):
        req = SimpleGiftRecommendationRequest(age=29, gender=Gender.FEMALE, relationship="친구")
        text = rationale.category_basis(req, ["뷰티·화장품"])
        assert "연령대" in text and "성별" in text and "관계" in text


class TestProductBasis:
    def test_counts_verified_and_in_range(self):
        products = [product(40000), product(200000), product(45000)]
        text = rationale.product_basis(products, 30000, 50000)
        assert "3개 중 3개는 상품 페이지에서 판매가를 확인" in text
        assert "2개가 30,000원 ~ 50,000원 안에" in text

    def test_no_products(self):
        assert "검색 결과가 없어" in rationale.product_basis([], 30000, 50000)


class TestWarnings:
    def test_flags_unverified_prices(self):
        notes = rationale.warnings([product(40000, verified=False)], 30000, 50000)
        assert any("확인하지 못해 참고용" in n for n in notes)

    def test_flags_missing_prices(self):
        notes = rationale.warnings([product(None)], 30000, 50000)
        assert any("가격을 확인하지 못했습니다" in n for n in notes)

    def test_flags_out_of_budget(self):
        notes = rationale.warnings([product(200000)], 30000, 50000)
        assert any("제안 가격대를 벗어납니다" in n for n in notes)

    def test_flags_listing_pages(self):
        notes = rationale.warnings([product(40000, kind="listing")], 30000, 50000)
        assert any("검색 결과 페이지" in n for n in notes)

    def test_clean_case_has_no_warnings(self):
        assert rationale.warnings([product(40000)], 30000, 50000) == []


def test_build_assembles_everything():
    req = SimpleGiftRecommendationRequest(
        gift_name="케이크", gift_price=35000, age=29, gender=Gender.FEMALE
    )
    result = rationale.build(req, ["식품·디저트"], [product(40000)], 28000, 42000)

    assert result.price_range_basis
    assert result.inputs_used
    assert result.category_basis
    assert result.product_basis
    assert result.warnings == []
