"""추천 근거 조립 테스트.

근거는 모델이 쓰는 문장이 아니라 규칙에서 나옵니다.
그래서 없는 근거를 있는 것처럼 말하지 않는지가 핵심 검증입니다.
"""

import pytest

from app.schemas.recommendation import (
    Gender,
    ProductSuggestion,
    SimpleGiftRecommendationRequest,
)
from app.services import recommendation_rationale as rationale
from app.services.recommendation_policy import price_range


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

    def test_candidates_found_but_none_priced_in_range(self):
        """실측 gift: 후보 9건을 찾아 놓고 "검색 결과가 없어" 라고 말할 참이었습니다.

        가격을 모르는 상품을 더 이상 노출하지 않으므로 0건이 자주 나옵니다. 그때
        검색이 빈 것과 가격이 안 맞은 것을 같은 문장으로 말하면 안 됩니다.
        """
        text = rationale.product_basis([], 8000, 12000, examined=9)

        assert "검색 결과가 없어" not in text
        assert "상품 후보 9개를 찾았지만" in text
        assert "8,000원 ~ 12,000원" in text


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

    def test_clean_case_has_no_warnings(self):
        assert rationale.warnings([product(40000)], 30000, 50000) == []

    def test_zero_products_is_explained_not_left_silent(self):
        """상품 0건은 화면에서 가장 설명이 필요한 상태입니다."""
        found = rationale.warnings([], 8000, 12000, examined=9)
        empty = rationale.warnings([], 8000, 12000)

        assert any("제안 가격대에 맞는 상품을 확인하지 못해" in n for n in found)
        assert any("상품을 찾지 못했습니다" in n for n in empty)
        assert found != empty


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


class TestPriceRangeBasisMatchesTheRealCalculation:
    """"80% ~ 120% 범위로 잡았습니다" 는 실제 계산과 맞을 때만 근거입니다."""

    def test_non_round_amount_shows_both_the_ratio_and_the_result(self):
        req = SimpleGiftRecommendationRequest(gift_name="아메리카노", gift_price=12300)
        low, high = price_range(req)
        text = rationale.price_range_basis(req, low, high)

        assert "9,840원" in text and "14,760원" in text  # 80% / 120% 원값
        assert f"{low:,}원 ~ {high:,}원" in text  # 화면에 실제로 보이는 값

    def test_low_amount_stays_in_the_band(self):
        req = SimpleGiftRecommendationRequest(gift_name="사탕", gift_price=3000)
        low, high = price_range(req)
        text = rationale.price_range_basis(req, low, high)

        assert (low, high) == (2400, 3600)
        assert "2,400원" in text and "3,600원" in text

    @pytest.mark.parametrize("price", [3000, 12300, 50000])
    def test_stated_numbers_are_the_ones_that_ship(self, price):
        req = SimpleGiftRecommendationRequest(gift_name="선물", gift_price=price)
        low, high = price_range(req)
        text = rationale.price_range_basis(req, low, high)
        assert f"{low:,}원 ~ {high:,}원으로 잡았습니다" in text

    def test_clamped_budget_says_so(self):
        """500원 예산이 1,000원으로 올라갔는데 "그대로 따랐습니다" 는 거짓입니다."""
        req = SimpleGiftRecommendationRequest(budget_min=500, budget_max=3000)
        low, high = price_range(req)
        text = rationale.price_range_basis(req, low, high)

        assert low == 1000
        assert "그대로 따랐습니다" not in text
        assert "500원" in text and "1,000원" in text

    def test_budget_max_only_does_not_claim_a_given_lower_bound(self):
        req = SimpleGiftRecommendationRequest(budget_max=50000)
        low, high = price_range(req)
        text = rationale.price_range_basis(req, low, high)

        assert "그대로 따랐습니다" not in text
        assert "하한은 지정이 없어" in text


class TestCategoryBasisDoesNotOverclaim:
    def test_failed_narrowing_is_not_reported_as_success(self):
        """좁히기에 실패하면 정책이 모델 카테고리를 그대로 내보냅니다."""
        req = SimpleGiftRecommendationRequest(preferred_categories=["뷰티·화장품"])
        text = rationale.category_basis(req, ["상품권"])

        assert "안에서만 골랐습니다" not in text
        assert "찾지 못해" in text

    def test_alias_of_the_chosen_category_still_counts_as_success(self):
        req = SimpleGiftRecommendationRequest(preferred_categories=["화장품"])
        assert "안에서만 골랐습니다" in rationale.category_basis(req, ["패션·잡화"])

    def test_subset_of_the_chosen_categories_is_still_within_them(self):
        req = SimpleGiftRecommendationRequest(preferred_categories=["뷰티·화장품", "상품권"])
        assert "안에서만 골랐습니다" in rationale.category_basis(req, ["상품권"])


class TestSentencesAreFinishedKorean:
    def test_no_dual_particle_notation(self):
        """"카테고리을(를)" 은 완성된 문장이 아닙니다."""
        req = SimpleGiftRecommendationRequest(age=29, relationship="친구")
        text = rationale.category_basis(req, ["뷰티·화장품"])

        assert "을(를)" not in text
        assert "연령대·관계를 고려해" in text
        assert "뷰티·화장품을 골랐습니다" in text

    def test_particle_follows_the_final_consonant(self):
        req = SimpleGiftRecommendationRequest()
        text = rationale.category_basis(req, ["커피·차"])

        assert "받은 선물의 성격을 고려해" in text  # 격: 받침 있음
        assert "커피·차를 골랐습니다" in text  # 차: 받침 없음


class TestCategoryBasisReportsWhatActuallyShipped:
    """"셋을 골랐습니다" 는 셋이 화면에 있을 때만 근거입니다.

    실측 /from-image: 커피·차(85), 식품·디저트(70), 생활용품(60) 을 골랐다고 써 놓고
    예산 안에 든 후보가 최저 점수 카테고리에만 남아 생활용품 볼펜 한 개가 나갔습니다.
    같은 응답의 category_basis 는 셋을 골랐다고만 말해 화면과 어긋났습니다.
    """

    def in_category(self, category: str, price: int = 9800) -> ProductSuggestion:
        return ProductSuggestion(
            title="상품",
            url="https://gift.kakao.com/product/1",
            source="카카오 선물하기",
            category=category,
            price=price,
            price_verified=False,
        )

    def test_partial_coverage_names_the_category_that_shipped(self):
        req = SimpleGiftRecommendationRequest(gift_name="빽다방 금액권", gift_price=10000)
        text = rationale.category_basis(
            req,
            ["커피·차", "식품·디저트", "생활용품"],
            [self.in_category("생활용품")],
        )

        assert "커피·차, 식품·디저트, 생활용품을 골랐습니다" in text
        assert "이 가격대에서 상품이 나온 것은 생활용품입니다" in text

    def test_full_coverage_says_nothing_extra(self):
        """어긋나지 않았는데 덧붙이면 문장만 길어집니다."""
        req = SimpleGiftRecommendationRequest(preferred_categories=["꽃·식물"])
        text = rationale.category_basis(req, ["꽃·식물"], [self.in_category("꽃·식물")])

        assert text == "사용자가 고른 카테고리 안에서만 골랐습니다: 꽃·식물"

    def test_no_products_says_nothing_extra(self):
        """상품이 없으면 product_basis 가 "검색 결과가 없어" 라고 이미 말합니다."""
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)
        assert "상품이 나온 것은" not in rationale.category_basis(req, ["커피·차"], [])

    def test_products_without_a_category_are_not_counted(self):
        """카테고리를 모르는 상품으로는 어느 카테고리가 나왔는지 말할 수 없습니다."""
        product = self.in_category("커피·차")
        product.category = None
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)

        assert "상품이 나온 것은" not in rationale.category_basis(req, ["커피·차"], [product])

    def test_sentence_is_still_punctuated_on_the_preferred_category_path(self):
        """그 갈래의 문장은 마침표 없이 끝납니다. 이어 붙일 때 문장이 붙으면 안 됩니다."""
        req = SimpleGiftRecommendationRequest(preferred_categories=["커피·차", "생활용품"])
        text = rationale.category_basis(
            req, ["커피·차", "생활용품"], [self.in_category("생활용품")]
        )

        assert "커피·차, 생활용품. 이 가격대에서" in text

    def test_build_carries_the_products_into_the_basis(self):
        """build 가 products 를 넘기지 않으면 이 교정이 응답에 실리지 않습니다."""
        req = SimpleGiftRecommendationRequest(gift_name="빽다방 금액권", gift_price=10000)
        result = rationale.build(
            req,
            ["커피·차", "생활용품"],
            [self.in_category("생활용품")],
            8000,
            12000,
        )

        assert "상품이 나온 것은 생활용품입니다" in result.category_basis
