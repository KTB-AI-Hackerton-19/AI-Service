"""Tavily 상품 검색 테스트. 실제 네트워크에 나가지 않습니다."""

import json
import logging
import threading

import httpx
import pytest
import respx
from fastapi.encoders import jsonable_encoder

from app.core.config import settings
from app.schemas.agent import RecommendRequest
from app.schemas.recommendation import (
    CategoryRecommendation,
    MessageSource,
    ProductSuggestion,
    SimpleGiftRecommendationRequest,
    SimpleGiftRecommendationResponse,
)
from app.services import product_search as product_search_module
from app.services.product_search import (
    TavilyProductSearch,
    build_query,
    clean_snippet,
    clean_title,
    extract_price,
    extract_title_price,
    out_of_season,
    product_search,
)
from app.services.product_search import SearchStats
from app.services.qwen_service import qwen_service
from app.services.recommendation_policy import SAFE_EXAMPLES
from app.services.tasks.recommendation import (
    RecommendationPreparationService,
    _search_targets,
    recommendation_preparation_service,
)

TAVILY_URL = settings.tavily_url


def tavily_response(*items: dict) -> httpx.Response:
    return httpx.Response(200, json={"results": list(items), "response_time": 1.2})


def result(title: str, url: str, content: str = "") -> dict:
    return {"title": title, "url": url, "content": content, "score": 0.9}


@pytest.fixture(autouse=True)
def forget_blocked_hosts():
    """봇 차단 기억은 프로세스 단위라 테스트끼리 새지 않게 비웁니다."""
    product_search_module._BOT_BLOCKED_HOSTS.clear()
    yield
    product_search_module._BOT_BLOCKED_HOSTS.clear()


@pytest.fixture
def tavily_on(monkeypatch):
    """검색만 켭니다. 가격 확정(Extract)은 전용 테스트에서만 켭니다."""
    monkeypatch.setattr(settings, "tavily_enabled", True)
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-test-key")
    monkeypatch.setattr(settings, "tavily_extract_limit", 0)


@pytest.fixture
def extract_on(monkeypatch):
    monkeypatch.setattr(settings, "tavily_extract_limit", 6)


EXTRACT_URL = settings.tavily_extract_url


def extract_response(*pairs: tuple[str, str]) -> httpx.Response:
    """(url, 본문) 쌍으로 Extract 응답을 만듭니다."""
    return httpx.Response(
        200,
        json={"results": [{"url": u, "raw_content": c} for u, c in pairs], "failed_results": []},
    )


class TestExtractPrice:
    def test_reads_korean_price(self):
        assert extract_price("프리미엄 디저트 세트 45,000원", 40000, 60000) == 45000

    def test_ignores_out_of_range_numbers(self):
        """검색 결과에는 배송비나 후기 수 같은 무관한 숫자도 '원'과 함께 나옵니다."""
        assert extract_price("배송비 3,000원 무료", 40000, 60000) is None

    def test_accepts_half_to_double_range(self):
        assert extract_price("25,000원", 40000, 60000) == 25000  # 하한의 절반 이상
        assert extract_price("119,000원", 40000, 60000) == 119000  # 상한의 두 배 이하
        assert extract_price("500,000원", 40000, 60000) is None  # 두 배 초과

    def test_no_price(self):
        assert extract_price("가격 문의", 40000, 60000) is None
        assert extract_price("", 40000, 60000) is None

    def test_title_price_is_kept_even_outside_recommended_range(self):
        """제목의 금액은 그 옵션의 가격일 가능성이 높아 범위 밖이어도 남깁니다."""
        assert extract_title_price("[10만원권] 상품권 교환권 99,000원", 40000, 60000) == 99000

    def test_title_price_drops_amounts_far_from_the_range(self):
        """'10,000원 이상 구매 시 무료배송'을 상품가로 읽으면 안 됩니다."""
        assert extract_title_price("10,000원 이상 구매 시 무료배송", 40000, 60000) is None
        assert extract_title_price("50만원 이상 구매 시 사은품 증정", 40000, 60000) is None

    def test_title_price_skips_to_the_plausible_amount(self):
        """앞의 숫자가 상품가가 아니어도 뒤에 있는 진짜 가격은 살립니다."""
        title = "10,000원 이상 무료배송 수제쿠키 선물세트 45,000원"
        assert extract_title_price(title, 40000, 60000) == 45000


class TestBuildQuery:
    def test_uses_product_example_over_category(self):
        """카테고리명만으로는 검색이 잘 되지 않습니다."""
        query = build_query("디저트", "프리미엄 디저트 세트", 40000, 60000)
        assert "프리미엄 디저트 세트" in query
        assert "5만원대" in query  # 중앙값 50,000

    def test_falls_back_to_category(self):
        assert "디저트" in build_query("디저트", None, 40000, 60000)

    def test_price_hint_uses_midpoint_not_ceiling(self):
        """상한을 쓰면 4만~24만원 같은 넓은 범위에서 29만원짜리만 걸려 나옵니다."""
        assert "14만원대" in build_query("디저트", None, 40000, 240000)

    def test_small_budget_uses_won(self):
        assert "7000원" in build_query("디저트", None, 5000, 9000)

    def test_the_hint_never_points_outside_the_budget(self):
        """4차 실측 gift: 예산 8,000~12,000 에 "1만원대"(10,000~19,999)를 물었습니다.

        예산의 아래 절반이 힌트에서 빠지고 힌트의 위쪽은 노출조차 불가능한 구간이라,
        검색이 위를 겨냥했습니다. 돌아온 후보가 19,100~45,000원이었고 노출 0건이었습니다.
        """
        assert "1만원대" not in build_query("디저트", None, 8000, 12000)
        assert "1만원" in build_query("디저트", None, 8000, 12000)

    def test_budgets_that_already_fit_keep_their_hint(self):
        """고장난 곳만 바뀌어야 합니다. 4차에서 상품이 나온 두 흐름은 그대로입니다."""
        assert "2만원대" in build_query("꽃·식물", None, 18000, 27000)
        assert "3만원대" in build_query("디저트", None, 28000, 42000)


class TestAvailability:
    def test_disabled_without_key(self, monkeypatch):
        monkeypatch.setattr(settings, "tavily_enabled", True)
        monkeypatch.setattr(settings, "tavily_api_key", "")
        assert product_search.is_available is False

    def test_disabled_by_flag(self, monkeypatch):
        monkeypatch.setattr(settings, "tavily_enabled", False)
        monkeypatch.setattr(settings, "tavily_api_key", "tvly-x")
        assert product_search.is_available is False

    async def test_search_returns_empty_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(settings, "tavily_enabled", False)
        assert await product_search.search([("디저트", None)], 40000, 60000) == []


class TestSearch:
    @respx.mock
    async def test_restricts_to_trusted_domains(self, tavily_on):
        """블로그·카페의 광고성 글을 걸러 내려면 도메인 제한이 필수입니다."""
        route = respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("수제쿠키 프리미엄 선물세트 45,000원", "https://www.coupang.com/vp/products/1")
            )
        )
        await TavilyProductSearch().search([("디저트", "디저트 세트")], 40000, 60000)

        body = route.calls.last.request.content.decode()
        assert "coupang.com" in body
        assert "gift.kakao.com" in body
        # country 를 include_domains 와 함께 보내면 결과가 0건이 됩니다(실측).
        assert "country" not in body

    @respx.mock
    async def test_maps_source_names(self, tavily_on):
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("디저트 세트 45,000원", "https://m.coupang.com/vp/products/1"),
                result("디저트 교환권 45,000원", "https://gift.kakao.com/product/2"),
                result("과일 디저트 세트 45,000원", "https://shopping.naver.com/products/3"),
            )
        )
        products = await TavilyProductSearch().search([("디저트", None)], 40000, 60000, limit=3)

        assert [p.source for p in products] == ["쿠팡", "카카오 선물하기", "네이버 쇼핑"]

    @respx.mock
    async def test_in_range_price_ranks_first(self, tavily_on):
        """9,000~14,000원을 권해 놓고 20,000원짜리를 맨 앞에 보여 주면 추천의 의미가 없습니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("비싼 쿠키 세트 20,000원", "https://www.coupang.com/vp/products/1"),
                result("범위 안 쿠키 세트 12,000원", "https://www.coupang.com/vp/products/2"),
            )
        )
        products = await TavilyProductSearch().search([("디저트", None)], 9000, 14000, limit=2)

        # 20,000원은 상한 14,000원에서 +43% 입니다. 예전에는 "절반~두 배" 안이라
        # 2번 자리를 채웠지만, 그 폭이 실측에서 18,000~27,000원 요청에 49,000원(+81%)을
        # 내보냈습니다. 이제는 맨 앞이 아니라 아예 나가지 않습니다.
        assert [p.price for p in products] == [12000]

    @respx.mock
    async def test_content_subdomains_are_excluded(self, tavily_on):
        """guide.coupang.com 은 기사이지 살 수 있는 물건이 아닙니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("가격대별 선물 세트 추천", "https://guide.coupang.com/gift-recommendation"),
                result("수제쿠키 세트 12,000원", "https://www.coupang.com/vp/products/9"),
            )
        )
        products = await TavilyProductSearch().search([("디저트", None)], 9000, 14000, limit=2)

        assert products[0].kind == "product"
        assert len(products) == 1

    @respx.mock
    async def test_listing_pages_are_excluded(self, tavily_on):
        """카카오 선물하기는 개별 상품보다 검색 페이지가 주로 잡힙니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("고급 디저트 세트 | 검색", "https://gift.kakao.com/search?q=디저트"),
                result("수제쿠키 선물세트 45,000원", "https://www.coupang.com/vp/products/9"),
            )
        )
        products = await TavilyProductSearch().search([("디저트", None)], 40000, 60000, limit=2)

        assert products[0].kind == "product"
        assert products[0].price == 45000
        assert len(products) == 1

    @respx.mock
    async def test_interleaves_categories(self, tavily_on):
        """한 카테고리가 결과를 독차지하면 추천이 단조로워집니다."""
        respx.post(TAVILY_URL).mock(
            side_effect=[
                tavily_response(
                    result("디저트 A", "https://www.coupang.com/vp/products/1", "45,000원"),
                    result("디저트 B", "https://www.coupang.com/vp/products/2", "46,000원"),
                ),
                tavily_response(
                    result("고급 타월 C", "https://www.11st.co.kr/products/3", "47,000원")
                ),
            ]
        )
        products = await TavilyProductSearch().search(
            [("디저트", None), ("생활용품", None)], 40000, 60000, limit=3
        )

        assert [p.category for p in products] == ["디저트", "생활용품", "디저트"]

    @respx.mock
    async def test_deduplicates_urls(self, tavily_on):
        same = "https://www.coupang.com/vp/products/1"
        respx.post(TAVILY_URL).mock(
            side_effect=[
                tavily_response(result("디저트 쿠키", same, "45,000원")),
                tavily_response(result("고급 타월 세트", same, "45,000원")),
            ]
        )
        products = await TavilyProductSearch().search(
            [("디저트", None), ("생활용품", None)], 40000, 60000
        )
        assert len(products) == 1

    @respx.mock
    async def test_deduplicates_mobile_and_desktop_product_urls(self, tavily_on):
        """같은 상품의 m/www 주소가 달라도 하나만 노출합니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("디저트 쿠키", "https://m.coupang.com/vp/products/123?itemId=1", "35,000원"),
                result(
                    "디저트 쿠키",
                    "https://www.coupang.com/vp/products/123?vendorItemId=2",
                    "35,000원",
                ),
            )
        )
        products = await TavilyProductSearch().search(
            [("디저트", None)], 30000, 50000
        )
        assert len(products) == 1

    @respx.mock
    async def test_excludes_semantically_unrelated_detail_product(self, tavily_on):
        """상품권 추천에 화장품 상세페이지가 섞이지 않아야 합니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result(
                    "랑콤 썸머 베이스 듀오",
                    "https://m.ssg.com/item/dealItemView.ssg?itemId=1",
                    "35,000원",
                ),
                result("외식 상품권 3만원권", "https://gift.kakao.com/product/2", "30,000원"),
            )
        )
        products = await TavilyProductSearch().search(
            [("상품권", None)], 28000, 42000
        )
        assert [p.title for p in products] == ["외식 상품권 3만원권"]

    @respx.mock
    async def test_category_cannot_be_bypassed_by_wrong_model_example(self, tavily_on):
        """모델 예시가 틀려도 상품권 카테고리에 화장품을 통과시키지 않습니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result(
                    "랑콤 썸머 베이스 듀오",
                    "https://m.ssg.com/item/dealItemView.ssg?itemId=1",
                    "상품권 선물 검색 결과에서 함께 노출된 상품",
                )
            )
        )
        products = await TavilyProductSearch().search(
            [("상품권", "랑콤 썸머 베이스 듀오")], 28000, 42000
        )
        assert products == []

    @respx.mock
    async def test_http_error_returns_empty(self, tavily_on):
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(429, text="rate limited"))
        assert await TavilyProductSearch().search([("디저트", None)], 40000, 60000) == []

    @respx.mock
    async def test_network_error_returns_empty(self, tavily_on):
        respx.post(TAVILY_URL).mock(side_effect=httpx.ConnectTimeout("timeout"))
        assert await TavilyProductSearch().search([("디저트", None)], 40000, 60000) == []

    @respx.mock
    async def test_one_category_failing_does_not_kill_the_rest(self, tavily_on):
        respx.post(TAVILY_URL).mock(
            side_effect=[
                httpx.Response(500, text="boom"),
                tavily_response(result("고급 타월", "https://www.11st.co.kr/products/3", "45,000원")),
            ]
        )
        products = await TavilyProductSearch().search(
            [("디저트", None), ("생활용품", None)], 40000, 60000
        )
        assert len(products) == 1
        assert products[0].category == "생활용품"

    @respx.mock
    async def test_ranking_words_in_a_title_do_not_exclude_a_detail_page(self, tavily_on):
        """'베스트'·'랭킹'은 쿠팡·네이버 상품 제목에 흔합니다.

        URL 이 상세페이지임을 보증하므로 제목 낱말만으로 버리면 정상 상품을 잃습니다.
        """
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result(
                    "베스트 랭킹1위 수제쿠키 선물세트 12,000원",
                    "https://www.coupang.com/vp/products/9",
                )
            )
        )
        products = await TavilyProductSearch().search([("디저트", None)], 9000, 14000, limit=2)

        assert [p.title for p in products] == ["베스트 랭킹1위 수제쿠키 선물세트 12,000원"]

    @respx.mock
    async def test_shipping_threshold_in_a_title_is_never_shown_as_price(self, tavily_on):
        """제목의 무료배송 기준액이 '검색 기준 약 50,000원(확인 필요)'로 노출됐습니다.

        그 숫자를 상품가로 읽지 않으므로 이 상품은 가격을 전혀 모르는 상태가 되고,
        가격을 모르는 상품은 노출하지 않습니다. 틀린 금액이 화면에 뜨는 것보다
        상품이 하나도 없는 편이 낫습니다.
        """
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result(
                    "50,000원 이상 구매 시 무료배송 수제쿠키 선물세트",
                    "https://www.coupang.com/vp/products/1",
                )
            )
        )
        products = await TavilyProductSearch().search([("디저트", None)], 9000, 14000, limit=1)

        assert products == []

    @respx.mock
    async def test_skips_results_without_url_or_title(self, tavily_on):
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                {"title": "", "url": "https://www.coupang.com/1", "content": ""},
                {"title": "제목만", "url": "", "content": ""},
                result("정상 디저트", "https://www.coupang.com/vp/products/2", "45,000원"),
            )
        )
        products = await TavilyProductSearch().search([("디저트", None)], 40000, 60000)
        assert [p.title for p in products] == ["정상 디저트"]


class TestRecommendationIntegration:
    """검색이 실패해도 추천 자체는 나가야 합니다."""

    def _recommendation(self) -> SimpleGiftRecommendationResponse:
        return SimpleGiftRecommendationResponse(
            input_gift_name="케이크",
            input_gift_price=35000,
            input_age=None,
            recommended_price_min=28000,
            recommended_price_max=42000,
            categories=[
                CategoryRecommendation(
                    category="디저트",
                    score=90,
                    reason="무난합니다",
                    product_examples=["프리미엄 디저트 세트"],
                )
            ],
            summary="요약",
            suggested_message="메시지" * 40,
        message_source=MessageSource.MODEL,
            model="gemma4-12b-qat",
            source="GEMMA_VLLM",
        )

    @respx.mock
    async def test_products_are_attached(self, tavily_on):
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("수제쿠키 선물세트 35,000원", "https://www.coupang.com/vp/products/1")
            )
        )
        products = await RecommendationPreparationService._find_products(
            self._recommendation(), SearchStats()
        )

        assert len(products) == 1
        assert products[0].price == 35000
        assert products[0].source == "쿠팡"

    async def test_no_products_when_search_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "tavily_enabled", False)
        stats = SearchStats()
        assert (
            await RecommendationPreparationService._find_products(
                self._recommendation(), stats
            )
            == []
        )

    @respx.mock
    async def test_search_failure_leaves_recommendation_intact(self, tavily_on):
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(503))
        recommendation = self._recommendation()
        recommendation.products = await RecommendationPreparationService._find_products(
            recommendation, SearchStats()
        )

        assert recommendation.products == []
        assert recommendation.categories[0].category == "디저트"
        assert recommendation.summary == "요약"


class TestPriceVerification:
    """검색 스니펫의 숫자는 같은 브랜드 다른 옵션의 가격일 수 있어 믿을 수 없습니다.

    실측: gift.kakao.com/product/2198213 의 실제 판매가는 39,000원인데
    스니펫에는 32,000 / 15,000 / 23,000 만 있고 39,000 은 없었습니다.
    """

    @respx.mock
    async def test_extract_overwrites_snippet_price(self, tavily_on, extract_on):
        url = "https://gift.kakao.com/product/2198213"
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(result("프리미엄 쿠키 선물", url, "판매가 32,000 원 15,000 원"))
        )
        respx.post(EXTRACT_URL).mock(
            return_value=extract_response((url, "기본 정보 가격 정보 판매가 39,000 원 배송비 3,000 원"))
        )
        products = await TavilyProductSearch().search([("디저트", None)], 30000, 50000, limit=1)

        assert products[0].price == 39000
        assert products[0].price_verified is True

    @respx.mock
    async def test_unverified_price_is_kept_but_flagged(self, tavily_on, extract_on):
        url = "https://www.coupang.com/vp/products/1"
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(result("쿠키 세트 32,000원", url))
        )
        respx.post(EXTRACT_URL).mock(return_value=extract_response((url, "가격 정보 없음")))
        products = await TavilyProductSearch().search([("디저트", None)], 30000, 50000, limit=1)

        # 지우지 않고 표시만 남깁니다. 화면에서 "약 32,000원(확인 필요)" 로 보여 줄 수 있습니다.
        assert products[0].price == 32000
        assert products[0].price_verified is False

    @respx.mock
    async def test_verified_in_range_ranks_first(self, tavily_on, extract_on):
        cheap = "https://www.coupang.com/vp/products/1"
        pricey = "https://www.coupang.com/vp/products/2"
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("쿠키 A", pricey), result("쿠키 B", cheap)
            )
        )
        respx.post(EXTRACT_URL).mock(
            return_value=extract_response(
                (pricey, "판매가 200,000 원"), (cheap, "판매가 40,000 원")
            )
        )
        products = await TavilyProductSearch().search([("디저트", None)], 30000, 50000, limit=2)

        assert products[0].price == 40000
        assert len(products) == 1

    @respx.mock
    async def test_shows_the_nearest_products_when_none_are_in_range(self, tavily_on, extract_on):
        """예산 안 상품이 없으면 **정말 가까운** 것만 참고용으로 제공합니다.

        가까운 순으로 limit 까지 채우는 동작은 그대로입니다. 다만 "가까운"의 기준이
        경계 ±15% 입니다. 20,000원(-33%)과 90,000원(+80%)은 30,000~50,000원 예산에
        가깝지 않습니다. 자리를 채우려고 그런 값을 내보낸 것이 실측의 +81%·+100%
        노출을 만들었습니다.
        """
        urls = [f"https://www.coupang.com/vp/products/{i}" for i in range(3)]
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                *(result(f"디저트 상품 {i}", url) for i, url in enumerate(urls))
            )
        )
        respx.post(EXTRACT_URL).mock(
            return_value=extract_response(
                (urls[0], "판매가 20,000 원"),
                (urls[1], "판매가 55,000 원"),
                (urls[2], "판매가 90,000 원"),
            )
        )
        products = await TavilyProductSearch().search(
            [("디저트", None)], 30000, 50000, limit=3
        )
        # 허용 폭은 25,500~57,500원. 55,000(+10%)만 남고 20,000(-33%)·90,000(+80%)은
        # 떨어집니다. 채울 것이 없으면 적게 나가는 편이 낫습니다.
        assert [p.price for p in products] == [55000]
        assert "가격대 안 상품을 찾지 못해" in products[0].reason

    @respx.mock
    async def test_extract_timeout_keeps_products(self, tavily_on, extract_on):
        """Extract 가 죽어도 검색 기준 가격이 있으면 참고용으로 나갑니다.

        가격을 **확정**하지 못한 것과 가격을 **전혀 모르는** 것은 다릅니다. 앞은
        참고용 표시를 달고 나가고, 뒤는 예산을 지켰는지 확인할 수 없어 나가지
        않습니다(아래 test_a_product_with_no_price_at_all_never_ships).
        """
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("쿠키 세트", "https://www.coupang.com/vp/products/1", "32,000원")
            )
        )
        respx.post(EXTRACT_URL).mock(side_effect=httpx.ReadTimeout("too slow"))
        products = await TavilyProductSearch().search([("디저트", None)], 30000, 50000, limit=1)

        assert len(products) == 1
        assert products[0].title == "쿠키 세트"
        assert products[0].price == 32000
        assert products[0].price_verified is False

    @respx.mock
    async def test_batches_are_split(self, tavily_on, extract_on, monkeypatch):
        """한 묶음에 몰아 보내면 느린 URL 하나가 나머지 결과까지 잃게 만듭니다."""
        monkeypatch.setattr(settings, "tavily_extract_batch_size", 2)
        urls = [f"https://www.coupang.com/vp/products/{i}" for i in range(4)]
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(*(result(f"디저트 상품{i}", u) for i, u in enumerate(urls)))
        )
        route = respx.post(EXTRACT_URL).mock(
            return_value=extract_response(*((u, "판매가 40,000 원") for u in urls))
        )
        await TavilyProductSearch().search([("디저트", None)], 30000, 50000, limit=4)

        # 4건이 묶음 크기 2로 나뉘어 두 번 호출됩니다.
        assert len(route.calls) == 2

    @respx.mock
    async def test_listing_pages_are_excluded_and_not_extracted(self, tavily_on, extract_on):
        """검색 결과 페이지는 최종 추천도 판매가 확인 대상도 아닙니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("디저트 | 검색", "https://gift.kakao.com/search?q=디저트")
            )
        )
        route = respx.post(EXTRACT_URL).mock(return_value=extract_response())
        products = await TavilyProductSearch().search([("디저트", None)], 30000, 50000, limit=1)

        assert len(route.calls) == 0
        assert products == []


class TestSuggestionCountIsFilled:
    """예산 안 후보가 1건이어도 상한(3)까지 채웁니다.

    실측 4회가 모두 정확히 1건이었습니다. 후보는 11·4·12·10건이었는데 예산 안이
    1건이라 거기서 끝났습니다.
    """

    @respx.mock
    async def test_one_in_budget_product_still_fills_the_limit(self, tavily_on, extract_on):
        urls = [f"https://www.coupang.com/vp/products/{i}" for i in range(4)]
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                *(result(f"수제쿠키 선물세트 {i}", url) for i, url in enumerate(urls))
            )
        )
        respx.post(EXTRACT_URL).mock(
            return_value=extract_response(
                (urls[0], "판매가 60,000 원"),  # +43%. 가깝지 않아 자리를 못 채웁니다
                (urls[1], "판매가 35,000 원"),  # 유일한 예산 안
                (urls[2], "판매가 25,000 원"),  # -11%
                (urls[3], "판매가 46,000 원"),  # +10%
            )
        )
        products = await TavilyProductSearch().search(
            [("디저트", None)], 28000, 42000, limit=3
        )

        # 허용 폭은 23,800~48,300원. 예산 안이 맨 앞, 나머지는 가까운 순
        # (25,000 은 3,000 / 46,000 은 4,000 차이). 60,000 은 폭 밖이라 빠집니다.
        assert [p.price for p in products] == [35000, 25000, 46000]
        assert "제안 가격대 안" in products[0].reason

    @respx.mock
    async def test_filler_products_say_they_are_outside_the_budget(self, tavily_on, extract_on):
        """자리를 채운 상품은 예산 밖이라는 사실이 reason 에 있어야 합니다."""
        urls = [f"https://www.coupang.com/vp/products/{i}" for i in range(2)]
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                *(result(f"수제쿠키 선물세트 {i}", url) for i, url in enumerate(urls))
            )
        )
        respx.post(EXTRACT_URL).mock(
            return_value=extract_response(
                (urls[0], "판매가 35,000 원"), (urls[1], "판매가 25,000 원")
            )
        )
        products = await TavilyProductSearch().search(
            [("디저트", None)], 28000, 42000, limit=3
        )

        assert "25,000원으로 제안 가격대보다 낮습니다" in products[1].reason
        # 예산 안 상품이 함께 나가므로 "찾지 못해" 가 아니라 "모자라" 입니다.
        assert "가격대 안 상품이 모자라" in products[1].reason
        assert "찾지 못해" not in products[1].reason

    @respx.mock
    async def test_unverified_filler_also_says_it_is_outside_the_budget(self, tavily_on):
        """미확인 가격도 예산 밖이면 그렇게 적어야 합니다.

        예전에는 "검색 기준 약 N원(확인 필요)" 에서 끝나 예산을 벗어났다는 말이
        빠졌습니다.
        """
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("수제쿠키 선물세트 35,000원", "https://www.coupang.com/vp/products/1"),
                result("수제쿠키 선물세트 25,000원", "https://www.coupang.com/vp/products/2"),
            )
        )
        products = await TavilyProductSearch().search(
            [("디저트", None)], 28000, 42000, limit=3
        )

        assert [p.price for p in products] == [35000, 25000]
        assert products[1].price_verified is False
        assert "검색 기준 약 25,000원(확인 필요)으로 제안 가격대보다 낮습니다" in products[1].reason

    @respx.mock
    async def test_far_products_never_fill_a_seat(self, tavily_on, extract_on):
        """8,000~12,000원 예산에 35,000원짜리로 자리를 채우면 안 됩니다.

        채울 것이 없으면 적게 나오는 편이 낫습니다.
        """
        urls = [f"https://www.coupang.com/vp/products/{i}" for i in range(3)]
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                *(result(f"수제쿠키 선물세트 {i}", url) for i, url in enumerate(urls))
            )
        )
        respx.post(EXTRACT_URL).mock(
            return_value=extract_response(
                (urls[0], "판매가 10,000 원"),
                (urls[1], "판매가 35,000 원"),
                (urls[2], "판매가 90,000 원"),
            )
        )
        products = await TavilyProductSearch().search(
            [("디저트", None)], 8000, 12000, limit=3
        )

        assert [p.price for p in products] == [10000]

    @respx.mock
    async def test_verified_in_budget_comes_before_unverified_in_budget(
        self, tavily_on, extract_on
    ):
        """빈자리를 채우더라도 확인된 가격이 앞에 옵니다."""
        urls = [f"https://www.coupang.com/vp/products/{i}" for i in range(2)]
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("수제쿠키 선물세트 30,000원", urls[0]),
                result("수제쿠키 선물세트 40,000원", urls[1]),
            )
        )
        # 두 번째만 상품 페이지에서 확인됩니다.
        respx.post(EXTRACT_URL).mock(
            return_value=extract_response((urls[1], "판매가 40,000 원"))
        )
        products = await TavilyProductSearch().search(
            [("디저트", None)], 28000, 42000, limit=3
        )

        assert [p.price_verified for p in products] == [True, False]
        assert [p.price for p in products] == [40000, 30000]

class TestMeasuredBudgetViolations:
    """2차 실측에서 실제로 화면에 나간 예산 위반을 그대로 재현합니다.

    노출 10건 중 예산 안은 3건(30%)뿐이었습니다. 1차는 4건 중 4건(100%)이었습니다.
    보충 폭이 "절반~두 배"(-50%~+100%)라 아래 값들이 전부 통과했습니다.
    """

    @respx.mock
    async def test_a_user_typed_budget_never_ships_a_product_81_percent_over(
        self, tavily_on, extract_on
    ):
        """실측: 사용자가 18,000~27,000원을 직접 지정했는데 49,000원이 나갔습니다.

        같은 응답의 product_basis 는 "0개가 18,000원 ~ 27,000원 안에 듭니다" 였습니다.
        화면에 숫자 두 개가 나란히 보이면 바로 들통납니다.
        """
        near = "https://www.11st.co.kr/products/8359844739"
        far = "https://www.11st.co.kr/products/4021010389"
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("드라이플라워 장미 꽃다발 미니 유리돔", near),
                result("로즈플로라 프리저브드 플라워 수제 꽃다발", far),
            )
        )
        respx.post(EXTRACT_URL).mock(
            return_value=extract_response(
                (near, "판매가 15,900 원"), (far, "판매가 49,000 원")
            )
        )
        products = await TavilyProductSearch().search([("꽃·식물", None)], 18000, 27000, limit=3)

        # 15,900원은 하한에서 -12% 라 가까운 축입니다. 49,000원(+81%)은 아닙니다.
        assert [p.price for p in products] == [15900]

    @respx.mock
    async def test_a_ten_thousand_won_budget_never_ships_a_24000_won_product(
        self, tavily_on, extract_on
    ):
        """실측 gift 웜: 8,000~12,000원 예산에 19,100원·23,990원만 나갔습니다.

        1차에서는 같은 실행이 9,800원 1건으로 예산 안이었습니다. 예산 안 후보가
        없으면 0건이 맞습니다. 없는 것을 있는 척하는 것보다 낫습니다.
        """
        urls = [f"https://gift.kakao.com/product/{i}" for i in range(2)]
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("[나폴레옹] 마들렌 세트 (5개입)", urls[0]),
                result("[스타벅스] 드립백 선물세트", urls[1]),
            )
        )
        respx.post(EXTRACT_URL).mock(
            return_value=extract_response(
                (urls[0], "판매가 19,100 원"), (urls[1], "판매가 23,990 원")
            )
        )
        products = await TavilyProductSearch().search([("디저트", None)], 8000, 12000, limit=3)

        assert products == []

    @respx.mock
    async def test_a_product_32_percent_below_the_budget_is_not_nearby(
        self, tavily_on, extract_on
    ):
        """실측 giftdata: 28,000~42,000원 예산에 19,100원이 3번째로 나갔습니다.

        받은 35,000원의 55% 를 답례로 권하는 셈이라 예산 위반이자 결례입니다.
        """
        urls = [f"https://gift.kakao.com/product/{i}" for i in range(3)]
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                *(result(f"디저트 선물세트 {i}", url) for i, url in enumerate(urls))
            )
        )
        respx.post(EXTRACT_URL).mock(
            return_value=extract_response(
                (urls[0], "판매가 35,000 원"),
                (urls[1], "판매가 39,000 원"),
                (urls[2], "판매가 19,100 원"),
            )
        )
        products = await TavilyProductSearch().search(
            [("디저트", None)], 28000, 42000, limit=3
        )

        assert [p.price for p in products] == [35000, 39000]

    @respx.mock
    async def test_a_product_with_no_price_at_all_never_ships(self, tavily_on, extract_on):
        """실측 3차 gift(콜드·웜 2/2): 8,000~12,000원 예산에 가격 미상 상품이 나갔습니다.

        노출된 것은 "[선물] 명품 나주배 세트 5kg(8-10과) 부모님 명절 선물" 한 건이고
        JSON 에는 price 키 자체가 없었으며, 같은 응답의 product_basis 는
        "0개가 8,000원 ~ 12,000원 안에 듭니다" 였습니다.

        직전 라운드는 "금액을 말하지 않으니 예산과 어긋날 수 없다"고 봤지만, 가격을
        모른다는 것은 어긋나지 않는다는 뜻이 아니라 **어긋났는지 확인할 방법이 없다**는
        뜻입니다. 예산은 이 서비스가 숫자로 내건 약속이라 확인할 수 없으면 내보내지
        않습니다.
        """
        priced = "https://www.coupang.com/vp/products/1"
        unknown = "https://www.coupang.com/vp/products/2"
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("비싼 쿠키 세트", priced), result("수제쿠키 선물세트", unknown)
            )
        )
        respx.post(EXTRACT_URL).mock(
            return_value=extract_response(
                (priced, "판매가 90,000 원"), (unknown, "가격 정보 없음")
            )
        )
        products = await TavilyProductSearch().search(
            [("디저트", None)], 8000, 12000, limit=3
        )

        assert products == []


class TestZeroProductsIsAnHonestAnswer:
    """가격 미상 상품을 뺀 대가로 0건이 자주 나옵니다. 그때 응답이 납득돼야 합니다.

    실측 3차 gift 는 후보 9건을 찾아 놓고 그중 어느 것도 예산(8,000~12,000원)에
    맞는 가격을 대지 못했습니다. 그때 "상품 검색 결과가 없어" 라고 말하면 새로운
    거짓말이 하나 늘어날 뿐입니다.
    """

    def _recommendation(self) -> SimpleGiftRecommendationResponse:
        return SimpleGiftRecommendationResponse(
            input_gift_name="빽다방 금액권",
            input_gift_price=10000,
            input_age=None,
            recommended_price_min=8000,
            recommended_price_max=12000,
            categories=[
                CategoryRecommendation(
                    category="디저트",
                    score=85,
                    reason="같은 카테고리로 답례합니다",
                    product_examples=["스페셜티 드립백 세트"],
                )
            ],
            summary="디저트 카테고리의 선물로 답례하는 것을 추천합니다.",
            suggested_message="메시지" * 40,
        message_source=MessageSource.MODEL,
            model="gemma4-12b-qat",
            source="GEMMA_VLLM",
        )

    @respx.mock
    async def test_search_reports_how_many_candidates_it_judged(self, tavily_on):
        """0건의 이유를 말하려면 후보가 몇 건이었는지를 밖으로 내야 합니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("스페셜티 드립백 선물세트", "https://www.ssg.com/item/itemView.ssg?itemId=1"),
                result("모모스 원두 선물세트", "https://gift.kakao.com/product/2"),
            )
        )
        stats = SearchStats()
        products = await TavilyProductSearch().search(
            [("디저트", None)], 8000, 12000, limit=3, stats=stats
        )

        assert products == []
        assert stats.examined == 2

    @respx.mock
    async def test_the_response_says_it_found_candidates_but_none_fit(self, tavily_on):
        """실측 gift 를 그대로 재현합니다. 세 문장이 서로를 뒷받침해야 합니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("스페셜티 드립백 선물세트", "https://www.ssg.com/item/itemView.ssg?itemId=1"),
                result("모모스 원두 선물세트", "https://gift.kakao.com/product/2"),
            )
        )
        request = SimpleGiftRecommendationRequest(gift_name="빽다방 금액권", gift_price=10000)
        recommendation = self._recommendation()
        stats = SearchStats()
        recommendation.products = await RecommendationPreparationService._find_products(
            recommendation, stats
        )
        info = RecommendationPreparationService._finalize(request, recommendation, stats)
        result_gift = info.recommend_gift

        assert result_gift.products == []
        assert result_gift.rationale.product_basis == (
            "상품 후보 2개를 찾았지만 8,000원 ~ 12,000원에 맞는 판매가를 확인하지 못해 "
            "카테고리와 가격대만 제안했습니다."
        )
        assert result_gift.rationale.warnings == [
            "제안 가격대에 맞는 상품을 확인하지 못해 이번에는 상품을 보여 드리지 못했습니다. "
            "카테고리와 가격대를 참고해 직접 골라 주세요."
        ]
        # 상품이 없으니 "이 가격대에서 상품이 나온 것은 …" 을 붙일 수 없습니다.
        assert "상품이 나온 것은" not in result_gift.rationale.category_basis
        # summary 는 카테고리 제안이라 0건과 어긋나지 않습니다. 손대지 않습니다.
        assert result_gift.summary == "디저트 카테고리의 선물로 답례하는 것을 추천합니다."

    @respx.mock
    async def test_an_empty_search_says_something_different(self, tavily_on):
        """검색이 정말 0건인 것과 찾았지만 안 맞는 것은 사용자에게 다른 말입니다."""
        respx.post(TAVILY_URL).mock(return_value=tavily_response())
        request = SimpleGiftRecommendationRequest(gift_name="빽다방 금액권", gift_price=10000)
        recommendation = self._recommendation()
        stats = SearchStats()
        recommendation.products = await RecommendationPreparationService._find_products(
            recommendation, stats
        )
        info = RecommendationPreparationService._finalize(request, recommendation, stats)

        assert stats.examined == 0
        assert info.recommend_gift.rationale.product_basis == (
            "상품 검색 결과가 없어 카테고리와 가격대만 제안했습니다."
        )


class TestHigherScoringCategoryComesFirst:
    """실측 gift 콜드: summary 는 "디저트를 최우선", 첫 상품은 생활용품 볼펜(60점).

    가격 적합성이 카테고리 점수보다 먼저입니다. 다만 **가격 조건이 같으면** 점수가
    높은 카테고리가 앞에 와야 합니다.
    """

    @respx.mock
    async def test_equal_price_fit_keeps_the_category_order(self, tavily_on, extract_on):
        coffee = "https://www.coupang.com/vp/products/1"
        living = "https://www.coupang.com/vp/products/2"
        respx.post(TAVILY_URL).mock(
            side_effect=[
                tavily_response(result("드립백 세트", coffee)),
                tavily_response(result("고급 타월 세트", living)),
            ]
        )
        respx.post(EXTRACT_URL).mock(
            return_value=extract_response(
                (coffee, "판매가 11,000 원"), (living, "판매가 9,800 원")
            )
        )
        products = await TavilyProductSearch().search(
            [("디저트", None), ("생활용품", None)], 8000, 12000, limit=3
        )

        # 둘 다 확인된 판매가가 예산 안이라 조건이 같습니다. 점수가 높아 먼저 넘어온
        # 디저트가 앞입니다. 9,800원이 더 싸다는 것은 순위 기준이 아닙니다.
        assert [p.category for p in products] == ["디저트", "생활용품"]

    @respx.mock
    async def test_price_fit_still_beats_the_category_score(self, tavily_on, extract_on):
        """실측 그대로: 디저트(85)가 23,990원, 생활용품(60)이 9,800원.

        점수를 앞에 두면 8,000~12,000원 예산에 23,990원짜리가 1번이 됩니다.
        그게 이 라운드에 고친 예산 위반 그 자체입니다.
        """
        coffee = "https://www.coupang.com/vp/products/1"
        living = "https://www.coupang.com/vp/products/2"
        respx.post(TAVILY_URL).mock(
            side_effect=[
                tavily_response(result("드립백 세트", coffee)),
                tavily_response(result("고급 타월 세트", living)),
            ]
        )
        respx.post(EXTRACT_URL).mock(
            return_value=extract_response(
                (coffee, "판매가 13,500 원"), (living, "판매가 9,800 원")
            )
        )
        products = await TavilyProductSearch().search(
            [("디저트", None), ("생활용품", None)], 8000, 12000, limit=3
        )

        assert [p.price for p in products] == [9800, 13500]


class TestProductReason:
    @respx.mock
    async def test_reason_explains_price_fit(self, tavily_on, extract_on):
        url = "https://www.coupang.com/vp/products/1"
        respx.post(TAVILY_URL).mock(return_value=tavily_response(result("쿠키 세트", url)))
        respx.post(EXTRACT_URL).mock(return_value=extract_response((url, "판매가 40,000 원")))
        products = await TavilyProductSearch().search([("디저트", None)], 30000, 50000, limit=1)

        assert "제안 가격대 안" in products[0].reason
        assert "쿠팡" in products[0].reason

    @respx.mock
    async def test_reason_names_the_category_without_saying_recommend_twice(
        self, tavily_on, extract_on
    ):
        """'디저트 추천에 맞는' 은 겹말입니다."""
        url = "https://www.coupang.com/vp/products/1"
        respx.post(TAVILY_URL).mock(return_value=tavily_response(result("쿠키 세트", url)))
        respx.post(EXTRACT_URL).mock(return_value=extract_response((url, "판매가 40,000 원")))
        products = await TavilyProductSearch().search([("디저트", None)], 30000, 50000, limit=1)

        assert products[0].reason.startswith("디저트 선물로 고른 쿠팡 상품")
        assert "추천에 맞는" not in products[0].reason

    @respx.mock
    async def test_reason_flags_over_budget(self, tavily_on, extract_on):
        url = "https://www.coupang.com/vp/products/1"
        respx.post(TAVILY_URL).mock(return_value=tavily_response(result("비싼 디저트 세트", url)))
        # 200,000원(+300%)은 이제 노출 자체가 되지 않습니다. reason 문구를 확인하려면
        # 실제로 나갈 수 있는 값, 즉 경계에서 15% 안쪽이어야 합니다.
        respx.post(EXTRACT_URL).mock(return_value=extract_response((url, "판매가 55,000 원")))
        products = await TavilyProductSearch().search([("디저트", None)], 30000, 50000, limit=1)

        assert "높습니다" in products[0].reason


# ------------------------------------------------- 상품 페이지 직접 조회로 판매가 확인
# Tavily Extract 는 페이지를 마크다운으로 바꾸며 HTML 안의 가격 데이터를 버립니다.
# 실측에서 컬리 드립백 세트(실제 55,000원)는 Extract 본문에 단가 11,000원과 배송비만
# 남아, 55,000원짜리를 8,000~12,000원 예산에 맞는 상품으로 보여줬습니다.

KURLY_HTML = '<html><body><script>{"salesPrice":55000,"name":"드립백"}</script></body></html>'
JSONLD_HTML = """<html><head>
<script type="application/ld+json">
{"@type": "Product", "name": "카스테라", "offers": {"price": 25900, "priceCurrency": "KRW"}}
</script></head></html>"""
# 무엇의 가격인지 규격이 보장하지 않는 임의 키는 믿지 않습니다.
AMBIGUOUS_HTML = '<html><body><script>{"finalPrice":41900,"dispPrice":25900}</script></body></html>'


@pytest.mark.asyncio
@respx.mock
async def test_direct_fetch_reads_site_specific_price():
    respx.get("https://www.kurly.com/goods/1").mock(
        return_value=httpx.Response(200, html=KURLY_HTML)
    )
    async with httpx.AsyncClient() as client:
        price = await product_search_module.fetch_price_direct(
            "https://www.kurly.com/goods/1", client
        )
    assert price == 55000


@pytest.mark.asyncio
@respx.mock
async def test_direct_fetch_reads_jsonld_product_offer():
    respx.get("https://www.11st.co.kr/products/1").mock(
        return_value=httpx.Response(200, html=JSONLD_HTML)
    )
    async with httpx.AsyncClient() as client:
        price = await product_search_module.fetch_price_direct(
            "https://www.11st.co.kr/products/1", client
        )
    assert price == 25900


@pytest.mark.asyncio
@respx.mock
async def test_direct_fetch_ignores_ambiguous_price_keys():
    respx.get("https://gift.kakao.com/product/1").mock(
        return_value=httpx.Response(200, html=AMBIGUOUS_HTML)
    )
    async with httpx.AsyncClient() as client:
        price = await product_search_module.fetch_price_direct(
            "https://gift.kakao.com/product/1", client
        )
    assert price is None


@pytest.mark.asyncio
@respx.mock
async def test_direct_fetch_gives_up_quietly_when_blocked():
    """쿠팡·SSG·G마켓은 봇 차단으로 403 입니다. 실패해도 Extract 로 넘어가야 합니다."""
    respx.get("https://www.coupang.com/vp/products/1").mock(
        return_value=httpx.Response(403)
    )
    async with httpx.AsyncClient() as client:
        price = await product_search_module.fetch_price_direct(
            "https://www.coupang.com/vp/products/1", client
        )
    assert price is None


@pytest.mark.asyncio
async def test_direct_fetch_refuses_domains_outside_the_whitelist():
    """검색 결과만 들어오지만, 이 함수만 보고도 안전해야 합니다."""
    async with httpx.AsyncClient() as client:
        assert await product_search_module.fetch_price_direct(
            "https://evil.example.com/goods/1", client
        ) is None


# ---------------------------------------- 이미지에 금액이 없을 때 상품명으로 판매가 검색
# 카테고리 추정가는 브랜드를 모릅니다. 실측에서 TWG Tea 티백 선물이 "음료"로 분류돼
# 10,000원으로 추정됐지만 실제 판매가는 36,000~76,000원이었습니다.

def tavily_results(urls: list[str]) -> httpx.Response:
    return httpx.Response(200, json={"results": [{"url": u, "title": "t", "content": ""} for u in urls]})


@pytest.mark.asyncio
@respx.mock
async def test_lookup_price_returns_the_median_of_found_prices(monkeypatch):
    """같은 브랜드의 다른 용량이 섞이므로 중앙값을 씁니다."""
    monkeypatch.setattr(settings, "tavily_enabled", True)
    monkeypatch.setattr(settings, "tavily_api_key", "k")
    urls = [f"https://www.kurly.com/goods/{i}" for i in (1, 2, 3)]
    respx.post(settings.tavily_url).mock(return_value=tavily_results(urls))
    prices = iter([32_000, 36_690, 73_150])

    async def fake_fetch(url, client):
        return next(prices)

    monkeypatch.setattr(product_search_module, "fetch_price_direct", fake_fetch)

    assert await product_search_module.lookup_price("TWG Tea Teabags Collection") == 36_690


@pytest.mark.asyncio
@respx.mock
async def test_lookup_price_prefixes_the_brand_only_when_missing(monkeypatch):
    monkeypatch.setattr(settings, "tavily_enabled", True)
    monkeypatch.setattr(settings, "tavily_api_key", "k")
    route = respx.post(settings.tavily_url).mock(return_value=tavily_results([]))

    await product_search_module.lookup_price("Teabags Collection", "TWG Tea")
    await product_search_module.lookup_price("TWG Tea Teabags", "TWG Tea")

    queries = [json.loads(call.request.content)["query"] for call in route.calls]
    assert queries == ["TWG Tea Teabags Collection", "TWG Tea Teabags"]


@pytest.mark.asyncio
@respx.mock
async def test_lookup_price_gives_up_quietly_when_no_price_is_readable(monkeypatch):
    """못 찾으면 None 입니다. 값을 지어내지 않습니다."""
    monkeypatch.setattr(settings, "tavily_enabled", True)
    monkeypatch.setattr(settings, "tavily_api_key", "k")
    respx.post(settings.tavily_url).mock(
        return_value=tavily_results(["https://www.kurly.com/goods/1"])
    )

    async def fake_fetch(url, client):
        return None

    monkeypatch.setattr(product_search_module, "fetch_price_direct", fake_fetch)

    assert await product_search_module.lookup_price("알 수 없는 상품") is None


@pytest.mark.asyncio
async def test_lookup_price_is_skipped_when_search_is_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "tavily_enabled", False)
    assert await product_search_module.lookup_price("TWG Tea") is None


# ------------------------------------------------------ 검색 횟수 예산과 병렬 실행
# Tavily Search 는 1회가 1크레딧입니다. 몇 번 부르는지는 비용이자 후보 수입니다.


def recommendation_with(
    categories: list[str], low: int = 18000, high: int = 27000
) -> SimpleGiftRecommendationResponse:
    """정책이 만드는 것과 같은 모양의 추천 결과. 상품 유형은 SAFE_EXAMPLES 입니다."""
    return SimpleGiftRecommendationResponse(
        input_gift_name="꽃",
        input_gift_price=23333,
        input_age=32,
        recommended_price_min=low,
        recommended_price_max=high,
        categories=[
            CategoryRecommendation(
                category=name,
                score=90,
                reason="무난합니다",
                product_examples=list(SAFE_EXAMPLES[name]),
            )
            for name in categories
        ],
        summary="요약",
        suggested_message="메시지" * 40,
        message_source=MessageSource.MODEL,
        model="gemma4-12b-qat",
        source="GEMMA_VLLM",
    )


def queries_of(route) -> list[str]:
    return [json.loads(call.request.content)["query"] for call in route.calls]


class TestSearchBudget:
    """카테고리가 적어도 검색 예산 3회를 채웁니다."""

    def test_two_categories_use_a_second_seed_to_fill_the_budget(self):
        """3 // 2 = 1 이라 카테고리 2개면 2회에서 멈춰 예산이 남았습니다."""
        targets = _search_targets([("디저트", ["A", "B"]), ("상품권", ["C", "D"])])

        assert targets == [("디저트", "A"), ("상품권", "C"), ("디저트", "B")]

    def test_never_exceeds_the_budget(self):
        targets = _search_targets(
            [("a", ["1", "2"]), ("b", ["3", "4"]), ("c", ["5", "6"])]
        )

        assert [name for name, _ in targets] == ["a", "b", "c"]

    def test_too_few_seeds_simply_search_less(self):
        """씨앗이 모자라면 못 채웁니다. 없는 유형을 지어내 크레딧을 쓰지 않습니다."""
        assert _search_targets([("상품권", ["외식 상품권", "문화생활 상품권"])]) == [
            ("상품권", "외식 상품권"),
            ("상품권", "문화생활 상품권"),
        ]

    def test_category_without_seeds_still_gets_one_search(self):
        assert _search_targets([("상품권", [])]) == [("상품권", None)]

    @respx.mock
    async def test_two_categories_produce_three_searches(self, tavily_on):
        route = respx.post(TAVILY_URL).mock(return_value=tavily_response())

        await RecommendationPreparationService._find_products(
            recommendation_with(["디저트", "상품권"]), SearchStats()
        )

        assert queries_of(route) == [
            "프리미엄 디저트 세트 선물 2만원대",
            "외식 상품권 선물 2만원대",
            "제철 과일 세트 선물 2만원대",
        ]


class TestConcurrentSearch:
    """예산과 카테고리가 사용자 입력으로 확정되면 검색은 추천 모델을 기다리지 않습니다."""

    def request(self, **overrides) -> RecommendRequest:
        payload = {
            "age": 32,
            "budget_min": 18000,
            "budget_max": 27000,
            "categories": ["꽃·식물"],
            "gift_price": 23333,
        }
        payload.update(overrides)
        return RecommendRequest(**payload)

    @respx.mock
    async def test_search_starts_before_the_model_answers(self, tavily_on, monkeypatch):
        searching = threading.Event()
        route = respx.post(TAVILY_URL).mock(
            side_effect=lambda request: (
                searching.set(),
                tavily_response(
                    result("미니 꽃다발 20,000원", "https://www.coupang.com/vp/products/1")
                ),
            )[1]
        )

        def fake_recommend(request):
            # 순차 실행이면 검색은 이 함수가 끝나야 시작하므로 여기서 시간이 다 갑니다.
            assert searching.wait(timeout=5), "검색이 모델 호출과 동시에 시작되지 않았습니다."
            return recommendation_with(["꽃·식물"])

        monkeypatch.setattr(qwen_service, "recommend_simple", fake_recommend)

        info = await recommendation_preparation_service.recommend_only(self.request())

        assert [p.title for p in info.recommend_gift.products] == ["미니 꽃다발 20,000원"]
        assert queries_of(route) == [
            "미니 꽃다발 선물 2만원대",
            "관리하기 쉬운 화분 선물 2만원대",
        ]

    @respx.mock
    async def test_parallel_search_uses_the_categories_the_user_chose(
        self, tavily_on, monkeypatch
    ):
        """모델이 다른 카테고리를 내도 검색은 사용자가 고른 카테고리로 갑니다.

        정규화가 지정 카테고리로 좁히므로 대개 같아지지만, 모델이 지정 카테고리를
        하나도 고르지 않으면 응답 카테고리와 검색 카테고리가 갈릴 수 있습니다.
        """
        route = respx.post(TAVILY_URL).mock(return_value=tavily_response())
        monkeypatch.setattr(
            qwen_service, "recommend_simple", lambda request: recommendation_with(["디저트"])
        )

        await recommendation_preparation_service.recommend_only(
            self.request(categories=["상품권"])
        )

        assert queries_of(route) == [
            "외식 상품권 선물 2만원대",
            "문화생활 상품권 선물 2만원대",
            "커피 기프티콘 선물 2만원대",
        ]

    @respx.mock
    async def test_alias_categories_are_resolved_before_searching(
        self, tavily_on, monkeypatch
    ):
        route = respx.post(TAVILY_URL).mock(return_value=tavily_response())
        monkeypatch.setattr(
            qwen_service, "recommend_simple", lambda request: recommendation_with(["패션·잡화"])
        )

        await recommendation_preparation_service.recommend_only(
            self.request(categories=["화장품"])
        )

        assert queries_of(route)[0].startswith("카드지갑")

    @respx.mock
    async def test_without_categories_the_model_decides_first(self, tavily_on, monkeypatch):
        """조건이 확정되지 않으면 예전처럼 모델 응답을 받고 나서 검색합니다."""
        answered = threading.Event()

        def fake_recommend(request):
            answered.set()
            return recommendation_with(["디저트"])

        monkeypatch.setattr(qwen_service, "recommend_simple", fake_recommend)

        def tavily(request):
            assert answered.is_set(), "모델보다 먼저 검색하면 카테고리를 알 수 없습니다."
            return tavily_response()

        route = respx.post(TAVILY_URL).mock(side_effect=tavily)

        await recommendation_preparation_service.recommend_only(
            self.request(categories=[])
        )

        assert queries_of(route)[0].startswith("프리미엄 디저트 세트")

    @respx.mock
    async def test_unknown_category_falls_back_to_the_sequential_path(
        self, tavily_on, monkeypatch
    ):
        """허용 목록에 없는 이름은 씨앗이 없으므로 모델 응답을 기다립니다."""
        answered = threading.Event()

        def fake_recommend(request):
            answered.set()
            return recommendation_with(["디저트"])

        monkeypatch.setattr(qwen_service, "recommend_simple", fake_recommend)

        def tavily(request):
            assert answered.is_set(), "확정되지 않은 조건으로 먼저 검색했습니다."
            return tavily_response()

        respx.post(TAVILY_URL).mock(side_effect=tavily)

        await recommendation_preparation_service.recommend_only(
            self.request(categories=["없는카테고리"])
        )

    @respx.mock
    async def test_model_failure_still_surfaces_in_the_parallel_path(
        self, tavily_on, monkeypatch
    ):
        respx.post(TAVILY_URL).mock(return_value=tavily_response())

        def boom(request):
            raise RuntimeError("model down")

        monkeypatch.setattr(qwen_service, "recommend_simple", boom)

        with pytest.raises(RuntimeError):
            await recommendation_preparation_service.recommend_only(self.request())

    @respx.mock
    async def test_search_failure_leaves_the_recommendation_intact(
        self, tavily_on, monkeypatch
    ):
        respx.post(TAVILY_URL).mock(side_effect=httpx.ConnectTimeout("timeout"))
        monkeypatch.setattr(
            qwen_service, "recommend_simple", lambda request: recommendation_with(["꽃·식물"])
        )

        info = await recommendation_preparation_service.recommend_only(self.request())

        assert info.recommend_gift.products == []
        assert info.recommend_gift.categories[0].category == "꽃·식물"


# ------------------------------------------------------ 실측(2026-08-20 E2E)에서 드러난 문제
# 3회 요청 모두 최종 상품이 1건이었고 그 1건이 예산을 벗어났습니다. 판정 통과 5건이
# 어디서 사라졌는지 로그로 알 수 없었습니다. 아래는 그 상황을 그대로 재현합니다.

REAL_JUNK_SNIPPET = (
    "Title: 드라이플라워 프리저브드플라워 장미 꽃다발 시들지 않는 꽃 선물 미니 유리돔\n"
    ":   고객이 판매자의 서비스를 평가한 리뷰 중, 4~5점의 긍정 평가의 비율 (최근 1년 기준). "
    ":   상품 Q&A 문의에 24시간 내 응답한 비율 (최근 30일 기준). ## 상품 카테고리 정보. "
    "* 드라이플라워 프리저브드플라워 장미 꽃다발 시들지 않는 꽃 선물 미"
)
REAL_GOOD_SNIPPET = (
    "과일의 풍미를 담아 스페셜티 드립백을 만드는 푸룻티커피를 만나 보세요. "
    "입에 닿는 순간 진한 과실향을 느낄 수 있는 티 블렌딩 드립백을 다채롭게 담았어요. "
    "배송비 3,000원."
)


class TestSnippetCleaning:
    """스크래핑 원문이 상품 설명 자리에 그대로 나가면 안 됩니다."""

    def test_scraping_boilerplate_becomes_nothing(self):
        """마크다운 잔재와 판매자 평점 안내문만 남으면 비웁니다."""
        assert clean_snippet(REAL_JUNK_SNIPPET, "드라이플라워 프리저브드플라워 장미 꽃다발") is None

    def test_real_description_survives_without_markup(self):
        cleaned = clean_snippet(REAL_GOOD_SNIPPET, "[선물세트] 푸룻티 커피 드립백 3종 세트")

        assert cleaned.startswith("과일의 풍미를 담아")
        assert "배송비" not in cleaned  # 상품 설명이 아닙니다.
        assert "#" not in cleaned and "*" not in cleaned

    def test_cuts_on_sentence_boundaries_within_the_schema_limit(self):
        """낱말 중간에서 끊긴 문장이 화면에 나갔습니다."""
        long_text = " ".join(f"{i}번째 문장은 이 상품의 구성을 설명합니다." for i in range(20))
        cleaned = clean_snippet(long_text, "제목")

        assert len(cleaned) <= 200
        assert cleaned.endswith("설명합니다.")

    def test_measured_scraped_field_is_dropped(self):
        """3차 실측 노출값입니다. 앞 쉼표와 '상품명 :' 라벨이 그대로 나갔습니다."""
        assert (
            clean_snippet(
                ", 상품명 :국내생산타월의품격",
                "2P 볼라 고급 수건 선물 세트 40수 수건 210g 이사 생일 답례 신혼",
            )
            is None
        )

    def test_the_second_round_price_field_is_the_same_family(self):
        """2차 실측 ', 판매가 :49,800 원 무료배송 장바구니 담기 ...' 도 같은 표 조각입니다."""
        assert clean_snippet(", 판매가 :49,800 원 무료배송 장바구니 담기 ...", "수건 선물세트") is None

    def test_measured_point_notice_is_not_a_description(self):
        """2차 실측: 11번가 상품 설명 자리에 적립 포인트 안내 네 문장이 나갔습니다."""
        measured = (
            "최대 적립 포인트 안내 11pay 신한은행 계좌이체 결제 시 구매적립 포인트 2%"
            "(기본 구매 적립 포함)가 적립됩니다. 11번가 신한카드 결제 시 적립 포인트는 "
            "전월 실적에 따라 차등 적립됩니다. 적립 포인트는 최종 결제하는 금액에 따라 "
            "달라질 수 있으니, 정확한 적립 포인트는 결제 페이지에서 확인해주세요."
        )
        assert clean_snippet(measured, "[11번가] [용문전통시장] 로즈플로라 수제 꽃다발") is None

    def test_a_description_with_commas_is_not_broken(self):
        """3차 실측 정상 설명. 쉼표로 쪼개면 10자 미만 조각으로 흩어져 통째로 사라집니다."""
        measured = (
            "과일의 풍미를 담아 스페셜티 드립백을 만드는 푸룻티커피를 만나 보세요. "
            "향료를 첨가하지 않고 사과, 살구, 복숭아 등의 과일과 찻잎을 사용해 완성했기에, "
            "뚜렷한 풍미를 자랑하면서도 향이 자연스럽게 배어들어 있죠."
        )
        cleaned = clean_snippet(measured, "[선물세트] 푸룻티 커피 드립백 3종 세트 (21개입)")

        assert cleaned == measured
        assert "사과, 살구, 복숭아" in cleaned

    def test_title_repetition_is_not_a_description(self):
        assert clean_snippet("수제쿠키 선물세트 12,000원", "수제쿠키 선물세트 12,000원") is None

    def test_gift_card_wording_is_not_treated_as_boilerplate(self):
        """'교환'·'쿠폰'은 상품권 카테고리에서는 상품 설명입니다."""
        cleaned = clean_snippet(
            "전국 백화점과 이마트에서 교환 가능한 쿠폰으로 사용하실 수 있습니다. 반품은 불가합니다.",
            "신세계상품권 5만원권",
        )

        assert cleaned == "전국 백화점과 이마트에서 교환 가능한 쿠폰으로 사용하실 수 있습니다."

    @respx.mock
    async def test_search_stores_a_cleaned_snippet(self, tavily_on):
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result(
                    "드라이플라워 꽃다발 미니 유리돔 19,900원",
                    "https://www.11st.co.kr/products/8359844739",
                    REAL_JUNK_SNIPPET,
                )
            )
        )
        products = await TavilyProductSearch().search([("꽃·식물", None)], 18000, 27000, limit=1)

        assert products[0].snippet is None


# 2차 실측(gift 콜드·웜 양쪽)에 그대로 나간 135자 제목입니다. 검색 페이지의 <title>
# 이 통째로 실려 와 상품명 뒤가 전부 목록 부스러기입니다. 1건만 노출하던 1차에서는
# 안 보였지만 2~3건이 되면서 화면에 떴습니다.
REAL_JUNK_TITLE = (
    "[스타벅스] 드립백 선물세트 - SSG.COM드립백 선물세트 - 추천•인기 상품, "
    "신세계몰드립백선물세트 - 추천•인기 상품, 이마트몰드립백 세트 - 추천•인기 상품, "
    "이마트몰드립백 - 추천•인기 상품, 신세계백화점G마켓 - 드립백세트 검색결과"
)
# 같은 실측 회차의 정상 제목들. 하나라도 망가지면 안 됩니다.
REAL_GOOD_TITLES = (
    "[나폴레옹] 마들렌 세트 (5개입)",
    "[ 삼청동 소샌드 흑임자 12개입 ] 프리미엄 쿠키 선물 l 달콤한 하루",
    "호버펜2.0 인터스텔라 에디션 23.5도의 무중력 고급 볼펜",
    "드라이플라워 프리저브드플라워 장미 꽃다발 시들지 않는 꽃 선물 미니 유리돔",
    "[11번가] [용문전통시장] 로즈플로라 프리저브드 플라워 수제 꽃다발",
    "[센터커피] No.7 스페셜티 커피 드립백 (10g X 7개)",
    "[T멤버십10%+선물]키즈 원리셈 5.6세 세트 (전6권)",
    '[플랜테리어/가드닝] 우드 원형 화분 받침대 3단 (소-95mm) "품절대란/인기템/스툴/집들이"',
    "크리스마스 트리 미니트리 풀세트 눈꽃 / 홀리데이 선물 겨울 집들이 졸업 졸업식 꽃 선물 벽트리 장식 이사",
    # 판매처 이름이 **맨 앞**에 오는 정상 제목. 실측 최장(63자)이기도 합니다.
    "G마켓 - PP 팬시 쇼핑백 10p 세트 18x14x6.5cm/쇼핑/백/선물/포장/봉투/고급/선물포장/가방/쇼핑팩",
)


class TestTitleCleaning:
    """제목은 상품을 알아보는 유일한 수단이라 비울 수 없습니다. 대신 자릅니다."""

    def test_measured_scraping_junk_is_cut_to_the_product_name(self):
        assert clean_title(REAL_JUNK_TITLE) == "[스타벅스] 드립백 선물세트"

    @pytest.mark.parametrize("title", REAL_GOOD_TITLES)
    def test_real_titles_are_left_alone(self, title):
        assert clean_title(title) == title

    def test_a_trailing_shop_name_is_dropped(self):
        assert (
            clean_title("[선물세트] 푸룻티 커피 드립백 3종 세트 (21개입) - 마켓컬리")
            == "[선물세트] 푸룻티 커피 드립백 3종 세트 (21개입)"
        )

    def test_the_kakao_page_title_suffix_is_dropped(self):
        """잘라 낸 자리에 드러난 말줄임표도 함께 다듬습니다."""
        assert (
            clean_title("[커피 리브레] 디카페인 블렌드 나이트호크 드립백 (7 ... - 상품 : 선물하기")
            == "[커피 리브레] 디카페인 블렌드 나이트호크 드립백 (7"
        )

    def test_a_measured_trailing_ellipsis_is_trimmed(self):
        """3차 실측 45자 제목. _TITLE_MAX(80) 절단이 아니라 원본에 있던 말줄임표입니다."""
        assert (
            clean_title("2P 볼라 고급 수건 선물 세트 40수 수건 210g 이사 생일 답례 신혼 ...")
            == "2P 볼라 고급 수건 선물 세트 40수 수건 210g 이사 생일 답례 신혼"
        )

    def test_a_single_trailing_period_is_left_alone(self):
        """점 하나로 끝나는 제목은 잘린 흔적이 아닙니다."""
        assert clean_title("드립백 세트 vol.") == "드립백 세트 vol."

    def test_a_cut_that_leaves_nothing_useful_keeps_the_original(self):
        """실측 '원 - 상품 : 선물하기'. 자르면 '원' 만 남아 상품을 알 수 없습니다."""
        assert clean_title("원 - 상품 : 선물하기") == "원 - 상품 : 선물하기"

    def test_a_long_title_is_cut_on_a_boundary_not_mid_word(self):
        """위 두 규칙이 못 잡은 반복 구간을 막는 마지막 그물입니다."""
        original = " ".join(["프리미엄 수제쿠키 선물세트"] * 8)
        cleaned = clean_title(original)

        assert len(cleaned) <= 80
        # 앞에서부터 자른 것이고, 낱말 중간에서 끊기지 않았습니다.
        assert original.startswith(cleaned)
        assert original[len(cleaned)] == " "

    @respx.mock
    async def test_search_stores_a_cleaned_title(self, tavily_on):
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result(
                    REAL_JUNK_TITLE,
                    "https://www.ssg.com/item/itemView.ssg?itemId=1",
                    "9,800원",
                )
            )
        )
        products = await TavilyProductSearch().search([("디저트", None)], 8000, 12000, limit=1)

        assert products[0].title == "[스타벅스] 드립백 선물세트"


class TestWhereCandidatesGo:
    """단계마다 몇 건이 왜 떨어졌는지 로그로 남아야 합니다."""

    @respx.mock
    async def test_selection_logs_the_drop_it_causes(self, tavily_on, extract_on, caplog):
        """판정 통과 5건이 노출 1건이 된 이유가 로그에 없었습니다."""
        urls = [f"https://www.coupang.com/vp/products/{i}" for i in range(5)]
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                *(result(f"수제쿠키 선물세트 {i}", url) for i, url in enumerate(urls))
            )
        )
        # 실측처럼 확인된 판매가가 모두 예산(8,000~12,000) 밖입니다.
        respx.post(EXTRACT_URL).mock(
            return_value=extract_response(*((url, "판매가 35,000 원") for url in urls))
        )
        with caplog.at_level(logging.INFO, logger="app.services.product_search"):
            products = await TavilyProductSearch().search(
                [("디저트", None)], 8000, 12000, limit=3
            )

        # 35,000원은 8,000~12,000원에서 +192% 입니다. 가까운 것이 아니므로 한 건도
        # 나가지 않습니다. 그래도 왜 0건인지는 로그에 남아야 합니다.
        assert products == []
        selection = [r.message for r in caplog.records if "상품 선별" in r.message]
        assert len(selection) == 1
        # 라벨이 실제 상황을 말해야 합니다. 여기서는 후보도 있고 확인된 판매가도
        # 있었으며, 그 값이 전부 예산 밖이었습니다. "예산 근처 후보 없음" 은 후보가
        # 없었다는 뜻으로 읽혀 4차 진단을 엉뚱한 곳으로 보냈습니다.
        assert "가격을 아는 후보 5건(확인 5건)이 모두 예산 밖" in selection[0]
        assert "가장 가까운 값=35,000" in selection[0]
        assert "후보 5건 → 노출 0건(탈락 5건)" in selection[0]

    @respx.mock
    async def test_dropped_addresses_are_named_so_the_gap_can_be_found(
        self, tavily_on, caplog
    ):
        """4차 실측에서 원본 34건 중 24건이 여기서 조용히 사라졌습니다.

        개수만 남아 있어 _is_product_detail_url 의 어느 패턴이 모자란지 알 수
        없었고, 후보 부족이 상품 0건의 실제 원인인데 진단이 막혔습니다.
        """
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("디저트 기획전", "https://guide.coupang.com/gift"),
                result("수제쿠키 세트 12,000원", "https://www.coupang.com/vp/products/9"),
            )
        )
        with caplog.at_level(logging.INFO, logger="app.services.product_search"):
            await TavilyProductSearch().search([("디저트", None)], 9000, 14000, limit=3)

        search_log = next(r.message for r in caplog.records if "상품 검색" in r.message)
        assert "버린 주소=" in search_log
        assert "https://guide.coupang.com/gift" in search_log
        # 통과한 주소는 버린 목록에 없어야 합니다.
        assert "vp/products/9" not in search_log.split("버린 주소=")[1]

    @respx.mock
    async def test_zero_products_says_candidates_had_no_price_at_all(
        self, tavily_on, caplog
    ):
        """"예산 근처 후보 없음" 은 후보가 없었다는 뜻으로 읽힙니다.

        가격을 아무도 모르는 것과 가격은 아는데 예산 밖인 것은 고칠 자리가
        다릅니다. 앞은 가격 확인 경로, 뒤는 검색어와 예산의 어긋남입니다.
        """
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                *(
                    result(f"수제쿠키 선물세트 {i}", f"https://www.coupang.com/vp/products/{i}")
                    for i in range(3)
                )
            )
        )
        with caplog.at_level(logging.INFO, logger="app.services.product_search"):
            products = await TavilyProductSearch().search(
                [("디저트", None)], 8000, 12000, limit=3
            )

        assert products == []
        selection = next(r.message for r in caplog.records if "상품 선별" in r.message)
        assert "후보 3건 전원 판매가 미상" in selection

    @respx.mock
    async def test_zero_products_says_nothing_survived_the_detail_page_check(
        self, tavily_on, caplog
    ):
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("디저트 기획전", "https://guide.coupang.com/gift"),
                result("디저트 검색", "https://gift.kakao.com/search?q=x"),
            )
        )
        with caplog.at_level(logging.INFO, logger="app.services.product_search"):
            products = await TavilyProductSearch().search(
                [("디저트", None)], 8000, 12000, limit=3
            )

        assert products == []
        selection = next(r.message for r in caplog.records if "상품 선별" in r.message)
        assert "상세페이지 후보 없음" in selection

    @respx.mock
    async def test_search_logs_results_that_are_not_detail_pages(self, tavily_on, caplog):
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("디저트 기획전", "https://guide.coupang.com/gift"),
                result("디저트 검색", "https://gift.kakao.com/search?q=x"),
                result("수제쿠키 세트 12,000원", "https://www.coupang.com/vp/products/9"),
            )
        )
        with caplog.at_level(logging.INFO, logger="app.services.product_search"):
            await TavilyProductSearch().search([("디저트", None)], 9000, 14000, limit=3)

        assert any(
            "결과 3건 중 상세페이지 1건(상세 아님 2건 제외)" in r.message for r in caplog.records
        )

    @respx.mock
    async def test_extract_stage_logs_what_it_bought_with_the_wait(
        self, tavily_on, extract_on, caplog
    ):
        """10초를 기다리고 0건을 확정한 요청이 있었는데 그 사실이 로그에 없었습니다."""
        url = "https://www.coupang.com/vp/products/1"
        respx.post(TAVILY_URL).mock(return_value=tavily_response(result("쿠키 세트", url)))
        respx.post(EXTRACT_URL).mock(return_value=extract_response((url, "가격 정보 없음")))

        with caplog.at_level(logging.INFO, logger="app.services.product_search"):
            await TavilyProductSearch().search([("디저트", None)], 30000, 50000, limit=1)

        assert any("판매가 Extract 1건(묶음 1개) → 0건 확정" in r.message for r in caplog.records)


class TestBotBlockedHosts:
    """확정적으로 실패하는 호스트에 매번 요청하지 않습니다."""

    @respx.mock
    async def test_a_host_that_returned_403_is_not_asked_again(self):
        route = respx.get(
            "https://www.ssg.com/item/itemView.ssg?itemId=1000613272222"
        ).mock(return_value=httpx.Response(403))
        url = "https://www.ssg.com/item/itemView.ssg?itemId=1000613272222"

        async with httpx.AsyncClient() as client:
            assert await product_search_module.fetch_price_direct(url, client) is None
            assert await product_search_module.fetch_price_direct(url, client) is None

        assert len(route.calls) == 1

    @respx.mock
    async def test_a_healthy_host_is_still_asked(self):
        respx.get("https://www.ssg.com/item/itemView.ssg?itemId=1").mock(
            return_value=httpx.Response(403)
        )
        route = respx.get("https://www.kurly.com/goods/1").mock(
            return_value=httpx.Response(200, html=KURLY_HTML)
        )

        async with httpx.AsyncClient() as client:
            await product_search_module.fetch_price_direct(
                "https://www.ssg.com/item/itemView.ssg?itemId=1", client
            )
            price = await product_search_module.fetch_price_direct(
                "https://www.kurly.com/goods/1", client
            )

        assert price == 55000
        assert len(route.calls) == 1


class TestCandidateBudget:
    """검색 결과 수와 후보 상한은 함께 움직여야 합니다."""

    def test_candidate_limit_covers_one_full_search(self):
        """product_candidate_limit 이 tavily_max_results 보다 작으면 안 됩니다.

        _interleave 는 예산이 아니라 **검색 관련도 순서**로 자르고, 예산으로 고르는
        _select_by_price 는 그 뒤에 옵니다. 상한이 낮으면 가격을 이미 아는 상태에서도
        예산에 맞는 후보가 잘려 나갑니다. Search 는 결과 수와 무관하게 1회 1크레딧이라
        max_results 만 올리는 것은 크레딧만 그대로 두고 효과를 버리는 셈입니다.
        """
        assert settings.product_candidate_limit >= settings.tavily_max_results

    @respx.mock
    async def test_search_asks_for_the_configured_result_count(self, tavily_on):
        route = respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("수제쿠키 선물세트", "https://www.coupang.com/vp/products/1")
            )
        )
        await TavilyProductSearch().search([("디저트", None)], 9000, 14000, limit=3)

        body = json.loads(route.calls[0].request.content)
        assert body["max_results"] == settings.tavily_max_results

    @respx.mock
    async def test_in_budget_candidate_below_the_relevance_top_survives(
        self, tavily_on, extract_on, monkeypatch
    ):
        """실측에서 상세페이지 후보 8건 중 예산 안이 0건이었습니다.

        관련도 상위가 전부 예산 밖이어도, 그 아래에 있는 예산 안 상품이 후보 상한에
        잘려서는 안 됩니다. product_candidate_limit 이 8이면 이 테스트가 깨집니다.
        """
        monkeypatch.setattr(settings, "tavily_extract_limit", 12)
        urls = [f"https://www.coupang.com/vp/products/{i}" for i in range(12)]
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                *(result(f"수제쿠키 선물세트 {i}", url) for i, url in enumerate(urls))
            )
        )
        # 관련도 상위 9건은 예산 밖, 10번째만 예산 안입니다.
        respx.post(EXTRACT_URL).mock(
            return_value=extract_response(
                *(
                    (url, "판매가 12,000 원" if index == 9 else "판매가 35,000 원")
                    for index, url in enumerate(urls)
                )
            )
        )

        products = await TavilyProductSearch().search(
            [("디저트", None)], 9000, 14000, limit=3
        )

        assert [p.url for p in products] == [urls[9]]
        assert products[0].price == 12000
        assert products[0].price_verified


class TestOutOfSeason:
    """시기가 어긋난 행사 상품은 달력만으로 걸러 냅니다.

    모델에게 오늘 날짜를 쥐어 주는 대신 여기서 정합니다. 요청마다 입력 토큰이 늘지
    않고, 외부 호출 없이 검증할 수 있습니다.
    """

    def test_a_christmas_tree_in_august(self):
        assert out_of_season("크리스마스 트리 미니트리 풀세트 눈꽃", 8) == "크리스마스"

    def test_the_same_tree_in_december(self):
        assert out_of_season("크리스마스 트리 미니트리 풀세트 눈꽃", 12) is None

    def test_november_is_early_enough_to_prepare(self):
        assert out_of_season("크리스마스 선물세트", 11) is None

    def test_spaces_do_not_hide_the_word(self):
        assert out_of_season("크리스 마스 리스", 8) == "크리스마스"

    def test_ordinary_gifts_are_untouched(self):
        for title in (
            "헤어 트리트먼트 세트",  # "트리" 를 낱말로 쓰지 않는 이유입니다
            "설레는 첫 만남 꽃다발",  # "설" 도 마찬가지입니다
            "프리미엄 디저트 세트 35,000원",
            "스타벅스 e카드교환권",
        ):
            assert out_of_season(title, 8) is None, title

    def test_a_trailing_event_word_is_a_search_keyword_not_the_product(self):
        """실측(4차, 8월): 사철 식물인 천리향 분재가 꼬리 낱말 하나로 걸렸습니다.

        후보가 5~7건뿐인 흐름에서 정상 상품을 지우는 손해가 더 큽니다.
        """
        assert out_of_season("천리까지 향이 천리향 분재 화산석화분 크리스마스", 8) is None

    def test_a_leading_event_word_still_disqualifies(self):
        """앞에 선 낱말은 뒤따르는 명사를 수식합니다. 이건 진짜 행사 상품입니다."""
        assert out_of_season("크리스마스 트리 미니트리 풀세트", 8) == "크리스마스"

    def test_a_title_that_is_only_the_event_word_is_not_a_tail(self):
        """꼬리로 보려면 앞에 상품명이 남아 있어야 합니다."""
        assert out_of_season("크리스마스", 8) == "크리스마스"
        assert out_of_season("미니 크리스마스", 8) == "크리스마스"

    def test_other_events_have_their_own_months(self):
        assert out_of_season("밸런타인 초콜릿 세트", 8) == "밸런타인"
        assert out_of_season("밸런타인 초콜릿 세트", 2) is None
        assert out_of_season("추석 선물세트 한우", 8) is None  # 8월은 추석 준비 기간
        assert out_of_season("추석 선물세트 한우", 3) == "추석"

    @respx.mock
    async def test_search_drops_it_before_it_reaches_the_user(self, tavily_on, monkeypatch):
        monkeypatch.setattr(product_search_module, "_current_month", lambda: 8)
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                # "눈꽃" 이 꽃·식물 키워드에 걸려 키워드 판정도 통과시킵니다.
                result("크리스마스 트리 미니트리 눈꽃 벽트리 19,900원", "https://gift.kakao.com/product/1"),
                result("미니 꽃다발 19,900원", "https://gift.kakao.com/product/2"),
            )
        )
        products = await TavilyProductSearch().search([("꽃·식물", None)], 18000, 27000, limit=3)

        assert [p.title for p in products] == ["미니 꽃다발 19,900원"]


class TestExtractIsBoundedByBatches:
    """묶음이 하나뿐이면 타임아웃 한 번이 가격 전량을 잃게 만듭니다.

    실측 4회 중 3회가 URL 5개 한 묶음으로 6초를 쓰고 0건을 확정했습니다. 유일한 성공은
    URL 3개 묶음의 0.7초 3/3 확정이었습니다.
    """

    def test_a_single_timeout_cannot_lose_every_price(self):
        assert settings.tavily_extract_batch_size < settings.tavily_extract_limit

    def test_the_wait_leaves_headroom_over_the_observed_success(self):
        """관측된 성공은 0.7초입니다. 그 4배를 남기되 6초는 과했습니다."""
        assert 2.0 <= settings.tavily_extract_timeout_seconds <= 4.0

    @respx.mock
    async def test_a_slow_batch_does_not_take_the_others_down(self, tavily_on, extract_on, monkeypatch):
        monkeypatch.setattr(settings, "tavily_extract_limit", 5)
        urls = [f"https://gift.kakao.com/product/{i}" for i in range(5)]
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(*(result(f"미니 꽃다발 {i}", u) for i, u in enumerate(urls)))
        )

        # 첫 묶음만 시간을 다 쓰고 잘립니다. 나머지 묶음의 가격은 살아남아야 합니다.
        calls = {"n": 0}

        def respond(request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ReadTimeout("too slow")
            body = json.loads(request.content)
            return extract_response(*((u, "판매가 19,900 원") for u in body["urls"]))

        respx.post(EXTRACT_URL).mock(side_effect=respond)
        products = await TavilyProductSearch().search([("꽃·식물", None)], 18000, 27000, limit=5)

        assert calls["n"] == 2  # 5건이 [3, 2] 로 나뉩니다.
        assert sum(1 for p in products if p.price_verified) == 2


class TestMobileDetailUrlsAreNotThrownAway:
    """5차 실측: 원본 검색 결과의 71% 가 상세페이지 판정에서 사라졌습니다.

    "버린 주소" 로그가 원인을 지목했습니다. 버려진 것 중 다섯 갈래는 모바일 상품
    상세 URL 이었고, 네 사이트가 저마다 PC 와 다른 경로를 씁니다. 호스트는 이미
    ``m.`` 을 떼고 비교하므로 문제는 경로 모양뿐이었습니다.

    아래 두 목록은 같은 실행의 로그에서 그대로 가져온 것입니다. 위는 통과해야 할
    것, 아래는 계속 막혀야 할 것입니다.
    """

    # 5차 로그가 잘못 버린 모바일 상세 주소.
    MOBILE_DETAILS = (
        "https://m.gmarket.co.kr/vi/product/4242045095",
        "https://m.gmarket.co.kr/vi/product/2543677304",
        "https://m.gmarket.co.kr/vi/product/4605856739?utparam-url=%7B%22scene%22%3A%22search%22%7D",
        "http://m.11st.co.kr/products/m/9511417705?catalog_no=406354823&lowest_yn=N",
        "https://m.oliveyoung.co.kr/m/G.do?goodsNo=B000000228285",
        "https://m.coupang.com/vm/products/9476933319?itemId=28200247970&vendorItemId=95238489152",
    )

    # 같은 로그에서 **정당하게** 버린 주소. 검색 결과·기획전·가이드 문서입니다.
    NOT_DETAILS = (
        "https://gift.kakao.com/search/result?query=%EC%BB%A4%ED%94%BC&searchType=search_related_keyword_item",
        "https://search.11st.co.kr/pc/total-search?kwd=%EB%B0%80%EC%96%91",
        "https://m.ssg.com/search.ssg?target=mobile&query=%EB%93%9C%EB%A6%BD&inflow=6005",
        "https://department.ssg.com/search.ssg?query=%EC%B6%95%ED%95%98%ED%99%94%EB%B6%84",
        "https://emart.ssg.com/search.ssg?query=%EA%B3%BC%EC%9E%90",
        "https://shinsegaemall.ssg.com/search.ssg?query=%23%EB%93%9C%EB%A6%BD%EC%9A%A9",
        "https://m-emart.ssg.com/disp/bundle.ssg?dispCtgId=&shppcstId=0001506925",
        "https://shinsegaemall.ssg.com/disp/bundle.ssg?ctgId=6000178565&itemSiteNo=6004",
        "https://guide.coupang.com/gift-recommendation-for-holidays-by-different-price-range",
        "https://rpp.gmarket.co.kr?exhib=60931",
        "https://m.oliveyoung.co.kr/m/mtn/gift",
    )

    # 5차·4차에서 실제로 상품이 나온 주소. 모바일 패턴을 더하면서 잃으면 안 됩니다.
    PC_DETAILS = (
        "https://item.gmarket.co.kr/Item?goodscode=1486647141",
        "http://mitem.gmarket.co.kr/Item?goodscode=3677564481",
        "https://m.ssg.com/item/itemView.ssg?itemId=1000641794699",
        "https://www.ssg.com/item/itemView.ssg?itemId=1000641794699",
        "https://www.coupang.com/vp/products/8439167843",
        "https://www.kurly.com/goods/1000136554",
        "https://www.11st.co.kr/products/8359844739",
        "https://gift.kakao.com/product/10276896",
    )

    @pytest.mark.parametrize("url", MOBILE_DETAILS)
    def test_a_mobile_detail_page_is_a_detail_page(self, url):
        assert product_search_module._is_product_detail_url(url)

    @pytest.mark.parametrize("url", NOT_DETAILS)
    def test_search_and_exhibition_pages_stay_out(self, url):
        assert not product_search_module._is_product_detail_url(url)

    @pytest.mark.parametrize("url", PC_DETAILS)
    def test_the_urls_that_already_produced_products_still_pass(self, url):
        assert product_search_module._is_product_detail_url(url)

    @respx.mock
    async def test_a_mobile_only_result_now_reaches_the_user(self, tavily_on):
        """검색 결과가 모바일 주소뿐이면 예전에는 상품 0건이었습니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result(
                    "송월타월 고급수건 답례품",
                    "http://m.11st.co.kr/products/m/9511417705?catalog_no=406354823",
                    "12,000원",
                ),
            )
        )
        products = await TavilyProductSearch().search([("생활용품", None)], 9000, 14000)

        assert [p.url for p in products] == [
            "http://m.11st.co.kr/products/m/9511417705?catalog_no=406354823"
        ]

    @pytest.mark.parametrize(
        ("mobile", "desktop"),
        [
            (
                "https://m.gmarket.co.kr/vi/product/1486647141?utparam-url=%7B%7D",
                "https://item.gmarket.co.kr/Item?goodscode=1486647141",
            ),
            (
                "http://m.11st.co.kr/products/m/9511417705?catalog_no=1",
                "https://www.11st.co.kr/products/9511417705",
            ),
            (
                "https://m.coupang.com/vm/products/9476933319?itemId=1",
                "https://www.coupang.com/vp/products/9476933319",
            ),
            (
                "https://m.oliveyoung.co.kr/m/G.do?goodsNo=B000000228285",
                "https://www.oliveyoung.co.kr/store/goods/getGoodsDetail.do?goodsNo=B000000228285",
            ),
        ],
    )
    def test_the_two_addresses_of_one_product_share_a_key(self, mobile, desktop):
        """모바일 주소를 받아들이면 같은 상품이 두 번 나갈 수 있습니다."""
        assert product_search_module._canonical_product_key(
            mobile
        ) == product_search_module._canonical_product_key(desktop)

    def test_two_different_gmarket_items_keep_their_own_keys(self):
        assert product_search_module._canonical_product_key(
            "https://m.gmarket.co.kr/vi/product/1"
        ) != product_search_module._canonical_product_key(
            "https://item.gmarket.co.kr/Item?goodscode=2"
        )

    def test_the_judged_candidates_stay_inside_the_search_budget(self):
        """모바일을 받아들이면 판정 입력과 직접 조회 GET 이 늘어납니다.

        product_candidate_limit 은 _interleave 에만 걸리고 그 단계는 판정·가격
        확인 **뒤**입니다. 판정에 들어가는 후보 수의 실제 상한은 검색 1회당
        tavily_max_results 이고, 이 함수는 그중 일부를 **빼기만** 합니다. 즉 이
        수정으로 늘어날 수 있는 최대치는 이미 예산에 잡혀 있는 값입니다.
        크레딧이 드는 Extract 만 따로 상한이 있습니다.
        """
        assert settings.tavily_extract_limit <= settings.tavily_max_results
        assert settings.product_candidate_limit >= settings.tavily_max_results


class TestSnippetNeverReachesTheResponse:
    """4·5차 실측에서 화면까지 나간 스니펫 셋 중 하나가 결제 화면 부스러기였습니다.

    필드는 스펙에 남겨 둡니다. 백엔드가 이 스펙으로 Java 클라이언트를 만들기
    때문에(``scripts/export_openapi.py``) 속성을 지우면 게터가 사라집니다.
    """

    # 5차 giftdata 응답 products[1] 에 그대로 실려 나간 판매자 홍보 문구입니다.
    MEASURED_MARKETING = (
        "과일의 풍미를 담아 스페셜티 드립백을 만드는 푸룻티커피를 만나 보세요. "
        "입에 닿는 순간 진한 과실향을 느낄 수 있는 티 블렌딩 드립백을 다채롭게 "
        "고루 담아 선물세트로 준비했어요."
    )
    # 4차 recommend 응답에 나간 결제 화면 부스러기. 47자로 위보다 **짧습니다**.
    MEASURED_CHECKOUT_JUNK = "감성 엽서 증정 무료배송. 장바구니 담기 사용안함,위시, 담은 수4.5만, 스위치."

    @pytest.mark.parametrize("value", [MEASURED_MARKETING, MEASURED_CHECKOUT_JUNK])
    def test_a_measured_snippet_is_not_serialised(self, value):
        product = ProductSuggestion(
            title="[선물세트] 푸룻티 커피 드립백 3종 세트 (21개입)",
            url="https://www.kurly.com/goods/1000136554",
            source="컬리",
            snippet=value,
        )
        assert product.model_dump()["snippet"] is None
        assert json.loads(product.model_dump_json())["snippet"] is None
        # 라우터가 실제로 쓰는 경로. exclude_none 이 걸려 키까지 사라집니다.
        assert "snippet" not in jsonable_encoder(product, exclude_none=True)

    def test_length_would_not_have_told_them_apart(self):
        """길이 제한으로 막자는 안을 배제한 근거입니다."""
        assert len(self.MEASURED_CHECKOUT_JUNK) < len(self.MEASURED_MARKETING)

    def test_the_value_still_exists_for_diagnosis(self):
        """응답에만 안 실릴 뿐, 값은 그대로 살아 있어야 로그로 원인을 볼 수 있습니다."""
        product = ProductSuggestion(
            title="t", url="https://www.kurly.com/goods/1", source="컬리", snippet="설명"
        )
        assert product.snippet == "설명"

    def test_the_field_stays_in_the_generated_client_contract(self):
        """백엔드가 이 스펙으로 Java 클라이언트를 생성합니다. 속성이 사라지면 안 됩니다."""
        properties = ProductSuggestion.model_json_schema(mode="serialization")["properties"]
        assert "snippet" in properties

    @respx.mock
    async def test_a_real_search_result_ships_without_its_snippet(self, tavily_on):
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result(
                    "[선물세트] 푸룻티 커피 드립백 3종 세트 (21개입)",
                    "https://www.kurly.com/goods/1000136554",
                    f"{self.MEASURED_MARKETING} 35,000원",
                )
            )
        )
        products = await TavilyProductSearch().search([("디저트", None)], 30000, 40000, limit=1)

        assert products[0].model_dump()["snippet"] is None
