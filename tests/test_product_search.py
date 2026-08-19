"""Tavily 상품 검색 테스트. 실제 네트워크에 나가지 않습니다."""

import httpx
import pytest
import respx

from app.core.config import settings
from app.schemas.recommendation import (
    CategoryRecommendation,
    SimpleGiftRecommendationResponse,
)
from app.services.product_search import (
    TavilyProductSearch,
    build_query,
    extract_price,
    product_search,
)
from app.services.tasks.recommendation import RecommendationPreparationService

TAVILY_URL = settings.tavily_url


def tavily_response(*items: dict) -> httpx.Response:
    return httpx.Response(200, json={"results": list(items), "response_time": 1.2})


def result(title: str, url: str, content: str = "") -> dict:
    return {"title": title, "url": url, "content": content, "score": 0.9}


@pytest.fixture
def tavily_on(monkeypatch):
    monkeypatch.setattr(settings, "tavily_enabled", True)
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-test-key")


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


class TestBuildQuery:
    def test_uses_product_example_over_category(self):
        """카테고리명만으로는 검색이 잘 되지 않습니다."""
        query = build_query("식품·디저트", "프리미엄 디저트 세트", 40000, 60000)
        assert "프리미엄 디저트 세트" in query
        assert "5만원대" in query  # 중앙값 50,000

    def test_falls_back_to_category(self):
        assert "식품·디저트" in build_query("식품·디저트", None, 40000, 60000)

    def test_price_hint_uses_midpoint_not_ceiling(self):
        """상한을 쓰면 4만~24만원 같은 넓은 범위에서 29만원짜리만 걸려 나옵니다."""
        assert "14만원대" in build_query("식품·디저트", None, 40000, 240000)

    def test_small_budget_uses_won(self):
        assert "7000원" in build_query("커피·차", None, 5000, 9000)


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
        assert await product_search.search([("식품·디저트", None)], 40000, 60000) == []


class TestSearch:
    @respx.mock
    async def test_restricts_to_trusted_domains(self, tavily_on):
        """블로그·카페의 광고성 글을 걸러 내려면 도메인 제한이 필수입니다."""
        route = respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("수제쿠키 프리미엄 선물세트 45,000원", "https://www.coupang.com/vp/products/1")
            )
        )
        await TavilyProductSearch().search([("식품·디저트", "디저트 세트")], 40000, 60000)

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
                result("선물 추천", "https://gift.kakao.com/product/2"),
                result("과일 세트", "https://mkt.shopping.naver.com/x/3"),
            )
        )
        products = await TavilyProductSearch().search([("식품·디저트", None)], 40000, 60000, limit=3)

        assert [p.source for p in products] == ["쿠팡", "카카오 선물하기", "네이버 쇼핑"]

    @respx.mock
    async def test_in_range_price_ranks_first(self, tavily_on):
        """9,000~14,000원을 권해 놓고 20,000원짜리를 맨 앞에 보여 주면 추천의 의미가 없습니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("비싼 세트 20,000원", "https://www.coupang.com/vp/products/1"),
                result("범위 안 세트 12,000원", "https://www.coupang.com/vp/products/2"),
            )
        )
        products = await TavilyProductSearch().search([("식품·디저트", None)], 9000, 14000, limit=2)

        assert products[0].price == 12000
        assert products[1].price == 20000

    @respx.mock
    async def test_content_subdomains_are_marked_as_listing(self, tavily_on):
        """guide.coupang.com 은 기사이지 살 수 있는 물건이 아닙니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("가격대별 선물 세트 추천", "https://guide.coupang.com/gift-recommendation"),
                result("수제쿠키 세트 12,000원", "https://www.coupang.com/vp/products/9"),
            )
        )
        products = await TavilyProductSearch().search([("식품·디저트", None)], 9000, 14000, limit=2)

        assert products[0].kind == "product"
        assert products[1].kind == "listing"

    @respx.mock
    async def test_product_pages_rank_above_listings(self, tavily_on):
        """카카오 선물하기는 개별 상품보다 검색 페이지가 주로 잡힙니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("고급 디저트 세트 | 검색", "https://gift.kakao.com/search?q=디저트"),
                result("수제쿠키 선물세트 45,000원", "https://www.coupang.com/vp/products/9"),
            )
        )
        products = await TavilyProductSearch().search([("식품·디저트", None)], 40000, 60000, limit=2)

        assert products[0].kind == "product"
        assert products[0].price == 45000
        assert products[1].kind == "listing"

    @respx.mock
    async def test_interleaves_categories(self, tavily_on):
        """한 카테고리가 결과를 독차지하면 추천이 단조로워집니다."""
        respx.post(TAVILY_URL).mock(
            side_effect=[
                tavily_response(
                    result("디저트 A", "https://www.coupang.com/vp/products/1"),
                    result("디저트 B", "https://www.coupang.com/vp/products/2"),
                ),
                tavily_response(result("드립백 C", "https://www.11st.co.kr/products/3")),
            ]
        )
        products = await TavilyProductSearch().search(
            [("식품·디저트", None), ("커피·차", None)], 40000, 60000, limit=3
        )

        assert [p.category for p in products] == ["식품·디저트", "커피·차", "식품·디저트"]

    @respx.mock
    async def test_deduplicates_urls(self, tavily_on):
        same = "https://www.coupang.com/vp/products/1"
        respx.post(TAVILY_URL).mock(
            side_effect=[tavily_response(result("A", same)), tavily_response(result("B", same))]
        )
        products = await TavilyProductSearch().search(
            [("식품·디저트", None), ("커피·차", None)], 40000, 60000
        )
        assert len(products) == 1

    @respx.mock
    async def test_http_error_returns_empty(self, tavily_on):
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(429, text="rate limited"))
        assert await TavilyProductSearch().search([("식품·디저트", None)], 40000, 60000) == []

    @respx.mock
    async def test_network_error_returns_empty(self, tavily_on):
        respx.post(TAVILY_URL).mock(side_effect=httpx.ConnectTimeout("timeout"))
        assert await TavilyProductSearch().search([("식품·디저트", None)], 40000, 60000) == []

    @respx.mock
    async def test_one_category_failing_does_not_kill_the_rest(self, tavily_on):
        respx.post(TAVILY_URL).mock(
            side_effect=[
                httpx.Response(500, text="boom"),
                tavily_response(result("드립백", "https://www.11st.co.kr/products/3")),
            ]
        )
        products = await TavilyProductSearch().search(
            [("식품·디저트", None), ("커피·차", None)], 40000, 60000
        )
        assert len(products) == 1
        assert products[0].category == "커피·차"

    @respx.mock
    async def test_skips_results_without_url_or_title(self, tavily_on):
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                {"title": "", "url": "https://www.coupang.com/1", "content": ""},
                {"title": "제목만", "url": "", "content": ""},
                result("정상", "https://www.coupang.com/vp/products/2"),
            )
        )
        products = await TavilyProductSearch().search([("식품·디저트", None)], 40000, 60000)
        assert [p.title for p in products] == ["정상"]


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
                    category="식품·디저트",
                    score=90,
                    reason="무난합니다",
                    product_examples=["프리미엄 디저트 세트"],
                )
            ],
            summary="요약",
            suggested_message="메시지" * 40,
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
        products = await RecommendationPreparationService._find_products(self._recommendation())

        assert len(products) == 1
        assert products[0].price == 35000
        assert products[0].source == "쿠팡"

    async def test_no_products_when_search_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "tavily_enabled", False)
        assert await RecommendationPreparationService._find_products(self._recommendation()) == []

    @respx.mock
    async def test_search_failure_leaves_recommendation_intact(self, tavily_on):
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(503))
        recommendation = self._recommendation()
        recommendation.products = await RecommendationPreparationService._find_products(recommendation)

        assert recommendation.products == []
        assert recommendation.categories[0].category == "식품·디저트"
        assert recommendation.summary == "요약"
