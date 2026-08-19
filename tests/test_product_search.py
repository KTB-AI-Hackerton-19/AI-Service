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
    extract_title_price,
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
        assert extract_title_price("[10만원권] 상품권 교환권 99,000원") == 99000


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
                result("디저트 교환권 45,000원", "https://gift.kakao.com/product/2"),
                result("과일 디저트 세트 45,000원", "https://shopping.naver.com/products/3"),
            )
        )
        products = await TavilyProductSearch().search([("식품·디저트", None)], 40000, 60000, limit=3)

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
        products = await TavilyProductSearch().search([("식품·디저트", None)], 9000, 14000, limit=2)

        assert products[0].price == 12000
        assert len(products) == 1

    @respx.mock
    async def test_content_subdomains_are_excluded(self, tavily_on):
        """guide.coupang.com 은 기사이지 살 수 있는 물건이 아닙니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("가격대별 선물 세트 추천", "https://guide.coupang.com/gift-recommendation"),
                result("수제쿠키 세트 12,000원", "https://www.coupang.com/vp/products/9"),
            )
        )
        products = await TavilyProductSearch().search([("식품·디저트", None)], 9000, 14000, limit=2)

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
        products = await TavilyProductSearch().search([("식품·디저트", None)], 40000, 60000, limit=2)

        assert products[0].kind == "product"
        assert products[0].price == 45000
        assert len(products) == 1

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
            side_effect=[
                tavily_response(result("디저트 쿠키", same)),
                tavily_response(result("커피 드립백", same)),
            ]
        )
        products = await TavilyProductSearch().search(
            [("식품·디저트", None), ("커피·차", None)], 40000, 60000
        )
        assert len(products) == 1

    @respx.mock
    async def test_deduplicates_mobile_and_desktop_product_urls(self, tavily_on):
        """같은 상품의 m/www 주소가 달라도 하나만 노출합니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("디저트 쿠키", "https://m.coupang.com/vp/products/123?itemId=1"),
                result("디저트 쿠키", "https://www.coupang.com/vp/products/123?vendorItemId=2"),
            )
        )
        products = await TavilyProductSearch().search(
            [("식품·디저트", None)], 30000, 50000
        )
        assert len(products) == 1

    @respx.mock
    async def test_excludes_semantically_unrelated_detail_product(self, tavily_on):
        """상품권 추천에 화장품 상세페이지가 섞이지 않아야 합니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(
                result("랑콤 썸머 베이스 듀오", "https://m.ssg.com/item/dealItemView.ssg?itemId=1"),
                result("외식 상품권 3만원권", "https://gift.kakao.com/product/2"),
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
                result("정상 디저트", "https://www.coupang.com/vp/products/2"),
            )
        )
        products = await TavilyProductSearch().search([("식품·디저트", None)], 40000, 60000)
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
        products = await TavilyProductSearch().search([("식품·디저트", None)], 30000, 50000, limit=1)

        assert products[0].price == 39000
        assert products[0].price_verified is True

    @respx.mock
    async def test_unverified_price_is_kept_but_flagged(self, tavily_on, extract_on):
        url = "https://www.coupang.com/vp/products/1"
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(result("쿠키 세트 32,000원", url))
        )
        respx.post(EXTRACT_URL).mock(return_value=extract_response((url, "가격 정보 없음")))
        products = await TavilyProductSearch().search([("식품·디저트", None)], 30000, 50000, limit=1)

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
        products = await TavilyProductSearch().search([("식품·디저트", None)], 30000, 50000, limit=2)

        assert products[0].price == 40000
        assert len(products) == 1

    @respx.mock
    async def test_uses_only_nearest_verified_price_when_all_are_outside(self, tavily_on, extract_on):
        """예산 안 상품이 없으면 범위와 가장 가까운 상세상품 하나만 제공합니다."""
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
            [("식품·디저트", None)], 30000, 50000, limit=3
        )
        assert len(products) == 1
        assert products[0].price == 55000

    @respx.mock
    async def test_extract_timeout_keeps_products(self, tavily_on, extract_on):
        """가격을 못 얻어도 상품명과 링크는 그대로 쓸 수 있어야 합니다."""
        respx.post(TAVILY_URL).mock(
            return_value=tavily_response(result("쿠키", "https://www.coupang.com/vp/products/1"))
        )
        respx.post(EXTRACT_URL).mock(side_effect=httpx.ReadTimeout("too slow"))
        products = await TavilyProductSearch().search([("식품·디저트", None)], 30000, 50000, limit=1)

        assert len(products) == 1
        assert products[0].title == "쿠키"
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
        await TavilyProductSearch().search([("식품·디저트", None)], 30000, 50000, limit=4)

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
        products = await TavilyProductSearch().search([("식품·디저트", None)], 30000, 50000, limit=1)

        assert len(route.calls) == 0
        assert products == []


class TestProductReason:
    @respx.mock
    async def test_reason_explains_price_fit(self, tavily_on, extract_on):
        url = "https://www.coupang.com/vp/products/1"
        respx.post(TAVILY_URL).mock(return_value=tavily_response(result("쿠키 세트", url)))
        respx.post(EXTRACT_URL).mock(return_value=extract_response((url, "판매가 40,000 원")))
        products = await TavilyProductSearch().search([("식품·디저트", None)], 30000, 50000, limit=1)

        assert "제안 가격대 안" in products[0].reason
        assert "쿠팡" in products[0].reason

    @respx.mock
    async def test_reason_flags_over_budget(self, tavily_on, extract_on):
        url = "https://www.coupang.com/vp/products/1"
        respx.post(TAVILY_URL).mock(return_value=tavily_response(result("비싼 디저트 세트", url)))
        respx.post(EXTRACT_URL).mock(return_value=extract_response((url, "판매가 200,000 원")))
        products = await TavilyProductSearch().search([("식품·디저트", None)], 30000, 50000, limit=1)

        assert "높습니다" in products[0].reason
