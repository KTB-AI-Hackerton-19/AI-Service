import asyncio

from app.schemas.recommendation import ProductRecommendation
from app.services.product_search import (
    ProductSearchProvider,
    ProductSearchService,
    TavilyProductSearchProvider,
)


class StubProductSearchProvider(ProductSearchProvider):
    async def search(self, query, category, minimum_price, maximum_price, limit=3):
        return [
            ProductRecommendation(
                name="실제 테스트 상품",
                price=1200,
                product_url="https://shop.example.com/product/1",
                image_url="https://shop.example.com/product/1.jpg",
                source="TEST_SEARCH",
            )
        ]


class FailedProductSearchProvider(ProductSearchProvider):
    async def search(self, query, category, minimum_price, maximum_price, limit=3):
        raise ValueError("search unavailable")


def test_product_search_returns_structured_real_product():
    service = ProductSearchService(StubProductSearchProvider())
    products = asyncio.run(
        service.search_safely("답례 간식", "식품·디저트", 800, 1400)
    )
    assert products[0].name == "실제 테스트 상품"
    assert products[0].price == 1200
    assert products[0].product_url.startswith("https://")


def test_product_search_failure_falls_back_to_empty_products():
    service = ProductSearchService(FailedProductSearchProvider())
    assert (
        asyncio.run(
            service.search_safely("답례 간식", "식품·디저트", 800, 1400)
        )
        == []
    )


def test_irrelevant_product_is_rejected_even_when_price_matches():
    product = TavilyProductSearchProvider._to_product(
        {
            "title": "고기접착제 관련 혜택과 특가",
            "url": "https://www.coupang.com/vp/products/123",
            "content": "판매가 15,000원",
        },
        "패션·잡화",
        9_000,
        15_000,
    )
    assert product is None


def test_generic_category_title_is_not_treated_as_product():
    product = TavilyProductSearchProvider._to_product(
        {
            "title": "패션잡화",
            "url": "https://www.ssg.com/item/itemView.ssg?itemId=123",
            "content": "판매가 8,010원",
        },
        "패션·잡화",
        9_000,
        15_000,
    )
    assert product is None


def test_search_result_page_is_not_treated_as_product():
    product = TavilyProductSearchProvider._to_product(
        {
            "title": "카드지갑 검색 결과",
            "url": "https://www.coupang.com/np/search?q=card-wallet",
            "content": "판매가 12,000원",
        },
        "패션·잡화",
        9_000,
        15_000,
    )
    assert product is None


def test_unknown_domain_is_not_treated_as_product():
    product = TavilyProductSearchProvider._to_product(
        {
            "title": "프리미엄 쿠키 선물 세트",
            "url": "https://unknown-blog.example.com/product/123",
            "content": "판매가 12,000원",
        },
        "식품·디저트",
        9_000,
        15_000,
    )
    assert product is None


def test_nearest_relevant_product_is_kept_when_price_is_outside_range():
    product = TavilyProductSearchProvider._to_product(
        {
            "title": "데일리 카드지갑",
            "url": "https://www.coupang.com/vp/products/456",
            "content": "판매가 17,000원",
        },
        "패션·잡화",
        9_000,
        15_000,
    )
    assert product is not None
    assert product.price == 17_000
    assert product.price_match == "NEAREST"
    assert product.price_difference == 2_000


def test_in_range_price_is_preferred_when_page_contains_multiple_prices():
    product = TavilyProductSearchProvider._to_product(
        {
            "title": "프리미엄 쿠키 선물 세트",
            "url": "https://gift.kakao.com/product/789",
            "content": "정가 18,000원, 할인가 14,000원",
        },
        "식품·디저트",
        9_000,
        15_000,
    )
    assert product is not None
    assert product.price == 14_000
    assert product.price_match == "IN_RANGE"
    assert product.price_difference == 0
