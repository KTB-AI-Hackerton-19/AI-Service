import asyncio

from app.schemas.recommendation import ProductRecommendation
from app.services.product_search import ProductSearchProvider, ProductSearchService


class StubProductSearchProvider(ProductSearchProvider):
    async def search(self, query, minimum_price, maximum_price, limit=3):
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
    async def search(self, query, minimum_price, maximum_price, limit=3):
        raise ValueError("search unavailable")


def test_product_search_returns_structured_real_product():
    service = ProductSearchService(StubProductSearchProvider())
    products = asyncio.run(service.search_safely("답례 간식", 800, 1400))
    assert products[0].name == "실제 테스트 상품"
    assert products[0].price == 1200
    assert products[0].product_url.startswith("https://")


def test_product_search_failure_falls_back_to_empty_products():
    service = ProductSearchService(FailedProductSearchProvider())
    assert asyncio.run(service.search_safely("답례 간식", 800, 1400)) == []
