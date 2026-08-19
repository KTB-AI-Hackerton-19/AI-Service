"""실제 상품 페이지를 찾는 외부 웹 검색 도구와 안전한 fallback을 제공합니다."""

import asyncio
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.core.config import settings
from app.schemas.recommendation import ProductRecommendation

PRICE_PATTERN = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})+|\d{3,8})\s*원")
# 블로그·뉴스를 실제 상품으로 오인하지 않도록 구매 가능한 국내 쇼핑 도메인만
# 검색합니다. 운영 중 제휴 판매처가 정해지면 이 목록을 해당 판매처로 좁힙니다.
SHOPPING_DOMAINS = [
    "gift.kakao.com",
    "coupang.com",
    "29cm.co.kr",
    "kurly.com",
    "ssg.com",
    "lotteon.com",
    "hmall.com",
]


class ProductSearchProvider(ABC):
    """검색 업체를 교체할 수 있도록 고정한 상품 검색 함수 시그니처."""

    @abstractmethod
    async def search(
        self,
        query: str,
        minimum_price: int,
        maximum_price: int,
        limit: int = 3,
    ) -> list[ProductRecommendation]:
        """검색어와 가격 범위를 받아 실제 상품 페이지를 반환합니다."""


class DisabledProductSearchProvider(ProductSearchProvider):
    """API 키가 없을 때 웹 호출 없이 빈 결과를 반환합니다."""

    async def search(
        self,
        query: str,
        minimum_price: int,
        maximum_price: int,
        limit: int = 3,
    ) -> list[ProductRecommendation]:
        return []


class TavilyProductSearchProvider(ProductSearchProvider):
    """Tavily Search API 결과를 실제 상품 후보로 변환합니다."""

    def __init__(self, api_key: str, timeout_seconds: float = 8.0) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    async def search(
        self,
        query: str,
        minimum_price: int,
        maximum_price: int,
        limit: int = 3,
    ) -> list[ProductRecommendation]:
        search_query = (
            f"{query} 실제 구매 상품 가격 {minimum_price}원~{maximum_price}원"
        )
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "query": search_query,
            "search_depth": "basic",
            "topic": "general",
            "country": "south korea",
            "max_results": min(max(limit * 3, 5), 10),
            "include_images": True,
            "include_answer": False,
            "include_domains": SHOPPING_DOMAINS,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        products: list[ProductRecommendation] = []
        seen_urls: set[str] = set()
        for item in response.json().get("results", []):
            product = self._to_product(item, minimum_price, maximum_price)
            if product is None or product.product_url in seen_urls:
                continue
            seen_urls.add(product.product_url)
            products.append(product)
            if len(products) >= limit:
                break
        return products

    @staticmethod
    def _to_product(
        item: dict[str, Any],
        minimum_price: int,
        maximum_price: int,
    ) -> ProductRecommendation | None:
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        if not title or not url.startswith(("http://", "https://")):
            return None

        searchable_text = f"{title} {item.get('content', '')}"
        price_match = PRICE_PATTERN.search(searchable_text)
        price = int(price_match.group(1).replace(",", "")) if price_match else None
        # 가격을 확인할 수 없는 문서·목록 글은 실제 상품 태그에서 제외합니다.
        if price is None or not minimum_price <= price <= maximum_price:
            return None

        images = item.get("images") or []
        image_url: str | None = None
        if images:
            first_image = images[0]
            image_url = (
                str(first_image.get("url"))
                if isinstance(first_image, dict)
                else str(first_image)
            )
        return ProductRecommendation(
            name=re.sub(r"<[^>]+>", "", title)[:300],
            price=price,
            product_url=url,
            image_url=image_url,
            source="TAVILY_WEB_SEARCH",
        )


class ProductSearchService:
    """카테고리별 검색을 병렬 실행하고 검색 장애를 추천 장애와 분리합니다."""

    def __init__(self, provider: ProductSearchProvider | None = None) -> None:
        self.provider = provider or self._provider_from_settings()

    @staticmethod
    def _provider_from_settings() -> ProductSearchProvider:
        provider = settings.product_search_provider.strip().lower()
        if provider in {"auto", "tavily"} and settings.tavily_api_key:
            return TavilyProductSearchProvider(settings.tavily_api_key)
        return DisabledProductSearchProvider()

    async def search_safely(
        self,
        query: str,
        minimum_price: int,
        maximum_price: int,
        limit: int = 3,
    ) -> list[ProductRecommendation]:
        """검색 실패·시간초과 시 전체 API 대신 빈 상품 목록으로 fallback합니다."""
        try:
            return await asyncio.wait_for(
                self.provider.search(query, minimum_price, maximum_price, limit),
                timeout=settings.product_search_timeout_seconds,
            )
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError):
            return []


product_search_service = ProductSearchService()
