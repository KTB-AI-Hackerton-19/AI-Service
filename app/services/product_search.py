"""실제 상품 페이지를 찾는 외부 웹 검색 도구와 안전한 fallback을 제공합니다."""

import asyncio
import re
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse

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
    "11st.co.kr",
    "gmarket.co.kr",
    "auction.co.kr",
    "smartstore.naver.com",
]

# 단순 검색어 일치만 사용하면 "패션 소품" 검색에 뜬 뜨개질 책처럼 의미가
# 다른 결과가 섞입니다. 상품 제목에 카테고리 핵심어가 실제로 있어야 합니다.
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "식품·디저트": (
        "디저트", "케이크", "쿠키", "과자", "사탕", "캔디", "초콜릿",
        "젤리", "마카롱", "베이커리", "과일", "간식", "떡", "캬라멜",
    ),
    "커피·차": ("커피", "원두", "드립백", "티백", "차", "티세트", "tea"),
    "생활용품": ("타월", "수건", "텀블러", "컵", "머그", "디퓨저", "캔들"),
    "패션·잡화": ("지갑", "가방", "파우치", "에코백", "키링", "액세서리", "의류"),
    "문화·취미": ("도서", "책", "문구", "노트", "다이어리", "공연", "전시"),
    "건강·웰니스": ("비타민", "영양제", "마사지", "스트레칭", "견과"),
    "꽃·식물": ("꽃", "꽃다발", "화분", "식물", "플라워"),
    "상품권": ("상품권", "교환권", "기프트카드", "이용권"),
    "디지털 액세서리": ("케이스", "충전", "케이블", "거치대", "이어폰", "보조배터리"),
    "유아·아동": ("유아", "아동", "어린이", "키즈", "장난감", "그림책", "놀이"),
}


class ProductSearchProvider(ABC):
    """검색 업체를 교체할 수 있도록 고정한 상품 검색 함수 시그니처."""

    @abstractmethod
    async def search(
        self,
        query: str,
        category: str,
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
        category: str,
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
        category: str,
        minimum_price: int,
        maximum_price: int,
        limit: int = 3,
    ) -> list[ProductRecommendation]:
        search_query = (
            f"{category} {query} 실제 구매 상품 가격 "
            f"{minimum_price}원~{maximum_price}원"
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
            product = self._to_product(
                item,
                category,
                minimum_price,
                maximum_price,
            )
            if product is None or product.product_url in seen_urls:
                continue
            seen_urls.add(product.product_url)
            products.append(product)

        # 범위 안 상품을 먼저, 부족한 자리는 경계값과 가격 차이가 가장 작은
        # 관련 상품으로 채웁니다.
        midpoint = (minimum_price + maximum_price) / 2
        products.sort(
            key=lambda product: (
                product.price_difference,
                abs(product.price - midpoint),
            )
        )
        return products[:limit]

    @staticmethod
    def _to_product(
        item: dict[str, Any],
        category: str,
        minimum_price: int,
        maximum_price: int,
    ) -> ProductRecommendation | None:
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        if (
            not title
            or not url.startswith(("http://", "https://"))
            or not TavilyProductSearchProvider._is_product_url(url)
            or not TavilyProductSearchProvider._is_semantically_relevant(
                title,
                category,
            )
        ):
            return None

        searchable_text = f"{title} {item.get('content', '')}"
        prices = [
            int(value.replace(",", ""))
            for value in PRICE_PATTERN.findall(searchable_text)
        ]
        if not prices:
            return None
        price = min(
            prices,
            key=lambda value: TavilyProductSearchProvider._price_distance(
                value,
                minimum_price,
                maximum_price,
            ),
        )
        price_difference = TavilyProductSearchProvider._price_distance(
            price,
            minimum_price,
            maximum_price,
        )

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
            price_match="IN_RANGE" if price_difference == 0 else "NEAREST",
            price_difference=price_difference,
            product_url=url,
            image_url=image_url,
            source="TAVILY_WEB_SEARCH",
        )

    @staticmethod
    def _is_semantically_relevant(title: str, category: str) -> bool:
        """상품 제목에 추천 카테고리의 구체적인 핵심어가 있는지 확인합니다."""
        normalized_title = re.sub(r"\s+", "", title).lower()
        normalized_category = category.replace("·", "").replace(" ", "").lower()
        # "패션잡화", "생활용품"처럼 카테고리명만 있는 목록 페이지는 상품이
        # 아니므로 관련 상품으로 인정하지 않습니다.
        if normalized_title.strip("-_|:/[]()") == normalized_category:
            return False
        keywords = CATEGORY_KEYWORDS.get(category, ())
        return bool(keywords) and any(
            keyword.lower().replace(" ", "") in normalized_title
            for keyword in keywords
        )

    @staticmethod
    def _is_product_url(url: str) -> bool:
        """검색 목록·홈페이지가 아닌 개별 상품으로 보이는 URL만 허용합니다."""
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if not any(
            hostname == domain or hostname.endswith(f".{domain}")
            for domain in SHOPPING_DOMAINS
        ):
            return False
        path = parsed.path.lower().rstrip("/")
        if not path or path in {"", "/"}:
            return False
        blocked_fragments = ("/search", "/np/search", "/category", "/categories")
        if any(fragment in path for fragment in blocked_fragments):
            return False
        if parsed.hostname and parsed.hostname.endswith("coupang.com"):
            return "/vp/products/" in f"{path}/"
        return True

    @staticmethod
    def _price_distance(price: int, minimum_price: int, maximum_price: int) -> int:
        """가격 범위 안이면 0, 밖이면 가장 가까운 경계까지의 차이를 반환합니다."""
        if price < minimum_price:
            return minimum_price - price
        if price > maximum_price:
            return price - maximum_price
        return 0


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
        category: str,
        minimum_price: int,
        maximum_price: int,
        limit: int = 3,
    ) -> list[ProductRecommendation]:
        """검색 실패·시간초과 시 전체 API 대신 빈 상품 목록으로 fallback합니다."""
        try:
            return await asyncio.wait_for(
                self.provider.search(
                    query,
                    category,
                    minimum_price,
                    maximum_price,
                    limit,
                ),
                timeout=settings.product_search_timeout_seconds,
            )
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError):
            return []


product_search_service = ProductSearchService()
