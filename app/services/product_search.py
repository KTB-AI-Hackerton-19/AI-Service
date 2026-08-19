"""Tavily 로 실제 구매 가능한 상품을 찾습니다.

왜 모델에게 검색 툴을 쥐어 주지 않는가
- 12B 급 모델이 툴을 부를지 말지 판단하게 하면 신뢰성이 떨어집니다.
- 호출 횟수가 정해지지 않아 지연을 예측할 수 없습니다. 추천은 네 갈래 병렬 중 하나라
  여기서 늘어난 시간이 전체 응답 시간이 됩니다.

그래서 모델은 카테고리와 가격 범위까지만 정하고, 검색은 파이프라인이 결정론적으로 부릅니다.
검색이 실패하거나 결과가 없어도 추천 자체는 그대로 나갑니다.

신뢰할 수 있는 국내 거래 플랫폼(``settings.product_search_domains``)으로만 제한합니다.
제한하지 않으면 블로그·카페의 광고성 글이 상위를 채웁니다.
"""

import asyncio
import logging
import re
from urllib.parse import urlparse

import httpx

from app.core.config import settings
from app.schemas.recommendation import ProductSuggestion

logger = logging.getLogger(__name__)

# "12,300원", "12300 원" 같은 표기에서 금액을 뽑습니다.
_PRICE_PATTERN = re.compile(r"([0-9][0-9,]{2,})\s*원")
# 개별 상품이 아니라 검색·목록 페이지인지 판단합니다.
# 카카오 선물하기는 상품 페이지보다 검색 페이지가 주로 잡힙니다.
_LISTING_HINTS = ("검색", "카테고리", "베스트", "랭킹", "기획전", "가이드", "추천 |")

# 거래 플랫폼 도메인이지만 상품이 아니라 콘텐츠를 서비스하는 서브도메인입니다.
# guide.coupang.com 은 "가격대별 선물 세트 추천" 같은 기사이고 살 수 있는 물건이 아닙니다.
_CONTENT_HOSTS = frozenset(
    {
        "guide.coupang.com",
        "news.coupang.com",
        "mkt.shopping.naver.com",
        "campaign.11st.co.kr",
        "event.gmarket.co.kr",
    }
)

_SOURCE_NAMES = {
    "coupang.com": "쿠팡",
    "gift.kakao.com": "카카오 선물하기",
    "shopping.naver.com": "네이버 쇼핑",
    "ssg.com": "SSG",
    "gmarket.co.kr": "G마켓",
    "11st.co.kr": "11번가",
    "lotteon.com": "롯데온",
    "kurly.com": "컬리",
    "oliveyoung.co.kr": "올리브영",
}


def _source_name(url: str) -> str:
    """URL 에서 사람이 읽는 플랫폼 이름을 만듭니다."""
    host = (urlparse(url).hostname or "").lower().removeprefix("www.").removeprefix("m.")
    for domain, name in _SOURCE_NAMES.items():
        if host == domain or host.endswith("." + domain):
            return name
    return host or "알 수 없음"


def extract_price(text: str, low: int, high: int) -> int | None:
    """본문에서 가격을 뽑되, 추천 범위와 동떨어진 숫자는 버립니다.

    검색 결과에는 배송비나 후기 수 같은 무관한 숫자도 "원" 과 함께 나옵니다.
    범위의 절반~두 배 안에 드는 값만 상품 가격으로 인정합니다.
    """
    floor, ceiling = max(1, low // 2), high * 2
    for raw in _PRICE_PATTERN.findall(text or ""):
        try:
            value = int(raw.replace(",", ""))
        except ValueError:
            continue
        if floor <= value <= ceiling:
            return value
    return None


# 선물 자체가 아니라 포장재만 파는 결과입니다. "생활용품" 같은 넓은 카테고리에서 걸려 나옵니다.
_PACKAGING_ONLY = ("쇼핑백", "포장지", "포장 박스", "리본끈", "선물박스", "택배박스")


def _is_packaging_only(title: str) -> bool:
    """포장재만 파는 상품인지. 선물로 추천할 물건이 아닙니다."""
    return any(word in title for word in _PACKAGING_ONLY)


def _is_listing(title: str, url: str) -> bool:
    """개별 상품 페이지가 아니라 검색·목록·기사 페이지인지."""
    host = (urlparse(url).hostname or "").lower()
    if host in _CONTENT_HOSTS:
        return True
    if any(hint in title for hint in _LISTING_HINTS):
        return True
    path = (urlparse(url).path or "").lower()
    return "/search" in path or path in ("", "/")


def build_query(category: str, example: str | None, low: int, high: int) -> str:
    """검색어를 만듭니다.

    카테고리명("식품·디저트")만으로는 검색이 잘 되지 않아, 모델이 낸 구체적인
    상품 유형("프리미엄 디저트 세트")을 앞세우고 가격대를 덧붙입니다.

    가격 힌트는 상한이 아니라 **범위 중앙값**을 씁니다. 상한을 쓰면 4만~24만원 같은
    넓은 범위에서 29만원짜리만 걸려 나옵니다.
    """
    seed = (example or category).strip()
    middle = (low + high) // 2
    price_hint = f"{middle // 10000}만원대" if middle >= 10_000 else f"{middle}원"
    return f"{seed} 선물 {price_hint}"


class TavilyProductSearch:
    """신뢰할 수 있는 국내 거래 플랫폼에서만 상품을 찾습니다."""

    @property
    def is_available(self) -> bool:
        """검색을 시도할 수 있는 상태인지."""
        return settings.tavily_enabled and bool(settings.tavily_api_key)

    async def search_one(
        self,
        category: str,
        example: str | None,
        low: int,
        high: int,
        client: httpx.AsyncClient,
    ) -> list[ProductSuggestion]:
        """카테고리 하나에 대해 검색합니다. 실패하면 빈 목록을 돌려줍니다."""
        query = build_query(category, example, low, high)
        body = {
            "query": query,
            "search_depth": settings.tavily_search_depth,
            "max_results": settings.tavily_max_results,
            "include_domains": list(settings.product_search_domains),
            "topic": "general",
            # country 파라미터는 include_domains 와 함께 쓰면 결과가 0건이 됩니다(실측).
        }

        try:
            response = await client.post(
                settings.tavily_url,
                json=body,
                headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            )
        except httpx.HTTPError as exc:
            logger.warning("상품 검색 실패 category=%s: %s", category, exc)
            return []

        if response.status_code != 200:
            logger.warning("상품 검색 HTTP %s category=%s", response.status_code, category)
            return []

        try:
            results = response.json().get("results") or []
        except ValueError:
            logger.warning("상품 검색 응답을 읽지 못했습니다 category=%s", category)
            return []

        suggestions = []
        for item in results:
            url = str(item.get("url") or "")
            title = str(item.get("title") or "").strip()
            if not url or not title:
                continue
            content = str(item.get("content") or "")
            if _is_packaging_only(title):
                continue
            is_listing = _is_listing(title, url)
            suggestions.append(
                ProductSuggestion(
                    title=title[:200],
                    url=url,
                    source=_source_name(url),
                    category=category,
                    # 검색·목록 페이지에서 읽은 숫자는 특정 상품의 가격이 아닙니다.
                    # 우연히 스니펫에 있던 값을 상품 가격처럼 보여 주면 안 됩니다.
                    price=None if is_listing else extract_price(f"{title} {content}", low, high),
                    kind="listing" if is_listing else "product",
                    snippet=content[:200] or None,
                )
            )
        return suggestions

    async def search(
        self,
        categories: list[tuple[str, str | None]],
        low: int,
        high: int,
        limit: int | None = None,
    ) -> list[ProductSuggestion]:
        """카테고리별로 동시에 검색하고 결과를 골라 돌려줍니다.

        Args:
            categories: (카테고리명, 대표 상품 유형) 목록. 모델이 낸 순서를 그대로 씁니다.
            low: 추천 가격 하한.
            high: 추천 가격 상한.
            limit: 최종 상품 수. 기본값은 설정값.

        Returns:
            카테고리를 골고루 섞은 상품 목록. 검색이 불가능하면 빈 목록.
        """
        if not self.is_available or not categories:
            return []

        limit = limit or settings.product_suggestion_limit
        async with httpx.AsyncClient(timeout=settings.tavily_timeout_seconds) as client:
            batches = await asyncio.gather(
                *(self.search_one(name, example, low, high, client) for name, example in categories),
                return_exceptions=True,
            )

        clean: list[list[ProductSuggestion]] = []
        for batch in batches:
            if isinstance(batch, BaseException):
                logger.warning("상품 검색 중 예외: %s", batch)
                continue
            clean.append(sorted(batch, key=lambda s: _rank(s, low, high)))

        return _interleave(clean, limit)


def _rank(item: ProductSuggestion, low: int, high: int) -> tuple:
    """검색 결과를 좋은 순으로 정렬하는 기준.

    우선순위는 셋입니다.
    1. 추천 가격 범위 안에 드는 상품. 9,000~14,000원을 권해 놓고 20,000원짜리를
       맨 앞에 보여 주면 추천의 의미가 없습니다.
    2. 검색·목록·기사 페이지가 아니라 개별 상품 페이지.
    3. 가격을 읽어 낸 것. 사용자가 클릭 전에 판단할 수 있습니다.
    """
    in_range = item.price is not None and low <= item.price <= high
    return (not in_range, item.kind != "product", item.price is None)


def _interleave(batches: list[list[ProductSuggestion]], limit: int) -> list[ProductSuggestion]:
    """카테고리별 결과를 한 개씩 번갈아 뽑습니다.

    한 카테고리가 결과를 독차지하면 추천이 단조로워집니다.
    """
    picked: list[ProductSuggestion] = []
    seen: set[str] = set()
    for round_index in range(max((len(b) for b in batches), default=0)):
        for batch in batches:
            if round_index >= len(batch):
                continue
            item = batch[round_index]
            if item.url in seen:
                continue
            seen.add(item.url)
            picked.append(item)
            if len(picked) >= limit:
                return picked
    return picked


product_search = TavilyProductSearch()
