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
from urllib.parse import parse_qs, urlparse, urlunparse

import httpx

from app.core.config import settings
from app.schemas.recommendation import ProductSuggestion

logger = logging.getLogger(__name__)

# "12,300원", "12300 원" 같은 표기에서 금액을 뽑습니다.
_PRICE_PATTERN = re.compile(r"([0-9][0-9,]{2,})\s*원")
# 상품 페이지 본문의 "판매가 39,000 원" 처럼 무엇의 값인지 분명한 표기입니다.
# 검색 스니펫의 숫자와 달리 이건 그 상품의 실제 판매가입니다.
_LABELED_PRICE_PATTERNS = (
    re.compile(r"판매\s*가[^0-9]{0,12}([0-9][0-9,]{2,})\s*원"),
    re.compile(r"할인\s*가[^0-9]{0,12}([0-9][0-9,]{2,})\s*원"),
    re.compile(r"(?<!원)가격\s*(?:정보)?[^0-9]{0,12}([0-9][0-9,]{2,})\s*원"),
)
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

# 카테고리와 검색 결과의 의미가 맞는지 결정론적으로 확인하는 핵심어입니다.
# LLM에게 재검증을 맡기면 검색마다 추론이 한 번 더 필요하고 결과도 흔들리므로,
# 상품 제목·검색 스니펫에 해당 카테고리의 명확한 단서가 있는지만 검사합니다.
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "상품권": ("상품권", "금액권", "교환권", "이용권", "쿠폰", "기프트카드"),
    "식품·디저트": ("식품", "디저트", "쿠키", "케이크", "과일", "초콜릿", "사탕", "마카롱", "베이커리"),
    "커피·차": ("커피", "원두", "드립백", "차", "티백", "텀블러"),
    "패션·잡화": ("패션", "지갑", "가방", "파우치", "액세서리", "잡화"),
    "생활용품": ("생활", "타월", "수건", "텀블러", "식기", "주방", "세제"),
    "꽃·식물": ("꽃", "꽃다발", "화분", "식물", "플라워"),
    "문화·취미": ("도서", "책", "문구", "공연", "전시", "취미", "티켓"),
    "디지털 액세서리": ("충전", "케이블", "거치대", "이어폰", "보조배터리", "디지털"),
    "뷰티": ("화장품", "뷰티", "립", "향수", "스킨", "로션", "메이크업"),
}


def _category_keywords(category: str) -> tuple[str, ...]:
    """표기 차이를 허용해 카테고리 검증용 핵심어를 찾습니다."""
    compact = re.sub(r"\s+", "", category)
    for name, keywords in _CATEGORY_KEYWORDS.items():
        if name in compact or compact in name:
            return keywords
    return ()


def _is_semantically_relevant(category: str, example: str | None, title: str, content: str) -> bool:
    """추천 카테고리·상품 유형과 검색 결과가 의미상 관련 있는지 확인합니다.

    알려진 카테고리는 핵심어가 하나도 없는 결과를 제외합니다. 상품 유형의 단어가
    제목에 있으면 카테고리 핵심어가 없어도 허용해 "구움과자 세트" 같은 구체적
    검색어가 사전에 없을 때의 과도한 누락을 피합니다.
    """
    title_text = re.sub(r"\s+", "", title).lower()
    haystack = re.sub(r"\s+", "", f"{title} {content}").lower()
    keywords = _category_keywords(category)
    if keywords:
        # 알려진 카테고리는 모델이 만든 product_example보다 카테고리 자체를 신뢰합니다.
        # 모델이 상품권 카테고리에 화장품명을 예시로 잘못 넣더라도 화장품 결과가
        # "예시명 일치"로 검증을 우회하면 안 됩니다.
        # Tavily 스니펫에는 검색어가 문맥 없이 반복될 수 있으므로 제목만 봅니다.
        return any(keyword.lower() in title_text for keyword in keywords)

    example_tokens = [
        token.lower()
        for token in re.findall(r"[가-힣A-Za-z0-9]{2,}", example or "")
        if token not in {"선물", "추천", "세트", "프리미엄"}
    ]
    if example_tokens and any(token in haystack for token in example_tokens):
        return True
    # 사전에 없는 카테고리는 구체적인 상품 유형이 있으면 그것으로 검증하고,
    # 유형마저 없으면 Tavily 관련도에 맡깁니다.
    return not example_tokens


def _is_product_detail_url(url: str) -> bool:
    """지원 쇼핑몰에서 바로 구매 가능한 개별 상품 상세 URL인지 확인합니다."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    path = parsed.path.lower()
    query = parse_qs(parsed.query.lower())

    if host.endswith("coupang.com"):
        return bool(re.search(r"/vp/products/\d+", path))
    if host == "gift.kakao.com":
        return bool(re.search(r"/product/\d+", path))
    if host.endswith("shopping.naver.com"):
        return bool(re.search(r"/(?:products|product)/\d+", path))
    if host.endswith("ssg.com"):
        return "itemview.ssg" in path and bool(query.get("itemid"))
    if host.endswith("gmarket.co.kr"):
        return "/item" in path and bool(query.get("goodscode"))
    if host.endswith("11st.co.kr"):
        return bool(re.search(r"/products/\d+", path))
    if host.endswith("lotteon.com"):
        return "/p/product/" in path
    if host.endswith("kurly.com"):
        return bool(re.search(r"/goods/\d+", path))
    if host.endswith("oliveyoung.co.kr"):
        return "getgoodsdetail.do" in path and bool(query.get("goodsno"))
    return False


def _canonical_product_key(url: str) -> str:
    """모바일/PC 주소와 추적 파라미터가 달라도 같은 상품이면 같은 키를 만듭니다."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    path = re.sub(r"/+", "/", parsed.path).rstrip("/").lower()
    query = parse_qs(parsed.query.lower())

    id_patterns = (
        (r"/vp/products/(\d+)", "coupang"),
        (r"/product/(\d+)", "kakao"),
        (r"/products/(\d+)", "product"),
        (r"/goods/(\d+)", "goods"),
    )
    for pattern, prefix in id_patterns:
        match = re.search(pattern, path)
        if match:
            return f"{host}:{prefix}:{match.group(1)}"
    for key in ("itemid", "goodscode", "goodsno"):
        if query.get(key):
            return f"{host}:{key}:{query[key][0]}"
    return urlunparse((parsed.scheme.lower() or "https", host, path, "", "", ""))


def _source_name(url: str) -> str:
    """URL 에서 사람이 읽는 플랫폼 이름을 만듭니다."""
    host = (urlparse(url).hostname or "").lower().removeprefix("www.").removeprefix("m.")
    for domain, name in _SOURCE_NAMES.items():
        if host == domain or host.endswith("." + domain):
            return name
    return host or "알 수 없음"


def labeled_price(text: str) -> int | None:
    """본문에서 "판매가 39,000 원" 처럼 이름표가 붙은 금액을 뽑습니다.

    검색 스니펫에는 같은 브랜드의 다른 옵션 가격이 함께 나열됩니다. 실측에서
    gift.kakao.com/product/2198213 의 실제 판매가는 39,000원이었는데 스니펫에는
    32,000 / 15,000 / 23,000 만 있고 39,000 은 없었습니다.
    그래서 상품 페이지 본문에서 이름표가 붙은 값만 신뢰합니다.
    """
    for pattern in _LABELED_PRICE_PATTERNS:
        match = pattern.search(text or "")
        if not match:
            continue
        try:
            value = int(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if value > 0:
            return value
    return None


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


def extract_title_price(title: str) -> int | None:
    """상품 제목에 직접 적힌 첫 원화 가격을 후보 가격으로 읽습니다.

    스니펫은 배송비·후기 수 등 잡음이 많아 범위 제한이 필요하지만, 상세상품 제목의
    ``99,000원`` 표기는 해당 옵션의 가격일 가능성이 높습니다. 최종 응답에서는
    Extract 확인 전까지 ``price_verified=False``로 남겨 신뢰 수준을 구분합니다.
    """
    match = _PRICE_PATTERN.search(title or "")
    if not match:
        return None
    try:
        value = int(match.group(1).replace(",", ""))
    except ValueError:
        return None
    return value if 0 < value <= 100_000_000 else None


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
            # 최종 추천에는 검색·목록·기사 URL을 넣지 않습니다. 사용자가 링크를 눌렀을 때
            # 바로 특정 상품의 가격과 구매 버튼이 보이는 상세페이지여야 합니다.
            if _is_listing(title, url) or not _is_product_detail_url(url):
                continue
            if not _is_semantically_relevant(category, example, title, content):
                logger.info("카테고리 불일치 상품 제외 category=%s title=%s", category, title)
                continue
            suggestions.append(
                ProductSuggestion(
                    title=title[:200],
                    url=url,
                    source=_source_name(url),
                    category=category,
                    # 검색·목록 페이지에서 읽은 숫자는 특정 상품의 가격이 아닙니다.
                    # 우연히 스니펫에 있던 값을 상품 가격처럼 보여 주면 안 됩니다.
                    price=extract_title_price(title) or extract_price(content, low, high),
                    kind="product",
                    snippet=content[:200] or None,
                )
            )
        return suggestions

    async def _extract_batch(
        self,
        products: list[ProductSuggestion],
        client: httpx.AsyncClient,
    ) -> None:
        """묶음 하나를 Extract 로 조회해 판매가를 채웁니다. 실패는 조용히 넘깁니다."""
        try:
            response = await asyncio.wait_for(
                client.post(
                    settings.tavily_extract_url,
                    json={
                        "urls": [p.url for p in products],
                        "extract_depth": settings.tavily_extract_depth,
                        "format": "text",
                    },
                    headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
                ),
                timeout=settings.tavily_extract_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "판매가 확인 묶음이 %.0f초를 넘겨 건너뜁니다(%d건).",
                settings.tavily_extract_timeout_seconds,
                len(products),
            )
            return
        except httpx.HTTPError as exc:
            logger.warning("판매가 확인 실패(%s)", exc)
            return

        if response.status_code != 200:
            logger.warning("판매가 확인 HTTP %s", response.status_code)
            return

        try:
            results = response.json().get("results") or []
        except ValueError:
            return

        by_url = {p.url: p for p in products}
        for item in results:
            product = by_url.get(str(item.get("url") or ""))
            if product is None:
                continue
            price = labeled_price(str(item.get("raw_content") or ""))
            if price is not None:
                product.price = price
                product.price_verified = True

    async def enrich_prices(
        self,
        products: list[ProductSuggestion],
        client: httpx.AsyncClient,
    ) -> list[ProductSuggestion]:
        """Extract API 로 각 상품의 실제 판매가를 채웁니다.

        검색 스니펫의 숫자는 같은 브랜드의 다른 옵션 가격일 수 있어 믿을 수 없습니다.
        상품 페이지 본문의 "판매가 N원" 만 그 상품의 가격입니다.

        한 번에 몰아 보내지 않고 작은 묶음으로 나눠 동시에 부릅니다.
        접근이 막힌 URL 하나가 재시도로 시간을 끌면 한 묶음에 전부 넣었을 때
        나머지 결과까지 함께 잃기 때문입니다(실측에서 8건 한 묶음이 12초를 넘겼습니다).
        """
        targets = [p for p in products if p.kind == "product"][: settings.tavily_extract_limit]
        if not targets:
            return products

        size = max(1, settings.tavily_extract_batch_size)
        batches = [targets[i : i + size] for i in range(0, len(targets), size)]
        await asyncio.gather(
            *(self._extract_batch(batch, client) for batch in batches),
            return_exceptions=True,
        )

        unverified = [p for p in targets if not p.price_verified]
        if unverified:
            # 지우지 않고 표시만 남깁니다. 화면에서 "약 32,000원(확인 필요)" 처럼
            # 보여 줄 수 있고, 순위에서는 확인된 가격보다 뒤로 갑니다.
            logger.info("판매가 미확인 %d건. 스니펫 값은 참고용으로만 씁니다.", len(unverified))
        return products

    async def search(
        self,
        categories: list[tuple[str, str | None]],
        low: int,
        high: int,
        limit: int | None = None,
    ) -> list[ProductSuggestion]:
        """카테고리별로 검색하고, 실제 판매가를 확인한 뒤 골라 돌려줍니다.

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
                clean.append(batch)

            # 가격을 확정한 뒤에 골라야 예산에 맞는 상품이 뽑힙니다.
            # 확정 대상은 최종 개수보다 넉넉히 잡습니다.
            candidates = _interleave(
                [sorted(b, key=lambda s: (s.kind != "product",)) for b in clean],
                settings.product_candidate_limit,
            )
            candidates = await self.enrich_prices(candidates, client)

        ranked = _select_by_price(candidates, low, high, limit)
        for item in ranked:
            item.reason = _reason(item, low, high)
        return ranked


def _reason(item: ProductSuggestion, low: int, high: int) -> str:
    """이 상품을 고른 이유를 한 문장으로. 화면에 그대로 보여 줄 수 있습니다."""
    parts = [f"{item.category} 추천에 맞는 {item.source} 상품"] if item.category else [f"{item.source} 상품"]
    if item.price is None:
        parts.append("가격은 링크에서 확인이 필요합니다")
    elif not item.price_verified:
        parts.append(f"검색 기준 약 {item.price:,}원(확인 필요)")
    elif low <= item.price <= high:
        parts.append(f"판매가 {item.price:,}원으로 제안 가격대 안입니다")
    else:
        gap = "높습니다" if item.price > high else "낮습니다"
        parts.append(f"판매가 {item.price:,}원으로 제안 가격대보다 {gap}")
    if item.kind == "listing":
        parts.append("개별 상품이 아니라 검색 결과 페이지입니다")
    return ". ".join(parts)[:200]


def _rank(item: ProductSuggestion, low: int, high: int) -> tuple:
    """검색 결과를 좋은 순으로 정렬하는 기준.

    우선순위는 이렇습니다.
    1. 상품 페이지에서 확인한 판매가가 추천 범위 안에 드는 것.
       9,000~14,000원을 권해 놓고 20,000원짜리를 맨 앞에 보여 주면 추천의 의미가 없습니다.
    2. 판매가를 확인한 것. 스니펫의 숫자는 같은 브랜드 다른 옵션의 가격일 수 있습니다.
    3. 미확인이라도 범위 안으로 보이는 것.
    4. 검색·목록·기사 페이지가 아니라 개별 상품 페이지.
    5. 가격을 아는 것.
    """
    in_range = item.price is not None and low <= item.price <= high
    return (
        not (item.price_verified and in_range),
        not item.price_verified,
        not in_range,
        item.kind != "product",
        item.price is None,
    )


def _select_by_price(
    candidates: list[ProductSuggestion], low: int, high: int, limit: int
) -> list[ProductSuggestion]:
    """예산 안의 상세상품을 우선하고, 없으면 가장 가까운 가격 하나만 선택합니다.

    검증된 범위 내 상품이 있으면 범위 밖 상품으로 자리를 억지로 채우지 않습니다.
    범위 안 상품이 전혀 없을 때만 검증된 판매가가 가장 가까운 상세상품 하나를
    대안으로 제공합니다. 판매가를 확인하지 못한 경우에는 상세페이지 결과만 최대
    ``limit``개 유지하고 화면 경고로 가격 확인이 필요함을 알립니다.
    """
    detail_products = [item for item in candidates if item.kind == "product"]
    verified_in_range = [
        item
        for item in detail_products
        if item.price_verified and item.price is not None and low <= item.price <= high
    ]
    if verified_in_range:
        return sorted(verified_in_range, key=lambda item: _rank(item, low, high))[:limit]

    # Extract가 막혀도 검색 결과에 읽힌 가격이 범위 안이면 범위 밖 상품보다 낫습니다.
    # 단, ``price_verified=False``가 유지되므로 화면에는 확인 필요 표시가 붙습니다.
    apparent_in_range = [
        item for item in detail_products if item.price is not None and low <= item.price <= high
    ]
    if apparent_in_range:
        return sorted(apparent_in_range, key=lambda item: _rank(item, low, high))[:limit]

    verified = [item for item in detail_products if item.price_verified and item.price is not None]
    if verified:
        def distance(item: ProductSuggestion) -> int:
            assert item.price is not None
            if item.price < low:
                return low - item.price
            return item.price - high

        return [min(verified, key=lambda item: (distance(item), _rank(item, low, high)))]

    priced = [item for item in detail_products if item.price is not None]
    if priced:
        def apparent_distance(item: ProductSuggestion) -> int:
            assert item.price is not None
            if item.price < low:
                return low - item.price
            return item.price - high

        return [min(priced, key=lambda item: (apparent_distance(item), _rank(item, low, high)))]

    return sorted(detail_products, key=lambda item: _rank(item, low, high))[:limit]


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
            key = _canonical_product_key(item.url)
            if key in seen:
                continue
            seen.add(key)
            picked.append(item)
            if len(picked) >= limit:
                return picked
    return picked


product_search = TavilyProductSearch()
