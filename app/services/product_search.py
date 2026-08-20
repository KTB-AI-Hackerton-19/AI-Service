"""Tavily 로 실제 구매 가능한 상품을 찾습니다.

왜 모델에게 검색 툴을 쥐어 주지 않는가
- 호출 횟수가 정해지지 않아 지연을 예측할 수 없습니다. 추천은 네 갈래 병렬 중 가장
  느린 갈래라, 여기서 늘어난 시간이 그대로 전체 응답 시간이 됩니다.
- 모델이 툴을 부를지 말지 판단하게 하면 그 판단 자체가 왕복 한 번입니다. Bedrock
  호출은 출력이 없어도 고정비가 약 1.2초입니다(``scripts/benchmark_split.py`` 실측).

이 판단은 모델 성능이 아니라 **지연 예측 가능성**에 근거합니다. 더 큰 모델로 바꿔도
근거가 사라지지 않습니다. 다만 상품 0건일 때 검색어를 한 번 고쳐 재검색하는 정도의
**상한이 정해진** 되풀이는 여기 논리와 어긋나지 않습니다.

그래서 모델은 카테고리와 가격 범위까지만 정하고, 검색은 파이프라인이 결정론적으로 부릅니다.
검색이 실패하거나 결과가 없어도 추천 자체는 그대로 나갑니다.

신뢰할 수 있는 국내 거래 플랫폼(``settings.product_search_domains``)으로만 제한합니다.
제한하지 않으면 블로그·카페의 광고성 글이 상위를 채웁니다.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date
from urllib.parse import parse_qs, urlparse, urlunparse

import httpx

from app.core.config import settings
from app.services import product_filter
from app.schemas.recommendation import ProductSuggestion

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchStats:
    """검색이 무엇을 보고 무엇을 걸렀는지. 호출 측이 응답 문장을 사실대로 쓰게 합니다.

    상품 0건은 이유가 둘입니다. 검색 자체가 비었을 수도 있고(``examined == 0``),
    후보는 있었지만 예산에 맞는 판매가를 확인하지 못했을 수도 있습니다. 같은
    "0건"이라도 사용자에게 할 말이 다른데, 반환값이 빈 목록 하나면 구분할 수
    없습니다. 실측 gift 응답이 정확히 그래서 "상품 검색 결과가 없어" 라고 말할
    참이었습니다 — 후보 9건을 찾아 놓고서.

    Attributes:
        examined: 가격 심사까지 간 상품 상세페이지 후보 수. 적합성 판정에서
            떨어진 것은 여기 들어오지 않습니다.
    """

    examined: int = 0


# "12,300원", "12300 원" 같은 표기에서 금액을 뽑습니다.
_PRICE_PATTERN = re.compile(r"([0-9][0-9,]{2,})\s*원")
# 상품 페이지 본문의 "판매가 39,000 원" 처럼 무엇의 값인지 분명한 표기입니다.
# 검색 스니펫의 숫자와 달리 이건 그 상품의 실제 판매가입니다.
_LABELED_PRICE_PATTERNS = (
    re.compile(r"판매\s*가[^0-9]{0,12}([0-9][0-9,]{2,})\s*원"),
    re.compile(r"할인\s*가[^0-9]{0,12}([0-9][0-9,]{2,})\s*원"),
    re.compile(r"(?<!원)가격\s*(?:정보)?[^0-9]{0,12}([0-9][0-9,]{2,})\s*원"),
)
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
# 1만원 이하 답례 선물은 상당수가 기프티콘·브랜드 상품인데, 제목에 "커피" 같은
# 일반명사가 없어 통째로 걸러졌습니다. 실측에서 "스타벅스 다크 로스트 아메리카노
# 30입", "스타벅스 e카드교환권"이 모두 탈락해 예산 내 후보가 0건이 됐습니다.
# 그래서 브랜드명과 실제 상품명 표기를 함께 넣습니다.
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "상품권": (
        "상품권", "금액권", "교환권", "이용권", "쿠폰", "기프트카드",
        "기프티콘", "e카드", "모바일교환권", "기프트콘",
    ),
    "식품·디저트": (
        "식품", "디저트", "쿠키", "케이크", "과일", "초콜릿", "사탕", "마카롱",
        "베이커리", "빵", "젤리", "아이스크림", "한과", "약과", "떡",
    ),
    "커피·차": (
        "커피", "원두", "드립백", "차", "티백", "텀블러",
        "아메리카노", "라떼", "콜드브루", "에스프레소", "카페", "스타벅스",
        "기프티콘", "음료",
    ),
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
    """지원 쇼핑몰에서 바로 구매 가능한 개별 상품 상세 URL인지 확인합니다.

    호스트가 아니라 **경로 모양**이 관문입니다. ``host`` 는 이미 ``m.`` 을 떼고
    비교하므로 모바일 주소도 같은 갈래로 들어오는데, 5차 실측에서 떨어진 것은
    전부 모바일 쪽이 PC 와 다른 경로를 쓰기 때문이었습니다. 원본 검색 결과의
    71% 가 이 함수에서 사라졌고 그중 아래 네 갈래가 명백한 상품 상세였습니다.

        m.gmarket.co.kr/vi/product/4242045095          PC 는 /Item?goodscode=
        m.11st.co.kr/products/m/9511417705             PC 는 /products/
        m.oliveyoung.co.kr/m/G.do?goodsNo=B0000002…    PC 는 getGoodsDetail.do
        m.coupang.com/vm/products/9476933319           PC 는 /vp/products/

    그래서 "m. 호스트면 통과" 같은 일반 규칙을 두지 않습니다. 네 사이트의 모바일
    경로가 서로 다른 데다, 같은 실측에서 ``m.ssg.com/search.ssg`` 와
    ``m.oliveyoung.co.kr/m/mtn/gift`` 는 **계속 막아야 할** 모바일 주소였습니다.
    호스트는 상세페이지 여부를 말해 주지 않으므로 사이트별 경로로만 판단합니다.

    실제 페이지를 받아 볼 수 없으므로 URL 구조가 스스로 증명하는 것만 넣습니다.
    위 네 갈래는 모두 경로에 상품 식별자가 그대로 박혀 있습니다.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    path = parsed.path.lower()
    query = parse_qs(parsed.query.lower())

    # /vp/ 는 PC, /vm/ 은 모바일. 뒤따르는 productId 는 같은 값입니다.
    if host.endswith("coupang.com"):
        return bool(re.search(r"/v[pm]/products/\d+", path))
    if host == "gift.kakao.com":
        return bool(re.search(r"/product/\d+", path))
    if host.endswith("shopping.naver.com"):
        return bool(re.search(r"/(?:products|product)/\d+", path))
    if host.endswith("ssg.com"):
        return "itemview.ssg" in path and bool(query.get("itemid"))
    # 모바일은 /vi/product/{아이템번호}, PC 는 /Item?goodscode={아이템번호}.
    # 기획전(rpp.gmarket.co.kr?exhib=)은 둘 중 어느 모양도 아니라 그대로 막힙니다.
    if host.endswith("gmarket.co.kr"):
        return bool(re.search(r"/vi/product/\d+", path)) or (
            "/item" in path and bool(query.get("goodscode"))
        )
    # 모바일은 상품번호 앞에 /m/ 이 하나 더 붙습니다(/products/m/9511417705).
    if host.endswith("11st.co.kr"):
        return bool(re.search(r"/products/(?:m/)?\d+", path))
    if host.endswith("lotteon.com"):
        return "/p/product/" in path
    if host.endswith("kurly.com"):
        return bool(re.search(r"/goods/\d+", path))
    # 모바일 상세는 getGoodsDetail.do 를 /m/G.do 로 줄여 씁니다. goodsNo 를 함께
    # 요구해야 기획전(/m/mtn/gift)이 같이 통과하지 않습니다.
    if host.endswith("oliveyoung.co.kr"):
        return ("getgoodsdetail.do" in path or path.endswith("/g.do")) and bool(
            query.get("goodsno")
        )
    return False


def _canonical_product_key(url: str) -> str:
    """모바일/PC 주소와 추적 파라미터가 달라도 같은 상품이면 같은 키를 만듭니다."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    path = re.sub(r"/+", "/", parsed.path).rstrip("/").lower()
    query = parse_qs(parsed.query.lower())

    # G마켓만 따로 봅니다. 같은 상품이 item. / mitem. / m. 세 호스트로 나와
    # 호스트를 키에 넣으면 절대 겹치지 않는데, 모바일 경로의 숫자가 PC 의
    # goodscode 와 같은 값이라는 것은 실측 URL 자체가 말해 줍니다.
    #   m.gmarket.co.kr/vi/product/4605856739?utparam-url={… "x_object_id":
    #   "4605856739", "x_object_type":"item" …}
    # 아이템 번호는 G마켓 안에서 유일하므로 호스트를 떼고 하나로 모읍니다.
    if host.endswith("gmarket.co.kr"):
        match = re.search(r"/vi/product/(\d+)", path)
        code = match.group(1) if match else (query.get("goodscode") or [""])[0]
        if code:
            return f"gmarket.co.kr:goodscode:{code}"

    id_patterns = (
        # /vp/(PC) 와 /vm/(모바일) 은 같은 productId 를 가리킵니다.
        (r"/v[pm]/products/(\d+)", "coupang"),
        (r"/product/(\d+)", "kakao"),
        # 11번가 모바일은 /products/m/{상품번호} 라 /m/ 을 건너뛰고 번호를 봅니다.
        (r"/products/(?:m/)?(\d+)", "product"),
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


# 금액 앞에 이런 말이 붙어 있으면 그 숫자는 상품 가격이 아닙니다.
# 실측: 컬리 드립백 세트(실제 55,000원) 페이지에서 "단위 당 가격 : 100g 당 11,000원"의
# 11,000 을 상품가로 읽어, 8,000~12,000원 예산에 맞는 상품이라고 사용자에게 보여줬습니다.
_NON_PRICE_CONTEXT = re.compile(
    r"(단위\s*당|당\s*가격|[0-9]\s*(?:g|kg|ml|L|개|매|입)\s*당"
    r"|배송비|배송료|왕복|반품|교환|쿠폰|적립|포인트|최소\s*주문)"
)
# 금액 바로 앞 이만큼만 봅니다. 더 넓히면 무관한 문장까지 걸려 정상 가격을 버립니다.
_CONTEXT_WINDOW = 40


def _in_non_price_context(text: str, start: int) -> bool:
    """이 위치의 금액이 상품 가격이 아닌 문맥에 있는지."""
    return bool(_NON_PRICE_CONTEXT.search(text[max(0, start - _CONTEXT_WINDOW) : start]))


# 봇 차단으로 403 을 돌려준 호스트입니다. 실측에서 www.ssg.com 이 세 요청 모두
# 403 이었습니다. 결과가 정해진 요청을 매번 보낼 이유가 없어 프로세스 단위로 기억합니다.
# 요청들이 동시에 나가므로 지연 이득은 크지 않지만, 커넥션과 로그 잡음이 줄어듭니다.
# 일시적 403 이면 프로세스가 살아 있는 동안 직접 조회를 못 하는데, 그때도 Extract 가
# 가격을 채우므로 추천이 깨지지는 않습니다.
_BOT_BLOCKED_HOSTS: set[str] = set()

# 상품 페이지를 직접 받아 가격을 읽을 때 쓰는 헤더입니다. 봇으로 보이면 403 이 납니다.
_DIRECT_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
}
_JSONLD_BLOCK = re.compile(
    r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>', re.S | re.I
)
# 표준 JSON-LD 가 없는 쇼핑몰의 자체 표기입니다. 사이트가 바뀌면 여기만 고치면 됩니다.
# 실측 커버리지: 9개 도메인 중 11번가(JSON-LD)와 컬리(salesPrice) 2곳입니다.
# 쿠팡·SSG·G마켓은 403 으로 막히고, 카카오·롯데온은 가격 마커가 없습니다.
_SITE_PRICE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("kurly.com", re.compile(r'"salesPrice"\s*:\s*"?(\d+)"?')),
)


def _jsonld_product_price(html: str) -> int | None:
    """schema.org Product 의 offers.price 를 읽습니다.

    임의의 ``finalPrice`` 같은 키는 쓰지 않습니다. 실측에서 11번가는 그런 키가
    41,900 이고 JSON-LD 는 25,900 이라 값이 갈렸습니다. 무엇의 가격인지 규격이
    보장하는 값만 신뢰합니다.
    """
    for match in _JSONLD_BLOCK.finditer(html):
        try:
            data = json.loads(match.group(1).strip())
        except (ValueError, TypeError):
            continue
        for node in data if isinstance(data, list) else [data]:
            if not isinstance(node, dict) or "Product" not in str(node.get("@type", "")):
                continue
            offers = node.get("offers") or {}
            for offer in offers if isinstance(offers, list) else [offers]:
                if not isinstance(offer, dict):
                    continue
                try:
                    value = int(float(offer["price"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if value > 0:
                    return value
    return None


async def fetch_price_direct(url: str, client: httpx.AsyncClient) -> int | None:
    """상품 페이지를 직접 받아 구조화된 판매가를 읽습니다.

    Tavily Extract 는 페이지를 마크다운으로 바꾸면서 HTML 안의 가격 데이터를
    버립니다. 실측에서 컬리 드립백 세트(실제 55,000원)는 Extract 본문에 단가
    11,000원과 배송비만 남아 가격을 확인할 수 없었지만, 원본 HTML 에는
    ``salesPrice: 55000`` 이 그대로 있었습니다.

    허용 도메인이 아니면 요청하지 않습니다. 검색 결과만 넘어오므로 이미 화이트
    리스트지만, 이 함수만 보고도 안전하도록 여기서 한 번 더 확인합니다.
    """
    host = (urlparse(url).hostname or "").lower()
    if not any(host == d or host.endswith("." + d) for d in settings.product_search_domains):
        return None
    if host in _BOT_BLOCKED_HOSTS:
        return None
    try:
        response = await client.get(
            url,
            headers=_DIRECT_FETCH_HEADERS,
            follow_redirects=True,
            timeout=settings.product_price_fetch_timeout_seconds,
        )
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        # 쿠팡·SSG·G마켓은 봇 차단으로 403 입니다. Extract 로 넘어갑니다.
        if response.status_code in (401, 403) and host not in _BOT_BLOCKED_HOSTS:
            _BOT_BLOCKED_HOSTS.add(host)
            logger.info("직접 조회가 %s 에서 %s. 이후 이 호스트는 Extract 로만 확인합니다.", host, response.status_code)
        return None

    html = response.text
    price = _jsonld_product_price(html)
    if price:
        return price
    for domain, pattern in _SITE_PRICE_PATTERNS:
        if host == domain or host.endswith("." + domain):
            match = pattern.search(html)
            if match:
                try:
                    value = int(match.group(1))
                except ValueError:
                    return None
                return value if value > 0 else None
    return None


def labeled_price(text: str) -> int | None:
    """본문에서 "판매가 39,000 원" 처럼 이름표가 붙은 금액을 뽑습니다.

    검색 스니펫에는 같은 브랜드의 다른 옵션 가격이 함께 나열됩니다. 실측에서
    gift.kakao.com/product/2198213 의 실제 판매가는 39,000원이었는데 스니펫에는
    32,000 / 15,000 / 23,000 만 있고 39,000 은 없었습니다.
    그래서 상품 페이지 본문에서 이름표가 붙은 값만 신뢰합니다.
    """
    body = text or ""
    for pattern in _LABELED_PRICE_PATTERNS:
        # 첫 매칭만 보면 "판매가 불가하여..." 같은 안내문에 걸려 진짜 가격을 놓칩니다.
        for match in pattern.finditer(body):
            if _in_non_price_context(body, match.start()):
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
    범위의 절반~두 배 안에 드는 값만 상품 가격으로 인정하고, 단가·배송비처럼
    상품 가격이 아닌 문맥에 있는 숫자는 범위 안에 들어도 버립니다.
    """
    body = text or ""
    floor, ceiling = max(1, low // 2), high * 2
    for match in _PRICE_PATTERN.finditer(body):
        if _in_non_price_context(body, match.start()):
            continue
        try:
            value = int(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if floor <= value <= ceiling:
            return value
    return None


def extract_title_price(title: str, low: int, high: int) -> int | None:
    """상품 제목에 적힌 원화 가격을 후보 가격으로 읽습니다.

    상세상품 제목의 ``99,000원`` 표기는 해당 옵션의 가격일 가능성이 높아 스니펫보다
    믿을 만하지만, 제목에도 상품가가 아닌 금액이 섞입니다. "10,000원 이상 구매 시
    무료배송" 같은 문구를 그대로 읽으면 그 값이 상품 가격이 되고, 검증된 가격이
    하나도 없을 때 "검색 기준 약 10,000원(확인 필요)"으로 사용자에게 노출됩니다.
    그래서 제안 가격대의 절반~두 배라는 상식 범위를 벗어난 값은 버리고 다음 금액을
    봅니다. 최종 응답에서는 Extract 확인 전까지 ``price_verified=False``로 남겨
    신뢰 수준을 구분합니다.
    """
    floor, ceiling = max(1, low // 2), high * 2
    for match in _PRICE_PATTERN.finditer(title or ""):
        try:
            value = int(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if floor <= value <= ceiling:
            return value
    return None


# 스니펫에 섞여 들어오는 상품 설명이 아닌 문장입니다. 실측(11번가 드라이플라워)에서
# "고객이 판매자의 서비스를 평가한 리뷰 중, 4~5점의 긍정 평가의 비율 (최근 1년 기준)."
# 같은 안내문이 상품 설명 자리에 그대로 노출됐습니다.
# "교환"·"쿠폰"은 넣지 않습니다. 상품권·기프티콘 카테고리에서는 그게 상품 설명입니다.
_SNIPPET_NOISE = (
    "고객이 판매자",
    "긍정 평가",
    "상품 Q&A",
    "24시간 내 응답",
    "카테고리 정보",
    "판매자 정보",
    "사업자등록",
    "반품",
    "배송비",
    "무이자",
    "리뷰 중",
    # "포인트 적립" 이었는데 실측 2차가 낱말 순서만 바꿔 빠져나갔습니다. 11번가
    # 상품 하나의 설명 자리에 "최대 적립 포인트 안내 11pay 신한은행 계좌이체 결제 시
    # 구매적립 포인트 2%…" 네 문장이 통째로 나갔습니다. 적립은 상거래 안내에만 쓰는
    # 말이라 선물 상품 설명에서 잃을 것이 없습니다.
    "적립",
)
# Extract/스크래핑이 남긴 마크다운 잔재입니다. "Title:" 머리글, "##" 제목, "*" 목록.
_SNIPPET_TITLE_LINE = re.compile(r"^\s*(?:title|url source|published time)\s*:.*$", re.I | re.M)
# 마크업 기호가 나오는 지점부터는 상품 설명이 아니라 페이지 구조물(제목·목록·표)입니다.
# 실측에서 "## 상품 카테고리 정보", "* <제목 반복>" 이 그렇게 들어왔고, 한 줄 안에
# 섞여 있어 줄 단위로는 못 걸렀습니다. 뒤쪽에 설명이 더 있어도 버립니다. 구조물을
# 상품 설명이라고 보여 주느니 짧게 끝내는 편이 낫습니다.
_SNIPPET_MARKUP = re.compile(r"[#*>`|\[\]]")
_SNIPPET_SPLIT = re.compile(r"(?<=[.!?。])\s+|[\n·]+")
_SNIPPET_MAX = 200
# 상품 정보 표가 한 줄로 이어 붙은 조각입니다. 실측 3차 ", 상품명 :국내생산타월의품격",
# 2차 ", 판매가 :49,800 원 무료배송 장바구니 담기 ..." 가 같은 계열입니다. 설명문이
# 아니라 표의 칸이라 그대로 화면에 나가면 안 됩니다.
#
# 낱말 뒤에 공백을 두고 콜론이 오는 형태("상품명 :")를 표식으로 씁니다. 한국어
# 설명문에는 사실상 없는 모양이라 정상 문장을 건드리지 않습니다. 실측 정상 설명
# "과일의 풍미를 담아 스페셜티 드립백을 만드는 푸룻티커피를 …" 에는 콜론이 없습니다.
#
# 쉼표로 쪼개지는 않습니다. 그 정상 설명에 "사과, 살구, 복숭아 등의 과일과" 가 있어
# 쪼개면 10자 미만 조각으로 흩어져 설명이 통째로 사라집니다. 조각 양끝의 쉼표만
# 떼어 냅니다(실측 노출값의 맨 앞 ", ").
_SNIPPET_FIELD_LABEL = re.compile(r"[가-힣A-Za-z]{2,10}\s+:")
_SNIPPET_EDGE = " :-·,、"


def clean_snippet(text: str, title: str) -> str | None:
    """검색 스니펫을 상품 설명으로 쓸 수 있게 다듬습니다.

    쓸 만한 문장이 남지 않으면 ``None`` 입니다. 마크다운 기호와 판매자 평점 안내문이
    섞인 채로 화면에 나가느니 비우는 편이 낫습니다. 문장 경계에서 자르므로 낱말
    중간에서 끊기지 않습니다.
    """
    body = _SNIPPET_MARKUP.split(_SNIPPET_TITLE_LINE.sub("", text or ""), maxsplit=1)[0]
    compact_title = re.sub(r"\s+", "", title or "")
    kept: list[str] = []
    length = 0
    for piece in _SNIPPET_SPLIT.split(body):
        piece = re.sub(r"\s+", " ", piece).strip(_SNIPPET_EDGE).strip()
        if len(piece) < 10 or any(noise in piece for noise in _SNIPPET_NOISE):
            continue
        if _SNIPPET_FIELD_LABEL.search(piece):
            continue
        # 제목을 그대로 되풀이하는 조각은 설명이 아닙니다.
        compact = re.sub(r"\s+", "", piece)
        if compact_title and (compact in compact_title or compact_title in compact):
            continue
        if length + len(piece) + 1 > _SNIPPET_MAX:
            break
        kept.append(piece)
        length += len(piece) + 1
    return " ".join(kept) if kept else None


# 검색 결과 제목에 붙어 오는 판매처·목록 잔재입니다. 실측(135자, gift 콜드·웜 양쪽):
#   "[스타벅스] 드립백 선물세트 - SSG.COM드립백 선물세트 - 추천•인기 상품, 신세계몰
#    드립백선물세트 - ... - 드립백세트 검색결과"
# 검색 페이지의 <title> 이 통째로 실려 온 것이라 상품명 뒤가 전부 목록 부스러기입니다.
#
# 스니펫과 달리 제목은 비울 수 없습니다. 제목이 사라지면 사용자가 어떤 상품인지
# 알아볼 수단이 없어집니다. 그래서 버리지 않고 **자릅니다**.
_TITLE_SITE_MARKERS = (
    # 카카오 선물하기 상품 페이지의 <title> 꼬리. "원 - 상품 : 선물하기" 처럼
    # 앞이 부실한 실측 제목도 있어 통째로 하나의 표식으로 둡니다.
    "상품 : 선물하기",
    "상품 : 카카오톡 선물하기",
    "카카오톡 선물하기",
    "SSG.COM",
    "SSG닷컴",
    "신세계몰",
    "신세계백화점",
    "이마트몰",
    "마켓컬리",
    "쿠팡",
    "G마켓",
    "지마켓",
    "11번가",
    "네이버쇼핑",
    "네이버 쇼핑",
    "롯데온",
    "올리브영",
    "옥션",
    "인터파크",
)
# 판매처 이름은 **구분자 뒤에** 있을 때만 잔재로 봅니다. 실측 정상 제목
# "G마켓 - PP 팬시 쇼핑백 10p 세트 ...", "[11번가] [용문전통시장] 로즈플로라 ..." 처럼
# 판매처가 제목 맨 앞에 오는 경우는 상품명의 일부라 건드리면 안 됩니다.
_TITLE_SITE_TAIL = re.compile(
    r"\s*[-|–—:]\s*(?:" + "|".join(re.escape(name) for name in _TITLE_SITE_MARKERS) + r")"
)
# 구분자가 없어도 제목이 아니라 목록 페이지임을 드러내는 말입니다.
_TITLE_LISTING_NOISE = re.compile(r"검색\s*결과|추천\s*[•·]\s*인기\s*상품")
# 실측 정상 제목의 최댓값은 63자입니다("G마켓 - PP 팬시 쇼핑백 10p 세트 18x14x6.5cm/...").
# 위 두 규칙이 못 잡은 반복 구간을 막는 마지막 그물이라 넉넉히 둡니다. 낱말 중간에서
# 끊기지 않게 구분자 자리에서 자릅니다.
_TITLE_MAX = 80
_TITLE_BOUNDARY = re.compile(r"[\s/,]")
# 자른 결과가 이보다 짧으면 상품을 알아볼 수 없으므로 원문을 그대로 씁니다.
# 실측 "원 - 상품 : 선물하기" 를 자르면 "원" 만 남습니다.
_TITLE_MIN = 4
# Tavily 가 넘겨주는 제목에 이미 붙어 있는 말줄임표입니다. 실측 45자
# "2P 볼라 고급 수건 선물 세트 40수 수건 210g 이사 생일 답례 신혼 ..." 은 _TITLE_MAX(80)
# 절단이 아니라 원본에 있던 것이라 길이 규칙으로는 잡히지 않습니다.
# 점 하나로 끝나는 제목은 정상일 수 있으므로 2개 이상만 봅니다.
_TITLE_ELLIPSIS = re.compile(r"\s*(?:\.{2,}|[…⋯]+)\s*$")


def clean_title(text: str) -> str:
    """검색 결과 제목에서 판매처·목록 잔재를 잘라 냅니다.

    ``clean_snippet`` 과 달리 결과가 비지 않습니다. 다듬은 결과가 상품을 알아볼 수
    없을 만큼 짧으면 원문을 그대로 돌려줍니다.
    """
    title = _TITLE_ELLIPSIS.sub("", re.sub(r"\s+", " ", text or "").strip())
    if len(title) <= _TITLE_MIN:
        return title

    cuts = [len(title)]
    site = _TITLE_SITE_TAIL.search(title)
    # start() > 0 이라야 합니다. 제목이 판매처 이름으로 시작하면 그건 상품명입니다.
    if site is not None and site.start() > 0:
        cuts.append(site.start())
    listing = _TITLE_LISTING_NOISE.search(title)
    if listing is not None and listing.start() > 0:
        cuts.append(listing.start())
    trimmed = title[: min(cuts)].strip(" -|–—:,·").strip()

    if len(trimmed) > _TITLE_MAX:
        head = trimmed[: _TITLE_MAX + 1]
        boundaries = [m.start() for m in _TITLE_BOUNDARY.finditer(head)]
        trimmed = head[: boundaries[-1]] if boundaries else head[:_TITLE_MAX]
        trimmed = trimmed.strip(" -|–—:,·").strip()

    # 잘라 낸 자리에 말줄임표가 새로 드러날 수 있습니다("… (7 ... - 상품 : 선물하기").
    trimmed = _TITLE_ELLIPSIS.sub("", trimmed)
    return trimmed if len(trimmed) >= _TITLE_MIN else title


# 선물 자체가 아니라 포장재만 파는 결과입니다. "생활용품" 같은 넓은 카테고리에서 걸려 나옵니다.
_PACKAGING_ONLY = ("쇼핑백", "포장지", "포장 박스", "리본끈", "선물박스", "택배박스")


def _is_packaging_only(title: str) -> bool:
    """포장재만 파는 상품인지. 선물로 추천할 물건이 아닙니다."""
    return any(word in title for word in _PACKAGING_ONLY)


# 특정 시기에만 의미가 있는 행사 상품입니다. 실측(8월)에서 "크리스마스 트리 미니트리
# 풀세트"가 꽃 답례의 **유일한** 추천으로 나갔습니다. 모델 판정은 "선물로 줄 수 있는
# 물건인가"만 보므로 이걸 통과시킵니다(실측 로그: 판정 5건 중 통과 4건에 포함).
#
# 달력은 코드가 이미 알고 있는 사실이라 모델에게 오늘 날짜를 쥐어 주는 대신 여기서
# 결정론적으로 거릅니다. 판정 프롬프트에 계절 지시를 더하면 요청마다 입력 토큰이
# 늘고 판단이 흔들리며, 외부 호출 없이는 검증할 수도 없습니다.
#
# 낱말은 오탐이 거의 없는 것만 넣습니다. "트리"는 "트리트먼트"에, "설"은 "설레는"에
# 걸리므로 넣지 않습니다. 허용 월은 앞뒤로 한 달을 두어 준비 기간을 남깁니다.
_SEASONAL_TERMS: tuple[tuple[tuple[str, ...], frozenset[int]], ...] = (
    (("크리스마스", "성탄", "산타", "루돌프"), frozenset({11, 12})),
    (("밸런타인", "발렌타인"), frozenset({1, 2})),
    (("화이트데이",), frozenset({2, 3})),
    (("할로윈", "핼러윈"), frozenset({9, 10})),
    (("설날", "구정"), frozenset({12, 1, 2})),
    (("추석", "한가위"), frozenset({8, 9, 10})),
    (("어버이날",), frozenset({4, 5})),
    (("수능",), frozenset({10, 11})),
)


def _current_month() -> int:
    """오늘이 몇 월인지. 테스트에서 갈아 끼울 수 있게 함수로 둡니다."""
    return date.today().month


# 국내 쇼핑몰 제목은 상품명 뒤에 검색 키워드를 늘어놓습니다. 그 자리에 있는 행사
# 낱말은 상품의 성격이 아니라 판매자가 노린 검색어입니다. 실측(8월)에서
# "천리까지 향이 천리향 분재 화산석화분 크리스마스" 가 마지막 낱말 하나 때문에
# 걸렸는데, 천리향 분재는 사철 식물이라 8월에 빼야 할 이유가 없습니다.
#
# 같은 실측에서 진짜로 빼야 했던 "크리스마스 트리 미니트리 풀세트 눈꽃" 은 행사
# 낱말이 **앞**에 서서 뒤따르는 명사를 수식합니다. 둘을 가르는 신호가 이것입니다.
#
# 앞에 상품명이 남아 있을 때만 꼬리로 봅니다. 제목이 통째로 "크리스마스" 라면
# 그건 꼬리가 아니라 상품 그 자체입니다.
_SEASONAL_TAIL_MIN_LEAD = 10


def out_of_season(title: str, month: int) -> str | None:
    """지금 보내면 시기가 어긋나는 행사 상품인지.

    제목 **끝에** 붙은 행사 낱말은 검색 키워드로 보고 넘깁니다. 후보가 모자란
    상황에서 정상 상품을 지우는 손해가, 8월에 판촉 상품 하나를 놓치는 손해보다
    큽니다(P0-2: 실측 gift 흐름의 후보가 5~7건뿐이었습니다).

    Returns:
        어긋나면 그렇게 판단한 낱말, 아니면 ``None``. 로그에 근거를 남기려고
        참/거짓 대신 낱말을 돌려줍니다.
    """
    compact = re.sub(r"\s+", "", title or "")
    for terms, months in _SEASONAL_TERMS:
        if month in months:
            continue
        for term in terms:
            if term not in compact:
                continue
            if compact.endswith(term) and len(compact) - len(term) >= _SEASONAL_TAIL_MIN_LEAD:
                continue
            return term
    return None


def _is_content_page(url: str) -> bool:
    """상품이 아니라 기사·기획전을 서비스하는 주소인지.

    검색·목록 페이지는 ``_is_product_detail_url`` 이 URL 패턴으로 이미 걸러 냅니다.
    예전에는 제목에 "베스트"·"랭킹" 같은 말이 있으면 목록으로 봤지만, 쿠팡·네이버의
    정상 상품 제목에 흔한 표현이라 멀쩡한 상세페이지가 함께 떨어졌습니다.
    """
    return (urlparse(url).hostname or "").lower() in _CONTENT_HOSTS


async def filter_relevant(
    batches: list[list[ProductSuggestion]],
    examples: list[str | None],
) -> list[list[ProductSuggestion]]:
    """후보 전체를 한 번에 판정해 추천할 만한 상품만 남깁니다.

    모델 판정이 기본이고, 모델이 빠뜨렸거나 호출이 실패한 항목만 키워드로 판정합니다.
    필터 하나 때문에 추천이 통째로 죽지 않도록 폴백을 남겨 둡니다.

    시기가 어긋난 행사 상품은 모델에 묻기 **전에** 뺍니다. 모델은 "선물로 줄 수 있는
    물건인가"를 보므로 8월의 크리스마스 트리도 통과시킵니다(실측). 달력은 코드가 아는
    사실이라 여기서 정하고, 뺀 만큼 판정 프롬프트의 입력 토큰도 줄어듭니다.
    """
    flat: list[tuple[int, int, ProductSuggestion]] = [
        (batch_index, item_index, item)
        for batch_index, batch in enumerate(batches)
        for item_index, item in enumerate(batch)
    ]
    if not flat:
        return batches

    month = _current_month()
    in_season: list[tuple[int, int, ProductSuggestion]] = []
    for entry in flat:
        term = out_of_season(entry[2].title, month)
        if term is None:
            in_season.append(entry)
        else:
            logger.info(
                "철 지난 행사 상품 제외 %d월 기준 '%s' category=%s title=%s",
                month,
                term,
                entry[2].category,
                entry[2].title,
            )

    verdicts: dict[int, bool] = {}
    if in_season and product_filter.is_available():
        verdicts = (
            await product_filter.judge([(item.category, item.title) for _, _, item in in_season])
            or {}
        )

    kept: list[list[ProductSuggestion]] = [[] for _ in batches]
    for position, (batch_index, _, item) in enumerate(in_season):
        decision = verdicts.get(position)
        if decision is None:
            decision = not _is_packaging_only(item.title) and _is_semantically_relevant(
                item.category, examples[batch_index], item.title, ""
            )
        if decision:
            kept[batch_index].append(item)
        else:
            logger.info("추천 부적합 제외 category=%s title=%s", item.category, item.title)
    logger.info("적합성 판정 후보 %d건 중 %d건 통과", len(flat), sum(len(b) for b in kept))
    return kept


def build_query(category: str, example: str | None, low: int, high: int) -> str:
    """검색어를 만듭니다.

    카테고리명("식품·디저트")만으로는 검색이 잘 되지 않아, 모델이 낸 구체적인
    상품 유형("프리미엄 디저트 세트")을 앞세우고 가격대를 덧붙입니다.

    가격 힌트는 상한이 아니라 **범위 중앙값**을 씁니다. 상한을 쓰면 4만~24만원 같은
    넓은 범위에서 29만원짜리만 걸려 나옵니다.

    "만원대"는 1만원 폭의 구간입니다(1만원대 = 10,000~19,999원). 예산이 그보다
    좁으면 이 구간이 예산 밖을 가리킵니다. 4차 실측 gift 가 정확히 그랬습니다.

        예산 8,000~12,000 → 허용(±15%) 6,800~13,800 → 힌트 "1만원대" 10,000~19,999
        힌트 구간의 62% 가 노출 불가 구간이고, 예산의 아래 절반(6,800~9,999)은
        힌트에서 아예 빠집니다. 검색이 위쪽을 겨냥하도록 만들어 놓은 셈입니다.
        실제로 돌아온 후보는 19,100 · 19,900 · 23,990 · 32,000 · 45,000원이었고
        노출 0건으로 끝났습니다.

    그래서 구간이 허용 범위 **안에 들어갈 때만** "만원대"를 씁니다. 4차의 나머지 두
    흐름은 이 조건을 이미 만족하므로 검색어가 그대로입니다(18,000~27,000 → "2만원대",
    28,000~42,000 → "3만원대"). 고장난 한 곳만 바뀝니다.
    """
    seed = (example or category).strip()
    middle = (low + high) // 2
    if middle < 10_000:
        return f"{seed} 선물 {middle}원"
    slack = max(0.0, settings.product_price_slack_ratio)
    decade = middle // 10_000
    fits = decade * 10_000 >= low * (1 - slack) and decade * 10_000 + 9_999 <= high * (1 + slack)
    price_hint = f"{decade}만원대" if fits else f"{decade}만원"
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
        skipped = 0
        # 어느 호스트의 어떤 주소를 버렸는지 남깁니다. 4차 실측에서 원본 34건 중
        # 24건(71%)이 이 자리에서 조용히 사라졌고(gift 웜: 생활용품 10건 중 9건,
        # 식품·디저트 12건 중 10건), 로그에 개수만 있어 _is_product_detail_url 의
        # 어느 패턴이 모자란지 알 수 없었습니다. 후보 부족이 상품 0건의 실제 원인인데
        # 가장 크게 깎이는 지점이 진단 불가였습니다.
        dropped: dict[str, str] = {}
        for item in results:
            url = str(item.get("url") or "")
            # 잔재를 여기서 잘라 냅니다. 이 값이 판정 프롬프트·카테고리 검증·화면까지
            # 그대로 흘러가므로, 한 곳에서 다듬어야 셋이 같은 제목을 봅니다.
            title = clean_title(str(item.get("title") or ""))
            if not url or not title:
                skipped += 1
                dropped.setdefault(_source_name(url) if url else "제목·주소 없음", url)
                continue
            content = str(item.get("content") or "")
            # 최종 추천에는 검색·목록·기사 URL을 넣지 않습니다. 사용자가 링크를 눌렀을 때
            # 바로 특정 상품의 가격과 구매 버튼이 보이는 상세페이지여야 합니다.
            if not _is_product_detail_url(url) or _is_content_page(url):
                skipped += 1
                dropped.setdefault(_source_name(url), url)
                continue
            # 포장재 여부와 카테고리 적합성은 여기서 거르지 않습니다. 검색마다 따로
            # 판정하면 모델을 검색 횟수만큼 부르게 되므로, 후보를 다 모은 뒤
            # ``filter_relevant`` 가 한 번에 판정합니다.
            suggestions.append(
                ProductSuggestion(
                    title=title[:200],
                    url=url,
                    source=_source_name(url),
                    category=category,
                    # 제목의 금액을 먼저 보고, 없으면 스니펫에서 찾습니다. 둘 다
                    # 제안 가격대의 절반~두 배를 벗어난 숫자는 상품가로 보지 않습니다.
                    # 여기 값은 어디까지나 후보이고, 확정은 Extract 가 합니다.
                    price=extract_title_price(title, low, high)
                    or extract_price(content, low, high),
                    kind="product",
                    snippet=clean_snippet(content, title),
                )
            )
        logger.info(
            "상품 검색 category=%s query=%s 결과 %d건 중 상세페이지 %d건(상세 아님 %d건 제외)%s",
            category,
            query,
            len(results),
            len(suggestions),
            skipped,
            (" 버린 주소=" + ", ".join(f"{name} {url}" for name, url in dropped.items()))
            if dropped
            else "",
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
                "판매가 확인 묶음이 %.0f초를 넘겨 건너뜁니다(%d건: %s). 그 시간만큼 응답이 늦어집니다.",
                settings.tavily_extract_timeout_seconds,
                len(products),
                ", ".join(_source_name(p.url) for p in products),
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
        targets = [p for p in products if p.kind == "product"]
        if not targets:
            return products

        # 1) 원본 HTML 에서 구조화된 판매가를 먼저 시도합니다. 성공하면 Extract 크레딧을
        #    쓰지 않고, Extract 가 마크다운 변환에서 잃어버리는 값도 잡습니다.
        if settings.product_price_fetch_enabled:
            direct = await asyncio.gather(
                *(fetch_price_direct(item.url, client) for item in targets),
                return_exceptions=True,
            )
            for item, price in zip(targets, direct):
                if isinstance(price, int) and price > 0:
                    item.price = price
                    item.price_verified = True

        # 2) 직접 읽지 못한 건만 Extract 로 넘깁니다. 크레딧이 드는 쪽이라 여기에만
        #    상한을 겁니다. 직접 조회는 비용이 없으므로 후보 전체에 시도합니다.
        remaining = [item for item in targets if not item.price_verified][
            : settings.tavily_extract_limit
        ]
        if remaining:
            size = max(1, settings.tavily_extract_batch_size)
            batches = [remaining[i : i + size] for i in range(0, len(remaining), size)]
            started = time.monotonic()
            await asyncio.gather(
                *(self._extract_batch(batch, client) for batch in batches),
                return_exceptions=True,
            )
            # 이 단계는 판정과 나란히 돌지만 둘 중 늦은 쪽이 응답 시간이 됩니다.
            # 걸린 시간 대비 확정 건수가 남아야 타임아웃 값을 근거로 조정할 수 있습니다.
            logger.info(
                "판매가 Extract %d건(묶음 %d개) → %d건 확정, %.1f초",
                len(remaining),
                len(batches),
                sum(1 for item in remaining if item.price_verified),
                time.monotonic() - started,
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
        *,
        stats: SearchStats | None = None,
    ) -> list[ProductSuggestion]:
        """카테고리별로 검색하고, 실제 판매가를 확인한 뒤 골라 돌려줍니다.

        Args:
            categories: (카테고리명, 대표 상품 유형) 목록. 모델이 낸 순서를 그대로 씁니다.
            low: 추천 가격 하한.
            high: 추천 가격 상한.
            limit: 최종 상품 수. 기본값은 설정값.
            stats: 주면 심사한 후보 수를 채웁니다. 상품 0건일 때 "검색 결과가 없었다"
                와 "찾았지만 가격이 맞지 않았다" 를 호출 측이 구분해 말하기 위한
                값이며, 반환값은 이 인자와 무관합니다.

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
            kept_examples: list[str | None] = []
            for (_, example), batch in zip(categories, batches):
                if isinstance(batch, BaseException):
                    logger.warning("상품 검색 중 예외: %s", batch)
                    continue
                clean.append(batch)
                kept_examples.append(example)

            # 판정과 가격 확인을 동시에 돌립니다. 둘은 서로를 기다릴 이유가 없고,
            # 순차로 두면 각 2초씩 4초가 그대로 응답 시간이 됩니다(실측).
            # 가격 확인은 판정에서 떨어질 상품까지 포함하지만, 직접 조회는 비용이
            # 없고 Extract 는 상한이 걸려 있어 낭비가 크지 않습니다.
            #
            # enrich_prices 는 ProductSuggestion 을 제자리에서 고치므로, 판정 뒤
            # 살아남은 객체에도 가격이 그대로 반영됩니다.
            everything = [item for batch in clean for item in batch]
            clean, _ = await asyncio.gather(
                filter_relevant(clean, kept_examples),
                self.enrich_prices(everything, client),
            )

            # 가격이 확정된 뒤에 골라야 예산에 맞는 상품이 뽑힙니다.
            candidates = _interleave(
                [sorted(b, key=lambda s: (s.kind != "product",)) for b in clean],
                settings.product_candidate_limit,
            )

        ranked = _select_by_price(candidates, low, high, limit, stats=stats)
        has_in_range = any(
            item.price is not None and low <= item.price <= high for item in ranked
        )
        for item in ranked:
            item.reason = _reason(item, low, high, has_in_range=has_in_range)
        return ranked


def _reason(
    item: ProductSuggestion, low: int, high: int, *, has_in_range: bool = False
) -> str:
    """이 상품을 고른 이유를 한 문장으로. 화면에 그대로 보여 줄 수 있습니다.

    Args:
        has_in_range: 이번 응답에 예산 안 상품이 함께 나가는지. 예산 밖 상품이
            "찾지 못해 대신 보여 주는 것"인지 "빈자리를 채운 참고"인지가 달라집니다.
    """
    # "{카테고리}로 고른" 은 받침에 따라 "상품권로"·"생활용품로" 가 되므로 조사를
    # 붙이지 않아도 되는 "선물로" 를 씁니다.
    parts = [f"{item.category} 선물로 고른 {item.source} 상품"] if item.category else [f"{item.source} 상품"]
    if item.price is None:
        # ``_select_by_price`` 가 가격 미상 후보를 뽑지 않으므로 여기 오지 않습니다.
        # 아래 포맷이 None 에서 터지지 않게 남겨 둔 방어선입니다.
        parts.append("가격은 링크에서 확인이 필요합니다")
        return ". ".join(parts)[:200]

    # 미확인 가격도 예산 밖이면 그 사실을 적습니다. 예전에는 "검색 기준 약 N원(확인
    # 필요)"에서 끝나 예산을 벗어났다는 말이 빠졌습니다.
    amount = (
        f"판매가 {item.price:,}원"
        if item.price_verified
        else f"검색 기준 약 {item.price:,}원(확인 필요)"
    )
    if low <= item.price <= high:
        parts.append(f"{amount}으로 제안 가격대 안입니다" if item.price_verified else amount)
        return ". ".join(parts)[:200]

    gap = "높습니다" if item.price > high else "낮습니다"
    parts.append(f"{amount}으로 제안 가격대보다 {gap}")
    parts.append(
        "가격대 안 상품이 모자라 가까운 것도 함께 보여 드립니다"
        if has_in_range
        else "가격대 안 상품을 찾지 못해 가장 가까운 것으로 보여 드립니다"
    )
    return ". ".join(parts)[:200]


def _rank(item: ProductSuggestion, low: int, high: int) -> tuple:
    """검색 결과를 좋은 순으로 정렬하는 기준.

    우선순위는 이렇습니다.
    1. 상품 페이지에서 확인한 판매가가 추천 범위 안에 드는 것.
       9,000~14,000원을 권해 놓고 20,000원짜리를 맨 앞에 보여 주면 추천의 의미가 없습니다.
    2. 판매가를 확인한 것. 스니펫의 숫자는 같은 브랜드 다른 옵션의 가격일 수 있습니다.
    3. 미확인이라도 범위 안으로 보이는 것.
    4. 가격을 아는 것.

    카테고리 점수는 여기 없지만 순서에는 반영됩니다. ``_select_by_price`` 의 정렬이
    안정 정렬이고 입력이 ``_interleave`` 가 만든 카테고리 순서라, **이 네 조건이 같은
    상품들 사이에서는 점수가 높은 카테고리가 앞에 남습니다**. 그 순서는
    ``recommendation_policy.normalize_recommendation`` 이 점수 내림차순으로 세웁니다.

    가격 적합성이 카테고리 점수보다 먼저인 이유: 가격은 사용자에게 숫자로 보여 준
    약속이고 카테고리 점수는 가격을 보기 전에 모델이 매긴 선호도입니다. 점수를 앞에
    두면 8,000~12,000원 예산에 23,990원(커피·차 85)짜리가 9,800원(생활용품 60)짜리보다
    앞에 서게 되는데, 그건 이 라운드에 고친 예산 위반 그 자체입니다.
    """
    in_range = item.price is not None and low <= item.price <= high
    return (
        not (item.price_verified and in_range),
        not item.price_verified,
        not in_range,
        item.price is None,
    )


def _distance_from_range(item: ProductSuggestion, low: int, high: int) -> int:
    """제안 가격대에서 얼마나 떨어졌는지. 범위 안이면 0."""
    if item.price is None:
        return 0
    if item.price < low:
        return low - item.price
    return max(0, item.price - high)


def _select_by_price(
    candidates: list[ProductSuggestion],
    low: int,
    high: int,
    limit: int,
    *,
    stats: SearchStats | None = None,
) -> list[ProductSuggestion]:
    """예산 안 상품을 앞세우고, 자리가 남으면 가격이 가까운 것으로 ``limit`` 까지 채웁니다.

    예전에는 갈래를 하나 고르고 끝냈습니다. 그래서 예산 안 후보가 1건이면 상한이 3인데도
    1건만 나갔습니다(실측 4회 모두 1건: 후보 11·4·12·10건 중 노출 1건). 사용자는 비교할
    대안 없이 한 건만 보게 됩니다.

    지금은 우선순위 단계를 순서대로 훑으며 빈자리를 채웁니다.
      1) 확인된 판매가가 예산 안  2) 검색 기준 가격이 예산 안(미확인)
      3) 예산 밖이지만 가까운 확인가  4) 예산 밖이지만 가까운 검색 기준 가격

    3·4단계는 경계에서 ``product_price_slack_ratio`` 안에 드는 것만 씁니다. 예전에는
    "절반~두 배"(-50%~+100%)를 썼는데 실측에서 무너졌습니다. 노출 10건 중 예산 안이
    3건뿐이었고, 사용자가 18,000~27,000원을 **직접 지정**한 요청에 49,000원(+81%)이
    나갔습니다. 그 폭은 ``extract_price`` 가 "이 숫자를 상품가로 볼 수 있는가" 를 판단할
    때 쓰는 값이지, "이 상품을 권해도 되는가" 의 기준이 아닙니다. 두 물음을 같은 숫자로
    답한 것이 잘못이었습니다. 15% 를 고른 근거는 config 주석에 있습니다.

    네 단계 모두 ``price is not None`` 을 요구합니다. 즉 **가격을 전혀 모르는 상품은
    어떤 경로로도 뽑히지 않습니다**. 채울 것이 없으면 그냥 적게, 없으면 0건으로 나갑니다.
    예산 밖이라는 사실은 상품마다 ``reason`` 에 적습니다.
    """
    detail_products = [item for item in candidates if item.kind == "product"]
    if stats is not None:
        stats.examined = len(detail_products)
    slack = max(0.0, settings.product_price_slack_ratio)
    floor, ceiling = low * (1 - slack), high * (1 + slack)

    def in_range(item: ProductSuggestion) -> bool:
        return item.price is not None and low <= item.price <= high

    def nearby(item: ProductSuggestion) -> bool:
        return item.price is not None and not in_range(item) and floor <= item.price <= ceiling

    def by_rank(pool: list[ProductSuggestion]) -> list[ProductSuggestion]:
        return sorted(pool, key=lambda item: _rank(item, low, high))

    def by_distance(pool: list[ProductSuggestion]) -> list[ProductSuggestion]:
        return sorted(
            pool,
            key=lambda item: (_distance_from_range(item, low, high), _rank(item, low, high)),
        )

    tiers = (
        (
            "확인된 판매가가 예산 안",
            by_rank([i for i in detail_products if i.price_verified and in_range(i)]),
        ),
        (
            "검색 기준 가격이 예산 안(미확인)",
            by_rank([i for i in detail_products if not i.price_verified and in_range(i)]),
        ),
        (
            "예산 밖이지만 가까운 확인가로 보충",
            by_distance([i for i in detail_products if i.price_verified and nearby(i)]),
        ),
        (
            "예산 밖이지만 가까운 검색 기준 가격으로 보충",
            by_distance([i for i in detail_products if not i.price_verified and nearby(i)]),
        ),
    )
    chosen: list[ProductSuggestion] = []
    basis: list[str] = []
    for name, pool in tiers:
        if len(chosen) >= limit:
            break
        take = pool[: limit - len(chosen)]
        if take:
            chosen.extend(take)
            basis.append(name)
    if chosen:
        _log_selection(" + ".join(basis), chosen, detail_products, low, high)
        return chosen

    # 아무것도 못 고른 경우입니다. 예전에는 여기서 거리 제한 없이 "가장 가까운
    # 순"으로 자리를 채웠습니다. 그 갈래가 실측의 49,000원(+81%)·23,990원(+100%)을
    # 만들어 냈습니다. 위에서 폭을 좁혀 놓고 여기서 무제한으로 다시 들이면 좁힌 의미가
    # 없으므로, 가격을 아는 후보는 여기서 더 보지 않습니다.
    #
    # 가격을 **모르는** 후보도 내보내지 않습니다. 직전 라운드에는 "금액을 말하지 않으니
    # 예산과 어긋날 일이 없다"고 보고 통과시켰는데, 실측이 그 판단을 반박했습니다.
    # 8,000~12,000원 요청에 "[선물] 명품 나주배 세트 5kg(8-10과)" 이 유일한 추천으로
    # 나갔고(콜드·웜 2/2), 같은 응답의 product_basis 는 "0개가 8,000원 ~ 12,000원 안에
    # 듭니다" 였습니다. 가격을 모른다는 것은 예산과 어긋나지 않는다는 뜻이 아니라
    # **어긋났는지 확인할 방법이 없다**는 뜻입니다.
    #
    # 예산은 이 서비스가 숫자로 내건 약속입니다. 지켰는지 확인할 수 없는 상품을 내보내면
    # 그 확인을 사용자에게 떠넘기게 되고, 사용자는 링크를 눌러 보기 전까지 추천이 맞는지
    # 알 수 없습니다. 그래서 상품 0건으로 나갑니다. 그때 응답은 카테고리와 가격대만
    # 제안하며, 후보가 있었는데 가격이 맞지 않았다는 사실은
    # ``recommendation_rationale.product_basis`` 가 ``examined`` 로 받아 말합니다.
    # 라벨은 실제 상황을 말해야 합니다. 4차 실측에서 "예산 근처 후보 없음" 이라고
    # 찍혔지만 실제로는 후보가 7건 있었고 그중 8건 중 2건은 판매가까지 확인된
    # 상태였습니다(gift 웜). 사용자 문구("판매가를 확인하지 못해")가 맞고 로그가
    # 틀렸던 것입니다. 셋을 갈라야 다음 라운드가 어디를 고칠지 알 수 있습니다.
    #   후보가 없다 → 검색·상세페이지 판정을 볼 일
    #   가격을 아무도 모른다 → 가격 확인 경로(직접 조회·Extract)를 볼 일
    #   가격은 아는데 전부 예산 밖 → 검색어와 예산이 어긋난 것
    priced = [item for item in detail_products if item.price is not None]
    if not detail_products:
        basis = "상세페이지 후보 없음"
    elif not priced:
        basis = f"후보 {len(detail_products)}건 전원 판매가 미상"
    else:
        nearest = min(priced, key=lambda item: _distance_from_range(item, low, high))
        verified = sum(1 for item in priced if item.price_verified)
        basis = (
            f"가격을 아는 후보 {len(priced)}건(확인 {verified}건)이 모두 예산 밖 "
            f"허용={floor:,.0f}~{ceiling:,.0f} 가장 가까운 값={nearest.price:,}"
        )
    _log_selection(basis, [], detail_products, low, high)
    return []


def _log_selection(
    basis: str,
    chosen: list[ProductSuggestion],
    pool: list[ProductSuggestion],
    low: int,
    high: int,
) -> None:
    """무엇을 왜 골랐고 몇 건이 여기서 떨어졌는지 남깁니다.

    후보가 줄어드는 마지막 지점입니다. 여기가 조용하면 "판정은 5건 통과인데 응답은
    1건"의 원인을 로그만 보고는 알 수 없습니다.
    """
    logger.info(
        "상품 선별 기준=%s 후보 %d건 → 노출 %d건(탈락 %d건) 예산=%s~%s 노출가=%s",
        basis,
        len(pool),
        len(chosen),
        len(pool) - len(chosen),
        f"{low:,}",
        f"{high:,}",
        ", ".join(f"{item.price:,}" if item.price is not None else "미상" for item in chosen)
        or "없음",
    )


def _interleave(batches: list[list[ProductSuggestion]], limit: int) -> list[ProductSuggestion]:
    """카테고리별 결과를 한 개씩 번갈아 뽑습니다.

    한 카테고리가 결과를 독차지하면 추천이 단조로워집니다.
    """
    picked: list[ProductSuggestion] = []
    seen: set[str] = set()
    total = sum(len(batch) for batch in batches)
    duplicates = 0
    for round_index in range(max((len(b) for b in batches), default=0)):
        for batch in batches:
            if round_index >= len(batch):
                continue
            item = batch[round_index]
            key = _canonical_product_key(item.url)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            picked.append(item)
            if len(picked) >= limit:
                _log_interleave(total, picked, duplicates)
                return picked
    _log_interleave(total, picked, duplicates)
    return picked


def _log_interleave(total: int, picked: list[ProductSuggestion], duplicates: int) -> None:
    if total != len(picked):
        logger.info(
            "후보 정리 %d건 → %d건(같은 상품 %d건, 후보 상한 초과 %d건)",
            total,
            len(picked),
            duplicates,
            total - len(picked) - duplicates,
        )


product_search = TavilyProductSearch()



async def lookup_price(name: str, brand: str | None = None) -> int | None:
    """상품명으로 실제 판매가를 찾습니다. 이미지에 금액이 없을 때 씁니다.

    카테고리 추정가는 브랜드를 모릅니다. 실측에서 TWG Tea 티백 선물이 "음료"
    카테고리로 분류돼 10,000원으로 추정됐지만, 실제 판매가는 36,000~76,000원이었습니다.
    3~7배 차이는 답례 가격대를 통째로 어긋나게 합니다.

    같은 상품의 다른 용량·구성이 섞이므로 **중앙값**을 씁니다. 최저가를 쓰면 낱개
    상품에, 최고가를 쓰면 대용량에 끌립니다. 정확한 SKU 매칭은 목표가 아니고,
    카테고리 추정가보다 나은 값을 찾는 것이 목표입니다.

    Returns:
        찾은 판매가의 중앙값. 검색이 불가능하거나 가격을 하나도 못 읽으면 ``None``.
    """
    if not product_search.is_available or not name.strip():
        return None

    query = name.strip()
    if brand and brand.strip() and brand.strip().lower() not in query.lower():
        query = f"{brand.strip()} {query}"

    async with httpx.AsyncClient(timeout=settings.tavily_timeout_seconds) as client:
        try:
            response = await client.post(
                settings.tavily_url,
                json={
                    "query": query,
                    "search_depth": settings.tavily_search_depth,
                    "max_results": settings.tavily_max_results,
                    "include_domains": list(settings.product_search_domains),
                    "topic": "general",
                },
                headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            )
        except httpx.HTTPError as exc:
            logger.warning("판매가 검색 실패 query=%s: %s", query, exc)
            return None
        if response.status_code != 200:
            logger.warning("판매가 검색 HTTP %s query=%s", response.status_code, query)
            return None

        try:
            results = response.json().get("results") or []
        except ValueError:
            return None
        urls = [
            str(item.get("url") or "")
            for item in results
            if _is_product_detail_url(str(item.get("url") or ""))
        ][: settings.product_price_lookup_limit]
        if not urls:
            logger.info("판매가 검색: 상세 상품을 찾지 못했습니다 query=%s", query)
            return None

        found = await asyncio.gather(*(fetch_price_direct(url, client) for url in urls))

    prices = sorted(price for price in found if isinstance(price, int) and price > 0)
    if not prices:
        logger.info("판매가 검색: 가격을 읽지 못했습니다 query=%s", query)
        return None
    median = prices[len(prices) // 2]
    logger.info("판매가 검색 query=%s 후보=%s 중앙값=%s", query, prices, median)
    return median
