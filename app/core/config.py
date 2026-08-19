"""환경변수와 .env 파일에서 Giftie 실행 설정을 읽습니다."""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션, 인증, 모델 및 타임아웃 설정."""
    app_name: str = "giftie-ai-service"
    api_key: str = "local-development-key"
    # mock | bedrock | vllm | mlx | transformers
    # bedrock 은 GPU 없이 쓰는 관리형 경로, vllm 은 자체 GPU 경로다.
    # 둘 다 추천과 이미지 분석에 같은 모델을 쓴다. mlx/transformers 는 Mac 로컬
    # 개발용이며, 이 두 값에서는 이미지 분석이 mock 으로 떨어진다(VLM 을 못 돌리기 때문).
    model_backend: str = "bedrock"
    model_id: str = "Qwen/Qwen3-4B"
    local_model_id: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    preload_model: bool = False
    max_new_tokens: int = 600
    # 추천·메시지 생성용 샘플링. **vLLM/Gemma 경로 전용**이다. Gemma 공식 권장값이고,
    # 그 경로는 response_format(json_schema)이 형식을 강제하므로 1.0 이어도 JSON 이
    # 깨지지 않는다. Bedrock(Claude) 경로는 구조화 출력이 없어 형식을 프롬프트로만
    # 요구하므로 아래 bedrock_temperature 를 따로 쓴다.
    # (이미지 추출은 아래 vision_temperature 로 따로 둔다)
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    # ------------------------------------------------------------------- 요청 예산
    # /from-image 는 "이미지 분석 → 네 후속 작업" 순으로 **직렬** 실행되므로 최악 지연은
    # 두 값의 합이다. README 가 백엔드에 권장하는 HTTP 타임아웃은 90초이므로, 합이
    # 그보다 확실히 낮아야 백엔드가 먼저 끊는 일이 없다. 45 + 30 = 75초로 두어
    # 15초를 네트워크·직렬화 여유로 남긴다.
    #
    # 짧게 잡아 정상 요청을 죽이는 쪽이 백엔드에서 끊기는 것보다 나쁘므로 넉넉히 잡았다.
    # 지금까지 나온 기준선은 /recommend 20.17초 하나뿐이라(task 쪽만 해당) 그대로 둔다.
    # 이미지 분석 기준선이 나오면 image_analysis 쪽을 줄일 것.
    #
    # task_timeout_seconds 는 /from-image·/from-gift-data 의 네 후속 작업과
    # **/recommend 라우터**가 함께 쓴다(routers/agent.py). 셋 다 "모델 1회 + 상품 검색"
    # 이라는 같은 일이므로, 한 번의 추천에 허용된 시간이 경로마다 갈리면 안 된다.
    #
    # 주의: 이 둘은 예전의 단일 REQUEST_TIMEOUT_SECONDS 를 대체한다. 그 값은 두 단계에
    # 각각 걸려 최악 2배(.env 의 60초 → 120초)가 됐고 권장값 90초를 넘었다.
    image_analysis_timeout_seconds: float = 45.0
    task_timeout_seconds: float = 30.0
    # --------------------------------------------------------------- 공용 Bedrock 엔진
    # 추천과 이미지 분석이 Amazon Bedrock 의 같은 Claude 모델을 쓴다. GPU 가 필요 없고
    # 모델 적재 시간도 없으므로 model_backend="bedrock" 이면 두 기능 모두 이 경로다.
    #
    # 계정마다 열려 있는 호출 경로가 다르다.
    #   invoke: 레거시 bedrock-runtime(InvokeModel). 추론 프로파일 ID 를 쓴다.
    #           예) us.anthropic.claude-haiku-4-5-20251001-v1:0
    #   mantle: Messages 엔드포인트(bedrock-mantle.{region}.api.aws). anthropic. 접두사 ID.
    #           예) anthropic.claude-haiku-4-5
    # 잘못 고르면 모든 모델이 403 이 된다. 403 이면 이 값을 가장 먼저 의심할 것.
    bedrock_api_style: str = "invoke"
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    # 이미지 분석은 추천보다 어렵습니다. Haiku 4.5 는 카카오톡 말풍선 위치와 프로필
    # 사진을 잘못 읽어 방향(received/sent)과 상대 이름을 틀립니다(실측). Sonnet 은
    # 같은 이미지를 정확히 읽으므로 비전만 상위 모델로 분리합니다.
    bedrock_vision_model_id: str = "global.anthropic.claude-sonnet-4-6"
    # max_new_tokens(600) 는 Gemma 기준입니다. Claude 는 스키마를 프롬프트로 받는 만큼
    # 출력이 길어 600 에서는 JSON 이 잘립니다(실측). 그래서 별도 예산을 둡니다.
    bedrock_max_tokens: int = 2_048
    # Bedrock 전용 샘플링. 위 temperature(1.0)는 Gemma 권장값이고 vLLM 경로는
    # response_format 이 JSON 을 강제하지만, Bedrock 경로는 구조화 출력이 없어
    # 형식 준수를 프롬프트에만 의존한다(_generate_with_bedrock 주석 참고).
    # 1.0 은 키 이름을 지어내거나 필드를 빠뜨릴 확률을 올리고, 그러면 응답이
    # BEDROCK_CLAUDE_FALLBACK 으로 떨어져 모델이 만든 추천이 통째로 버려진다.
    # 반대로 0.0 은 suggested_message 가 매번 같은 문장 틀로 굳는다. 형식 안정성을
    # 우선하되 문장에 최소한의 변주를 남기는 값으로 0.4 를 쓴다.
    bedrock_temperature: float = 0.4
    bedrock_max_retries: int = 2
    bedrock_timeout_seconds: float = 90.0
    # 인증은 아래 둘 중 하나만 쓴다. 함께 지정하면 SDK 가 거부한다.
    #   1) Bedrock API 키(Bearer 토큰). SDK 가 쓰는 환경변수 이름으로 .env 에 적어도 인식한다.
    #   2) 미지정 시 표준 AWS credential chain(환경변수 / ~/.aws / IAM 역할)의 SigV4.
    bedrock_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("BEDROCK_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"),
    )
    bedrock_aws_profile: str | None = None
    # 사설 엔드포인트(VPC/PrivateLink)나 로컬 검증 스텁을 쓸 때만 지정한다.
    bedrock_base_url: str | None = None

    # ------------------------------------------------------------------ 공용 vLLM 엔진
    # 추천과 이미지 분석이 같은 모델(Gemma4-12B-QAT + MTP)을 같은 vLLM 서버에서 쓴다.
    # GPU 한 장에 모델을 두 벌 올리지 않으므로 메모리와 기동 시간이 모두 절약된다.
    # model_backend 를 "vllm" 으로 두면 두 기능 모두 이 엔진을 사용한다.
    # FastAPI 가 8000 을 쓰므로 vLLM 은 8001 로 띄운다(-p 8001:8000).
    vllm_base_url: str = "http://localhost:8001"
    vllm_model: str = "gemma4-12b-qat"
    vllm_api_key: str = "EMPTY"
    vllm_timeout_seconds: float = 90.0

    # 이미지 분석 전용 생성 파라미터. 엔진은 같지만 추출은 창의성이 필요 없어 temperature 0 이다.
    vision_max_new_tokens: int = 900
    vision_temperature: float = 0.0

    # 이미지 다운로드·정규화
    image_max_bytes: int = 12 * 1024 * 1024
    image_max_edge: int = 1280  # 장변 리사이즈. 벤치 이미지(720x1280)에는 무변환
    image_fetch_timeout_seconds: float = 15.0
    # 기본값은 사설·루프백 주소 차단(SSRF 방어)이다.
    # 로컬에서 이미지를 직접 띄워 종단 테스트할 때만 켜고, 운영에서는 절대 켜지 않는다.
    allow_private_image_hosts: bool = False

    # GiftData.gift_price 는 필수이고 0을 못 받는다. 이미지에서 금액을 못 읽었을 때
    # True 면 502 로 실패시키고, False 면 카테고리별 추정가로 채우고 이름에 "(금액 미상)"을 붙인다.
    strict_price: bool = False

    # ------------------------------------------------------------------ 상품 검색(Tavily)
    # 추천 카테고리와 가격 범위가 정해진 뒤, 실제로 살 수 있는 상품을 찾아 링크를 붙인다.
    # 검색 여부를 모델이 판단하지 않고 파이프라인이 결정론적으로 호출한다.
    # 12B 급 모델의 tool calling 신뢰성에 기대지 않고, 호출 횟수가 고정이라 지연도 예측 가능하다.
    tavily_api_key: str = ""
    tavily_enabled: bool = True
    tavily_url: str = "https://api.tavily.com/search"
    tavily_timeout_seconds: float = 15.0
    tavily_search_depth: str = "basic"  # basic | advanced (advanced 는 크레딧 2배)
    # Search 는 **결과 수와 무관하게 1회 = 1크레딧**이라 이 값을 올려도 크레딧이 늘지
    # 않습니다. 실측에서 검색 3회로 상세페이지 후보 8건을 얻었는데 그중 예산 안이
    # 0건이었습니다(/recommend 는 4건 중 0건). 후보가 모자라면 _select_by_price 가
    # "예산 밖에서 가장 가까운 것"으로 떨어집니다.
    # 비용은 크레딧이 아니라 지연입니다. search_one 이 만드는 후보가 늘면
    # enrich_prices 의 직접 조회 GET(무료)과 상품 판정 프롬프트 입력 토큰
    # (제목 하나에 약 25토큰)이 그만큼 늘어납니다.
    # 반드시 product_candidate_limit 과 함께 올릴 것. 아래 주석 참고.
    tavily_max_results: int = 12
    # 검색 스니펫의 숫자는 같은 브랜드 다른 옵션의 가격일 수 있어 믿을 수 없다.
    # Extract 로 상품 페이지 본문의 "판매가 N원" 을 읽어 실제 가격을 확정한다.
    # 유효한 URL 4개까지 1~2초면 끝나지만, 접근이 막힌 URL 이 섞이면 재시도로 길어져 제한 시간을 둔다.
    tavily_extract_url: str = "https://api.tavily.com/extract"
    tavily_extract_depth: str = "advanced"  # basic 은 국내 쇼핑몰 상당수를 못 읽는다(실측)
    # Extract 과금 단위는 **성공 URL 5개**입니다(basic 1크레딧 / advanced 2크레딧,
    # 실패 URL 은 과금 없음). 한 요청이 5개 미만이어도 한 단위로 올림되므로, 대상 수와
    # 묶음 크기를 5의 배수에 맞추지 않으면 남는 자리를 그냥 버립니다.
    #   limit 8 / batch 3 = 3+3+2 → 3요청 = 3단위 = 6크레딧(advanced)
    #   limit 5 / batch 5 = 5     → 1요청 = 1단위 = 2크레딧(advanced)
    #   limit 5 / batch 3 = 3+2   → 2요청 = 2단위 = 4크레딧(advanced)  ← 현재
    # 현재 값을 고른 이유는 tavily_extract_batch_size 주석에 있습니다. 한 묶음이 잘려도
    # 다른 묶음의 가격은 남기려고 2크레딧을 더 씁니다.
    # 최종 노출은 3개(product_suggestion_limit)이므로 확정 대상 5개면 충분합니다.
    # 대상이 많을수록 접근 안 되는 URL 이 섞일 확률이 올라가고 그만큼 느려집니다.
    tavily_extract_limit: int = 5
    # 이 값은 **임계 경로**입니다. search() 가 gather(filter_relevant, enrich_prices)
    # 로 돌리므로 늦은 쪽이 그대로 응답 시간이 됩니다.
    #
    # 실측이 둘 있고 서로 다른 배치에서 나왔습니다. 둘 다 남깁니다.
    #  (1) 과거(limit 8 / batch 3, 직접 조회 도입 전): 8초에서는 묶음이 통째로 잘려
    #      가격 미확인이 늘었습니다. 당시엔 묶음이 3개라 하나가 잘려도 2/3은 남았습니다.
    #  (2) 현재(limit 5 / batch 5, 직접 조회 도입 후): 성공한 Extract 는 두 번 모두
    #      1초 미만이었고(00:11:17→18, 00:11:39→40), 실패한 한 요청은 10초를 다 쓰고
    #      0건을 얻었습니다. /recommend 응답 20.17초 중 약 8초가 이 죽은 대기였습니다.
    #
    #  (3) 최신(limit 5 / batch 5): 4회 중 3회가 6초를 통째로 쓰고 0건을 확정했습니다.
    #      성공한 1회는 0.7초에 3/3을 확정했습니다. 성공과 실패가 시간으로 갈립니다.
    #        gift 콜드  URL 5 → 0건 확정 6.0초(타임아웃)
    #        recommend  URL 3 → 3건 확정 0.7초
    #        giftdata   URL 5 → 0건 확정 6.0초(타임아웃)
    #        gift 웜    URL 5 → 0건 확정 6.0초(타임아웃)
    #      이 6초는 그대로 응답 시간입니다. gift 콜드에서 판정은 00:50:55 에 끝났는데
    #      응답은 Extract 가 잘리는 00:50:59 에 나갔습니다.
    #
    # 6초 → 3초. 관측된 성공은 0.7초 하나뿐이라 그 4배를 남깁니다. 6초는 8.5배였고,
    # 그 차이만큼을 세 번 다 버렸습니다. 잘려도 상품은 그대로 나가며 "확인 필요"로
    # 표시됩니다(_select_by_price → _reason).
    #  (4) 4차 실측(limit 5 / batch 3): 묶음 6개 중 4개가 3초를 넘겨 잘렸습니다(67%).
    #      잘린 4개는 **모두** 카카오 선물하기를 포함했고(gift 콜드 제외 전 흐름),
    #      살아남은 묶음은 gift 웜 2건 확정·giftdata 2건 확정이었습니다.
    #      그래도 3초를 유지합니다. 근거 둘:
    #        - 4차 gift 흐름은 Extract 없이도 가격이 채워졌습니다. 콜드 후보 7건 중
    #          6건, 웜 10건 중 6건을 직접 조회가 **크레딧 0원**으로 확정했습니다.
    #          그러고도 노출 0건이었습니다. 즉 이 타임아웃은 gift 상품 0건의 원인이
    #          아닙니다(원인은 후보 자체가 예산 밖인 것 — product_search 참고).
    #        - 올리면 그 시간이 그대로 응답 시간입니다. 현재 gift 콜드 18.16초 /
    #          웜 19.45초이고 더 늘릴 여유가 없습니다.
    #      카카오만 따로 빼지도 않습니다. giftdata 에서는 카카오가 낀 묶음이
    #      2건을 확정했으므로 항상 실패하는 호스트가 아닙니다.
    tavily_extract_timeout_seconds: float = 3.0
    # Extract 는 페이지를 마크다운으로 바꾸며 HTML 안의 가격 데이터를 버립니다.
    # 그래서 상품 페이지를 직접 받아 구조화된 판매가를 먼저 시도하고, 실패한 건만
    # Extract 로 넘깁니다. 실측 커버리지는 9개 도메인 중 2곳(11번가·컬리)입니다.
    #
    # 도메인 수는 2곳이지만 **건수 기준 커버리지는 훨씬 높습니다**(4차 실측).
    #   gift 콜드  후보 7건 중 6건 확정(11번가 3 · 컬리 3), Extract 로 넘어간 건 1건
    #   gift 웜    후보 10건 중 6건 확정, Extract 로 넘어간 4건은 카카오 3 · SSG 1
    # 넘어간 4건이 정확히 "가격 마커가 없는 카카오"와 "403 인 SSG" 였습니다. 즉
    # 남은 도메인을 넓히려면 마크업이 아니라 봇 차단을 뚫어야 하는 문제입니다.
    # 실제 페이지를 받아 확인하지 않은 채 사이트별 셀렉터를 추가하지 마세요.
    # 4차 로그상 이 경로는 이미 크레딧 0원으로 대부분을 확정하고 있고, 상품 0건의
    # 원인이 아닙니다.
    product_price_fetch_enabled: bool = True
    # 검색 결과가 카테고리에 맞는 선물인지 모델이 판정합니다. 키워드 사전은
    # 부분 문자열 매칭이라 '차'가 '차량'에 걸리고 브랜드 표기는 놓칩니다.
    # 후보를 한 번에 묶어 한 번만 부르므로 지연은 3초 안쪽입니다.
    product_llm_filter_enabled: bool = True
    # 판정용 샘플링. 이 호출은 "이 상품이 이 카테고리의 답례 선물인가" 를 **판정**하는
    # 일이지 문장을 짓는 일이 아닙니다. 지정하지 않으면 API 기본값 1.0 이 적용되고,
    # 실측에서 같은 제목이 실행마다 뒤집혔습니다: "[선물] 명품 나주배 세트 5kg(8-10과)"
    # 가 28,000~42,000원 요청에서는 제외됐는데(로그: 판정 16건 중 3건 제외에 포함)
    # 8,000~12,000원 요청에서는 통과해 유일한 추천으로 나갔습니다. 같은 서버·23초 차이,
    # 판정 프롬프트는 가격을 보지 않으므로 입력 차이로 설명되지 않는 출력 변동입니다.
    #
    # 그래서 greedy(0.0) 로 둡니다. vision_temperature 가 0.0 인 것과 같은 이유이고,
    # bedrock_temperature(0.4) 를 쓰지 않는 이유는 그 값이 suggested_message 가 한
    # 문장 틀로 굳지 않게 하려는 것이라 여기에는 해당 사항이 없기 때문입니다. 출력은
    # 번호 배열 두 개라 변주가 이득이 되는 자리가 없습니다.
    #
    # 주의: Claude 는 temperature 와 top_p 를 동시에 받지 않습니다. temperature 만 보냅니다.
    product_filter_temperature: float = 0.0
    # 이미지에 금액이 없을 때 상품명으로 실제 판매가를 검색해 채웁니다.
    # 카테고리 추정가는 브랜드를 모릅니다(TWG Tea 를 10,000원으로 추정, 실제 3~7만원).
    product_price_lookup_enabled: bool = True
    product_price_lookup_limit: int = 5
    product_llm_filter_max_tokens: int = 2_000
    product_price_fetch_timeout_seconds: float = 6.0
    # 5 → 3. 이전 주석은 과금 단위(성공 URL 5개)에 맞춰 5를 골랐고, "묶음이 하나뿐이라
    # 다른 묶음까지 번지는 경우는 사라진다"고 적었습니다. 실측이 그 반대를 보여 줬습니다.
    # 묶음이 하나라는 것은 그 하나가 잘리면 **전량 미확인**이라는 뜻이고, 4회 중 3회가
    # 정확히 그랬습니다(URL 5개 → 0건 확정). 관측된 유일한 성공은 URL 3개 묶음의
    # 0.7초 3/3 확정입니다.
    #
    # 3으로 나누면 대상 5개가 [3, 2] 두 묶음이 되어 asyncio.gather 로 동시에 나갑니다.
    # 느린 URL 이 낀 묶음만 잘리고 나머지 묶음의 가격은 살아남습니다.
    #
    # 대가는 크레딧입니다. 과금은 요청당 성공 URL 5개 단위 올림이라 요청이 1개에서
    # 2개로 늘면 advanced 기준 2크레딧 → 4크레딧입니다. 가격을 하나도 확인하지 못한
    # 채 6초를 버리는 것보다 2크레딧이 낫다고 봤습니다. price_verified=false 가 늘면
    # 미확인 가격이 "예산 안" 판정의 근거가 되므로, 확인율은 지연만큼 중요합니다.
    tavily_extract_batch_size: int = 3
    # 신뢰할 수 있는 국내 거래 플랫폼만 검색한다. 블로그·카페의 광고성 글을 걸러 내기 위함이다.
    # 주의: Tavily 는 country 파라미터를 include_domains 와 함께 쓰면 결과가 0건이 된다(실측).
    product_search_domains: list[str] = [
        "coupang.com",
        "gift.kakao.com",
        "shopping.naver.com",
        "ssg.com",
        "gmarket.co.kr",
        "11st.co.kr",
        "lotteon.com",
        "kurly.com",
        "oliveyoung.co.kr",
    ]
    # 가격을 확정하기 전에 모아 두는 후보 수. 최종 개수보다 넉넉해야 예산에 맞는 것이 남는다.
    #
    # tavily_max_results 와 **함께** 움직여야 합니다. _interleave 는 카테고리별 결과를
    # 검색 관련도 순서 그대로 번갈아 뽑아 이 수에서 자르고, 예산으로 고르는
    # _select_by_price 는 그 뒤에 옵니다. 즉 이 값이 낮으면 가격을 아는 상태에서도
    # 관련도 상위 N건만 남기고 예산에 맞는 후보를 잘라 버립니다.
    # (product_search.py: search() → _interleave(..., product_candidate_limit)
    #  → _select_by_price(candidates, low, high, limit))
    product_candidate_limit: int = 12
    product_suggestion_limit: int = 3
    # 예산 안 후보가 상한에 못 미칠 때, 예산 밖 상품을 "가까운 것"으로 보충하며
    # 허용하는 이탈 폭입니다. 경계값 기준이라 8,000~12,000원이면 6,800~13,800원입니다.
    #
    # 직전 값은 "절반~두 배"(-50%~+100%)였고 실측에서 무너졌습니다. 노출 10건 중
    # 예산 안은 3건(30%)뿐이었고, 이탈률은 -32%, -12%, +58%, +59%, +81%, +100% 였습니다.
    # 사용자가 18,000~27,000원을 **직접 지정**한 요청에 49,000원(+81%)이 나갔고,
    # 같은 응답의 product_basis 는 "0개가 18,000원 ~ 27,000원 안에 듭니다" 였습니다.
    # 화면에 숫자 두 개가 나란히 보이는 순간 들통나는 모순입니다.
    #
    # 0.15 를 고른 근거는 가격대를 만드는 규칙 자체입니다. 추천 범위는 받은 금액의
    # 80~120%(recommendation_policy.price_range)라 중심값 기준 반폭이 20% 입니다.
    # 보충 폭이 그보다 넓으면 사용자에게 알려 준 범위를 소리 없이 두 배로 넓히는 셈이라,
    # 반폭보다 좁은 15% 로 둡니다. 이 값이면 실측 이탈 6종 중 -12% 하나만 통과하고
    # 나머지(-32%, +58%, +59%, +81%, +100%)는 전부 떨어집니다.
    #
    # 채울 것이 없으면 적게 나갑니다. 1건이라도 예산 안인 편이 3건 예산 밖보다 낫습니다.
    product_price_slack_ratio: float = 0.15

    # ------------------------------------------------------------------ 캘린더(MCP)
    calendar_mcp_url: str = "http://localhost:8300/mcp"
    calendar_mcp_timeout_seconds: float = 30.0
    # 데모용 단일 계정 토큰. 비어 있으면 실제 등록 없이 초안 JSON 만 만든다.
    google_access_token: str = ""
    # 기본값 false: 캘린더 등록은 사용자가 확인 화면에서 승인한 뒤 /confirm 에서만 일어난다.
    # 승인 UI 가 없는 개발 단계에서 흐름을 확인할 때만 true 로 둔다.
    calendar_auto_register: bool = False
    google_calendar_id: str = "primary"
    calendar_default_lead_days: int = 30  # target_date 가 없을 때 답례일까지의 기본 간격
    notification_lead_days: int = 7  # 답례일 며칠 전에 알릴지
    default_timezone: str = "Asia/Seoul"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """프로세스 동안 재사용할 설정 객체를 최초 한 번만 생성합니다."""
    return Settings()


# 다른 모듈은 이 singleton을 가져와 동일한 설정을 사용합니다.
settings = get_settings()
