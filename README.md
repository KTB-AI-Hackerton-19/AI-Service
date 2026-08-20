# Giftie AI Service

Giftie의 FastAPI 기반 AI 오케스트레이터입니다. Spring Boot 백엔드에서 선물데이터 또는 S3 이미지
주소를 받아 네 작업을 비동기로 실행한 뒤 하나의 JSON으로 반환합니다.

1. 선물 기록 저장 데이터 준비
2. Google Calendar 등록 (준비 단계는 초안까지, 실제 등록은 `/confirm`)
3. 알림 예약 데이터 준비
4. 답례 상품 추천과 감사 메시지 준비

추천과 이미지 분석은 같은 모델 설정을 공유합니다. 기본 실행 경로는 Amazon Bedrock의
Claude Sonnet 4.6이며 GPU나 로컬 모델 적재가 필요 없습니다. 자체 GPU를 쓰는 경우에는 같은
vLLM 서버의 같은 모델(Gemma4-12B-QAT + MTP)을 대신 씁니다. 어느 쪽이든 모델을 두 벌 올리지
않으므로 두 종류의 요청이 동시에 들어와도 한 엔진에서 함께 처리됩니다.

`MODEL_BACKEND=mock` 이면 네트워크를 타지 않고 고정된 결과로 흐름만 확인할 수 있습니다.

## API 개요

| Method | Path | 입력 | 처리 |
|---|---|---|---|
| POST | `/api/v1/agent/from-gift-data` | 구조화된 선물데이터 | 네 작업을 바로 실행 |
| POST | `/api/v1/agent/from-image` | S3 이미지 URL | 이미지 분석 후 네 작업 실행 |
| POST | `/api/v1/agent/confirm` | 사용자 수정본 | 확정하고 캘린더에 실제 등록 |
| POST | `/api/v1/agent/recommend` | 나이·가격대·카테고리·성별 | 추천만 단독 실행 |

앞의 두 API는 캘린더에 등록하지 않고 초안까지만 만듭니다. 잘못 추출된 일정이 사용자 캘린더에
바로 박히면 되돌리기 어렵기 때문입니다. 실제 등록은 사용자가 확인 화면에서 검토·수정한 뒤
`/confirm`에서 일어납니다.

```text
[준비]  POST /from-image  ->  네 작업 동시 실행  ->  응답(requires_confirmation=true)
                                                          |
                                              사용자가 확인 화면에서 검토·수정
                                              (금액 정정, 저장할 건 선택, 일정 변경)
                                                          |
[확정]  POST /confirm  ->  기록·알림 재계산 + Google Calendar 등록  ->  응답
```

AI 서비스는 상태를 보관하지 않습니다. 백엔드가 준비 응답을 들고 있다가 사용자 수정본과 함께
`/confirm`으로 되돌려주면 됩니다. 세션을 AI 쪽에 두면 재시작이나 인스턴스 증설에서 그대로
깨지는데, 확정에 필요한 데이터는 어차피 백엔드가 DB에 저장할 것들입니다.

## 처리 흐름

![Giftie AI Service 전체 아키텍처](docs/images/giftie-ai-architecture.png)

```mermaid
flowchart LR
    Client[프론트엔드] -->|사용자 요청| Backend[Spring Boot 백엔드]

    Backend -->|POST from-gift-data| GiftAPI[선물데이터 API]
    Backend -->|POST from-image| ImageAPI[이미지 API]

    subgraph Giftie[Giftie FastAPI]
        direction TB

        GiftAPI --> CommonData[공통 GiftData]
        ImageAPI --> ImageAnalyzer[이미지 분석 서비스<br/>Bedrock Claude / vLLM Gemma4-12B-QAT]
        ImageAnalyzer --> CommonData

        CommonData --> Orchestrator[GiftAgentService<br/>오케스트레이터]

        Orchestrator -->|비동기 실행| GiftTask[선물 기록 JSON 준비]
        Orchestrator -->|비동기 실행| CalendarTask[캘린더 등록<br/>Google Calendar MCP]
        Orchestrator -->|비동기 실행| NotificationTask[알림 예약 JSON 준비]
        Orchestrator -->|비동기 실행| RecommendationTask[추천 상품과 메시지 준비]

        RecommendationTask --> QwenService[QwenRecommendationService]
        QwenService --> Prompt[프롬프트 생성]
        QwenService --> Model[공용 모델 엔진<br/>Bedrock Claude Sonnet 4.6 (기본) /<br/>vLLM Gemma4-12B-QAT + MTP]
        Model --> Parser[모델 JSON 파싱]
        Parser --> Policy[가격과 카테고리 안전 정책]
        Policy --> ProductSearch[Tavily 실상품 검색<br/>카테고리별 병렬 실행]

        GiftTask --> Merger[결과 병합]
        CalendarTask --> Merger
        NotificationTask --> Merger
        ProductSearch --> Merger
    end

    Merger -->|통합 JSON| Backend

    classDef actual fill:#d1e7dd,stroke:#198754,color:#0f5132;
    classDef external fill:#cfe2ff,stroke:#0d6efd,color:#084298;

    class RecommendationTask,QwenService,Prompt,Model,Parser,Policy actual;
    class ImageAnalyzer,GiftTask,CalendarTask,NotificationTask actual;
    class Client,Backend external;
```

네 작업은 공통 `GiftData`가 준비되는 즉시 `asyncio.gather(..., return_exceptions=True)`로 동시에
시작합니다. 추천 작업 안에서는 모델이 검색어와 카테고리를 만든 다음 카테고리별 Tavily 검색을
다시 병렬로 실행합니다. 네 작업 중 하나가 실패해도 나머지 결과는 유지하며 실패한 항목만
`ERROR` 상태로 반환합니다.

### 제한 시간

| 단계 | 설정 | 기본값 |
|---|---|---|
| `/from-image` 의 이미지 분석 | `IMAGE_ANALYSIS_TIMEOUT_SECONDS` | 45초 |
| 네 후속 작업과 `/recommend` | `TASK_TIMEOUT_SECONDS` | 30초 |

두 단계는 직렬이므로 서버가 스스로 끊는 최악 지연은 두 값의 합인 **75초**입니다. 백엔드 HTTP
타임아웃을 90초로 잡으면 백엔드가 먼저 끊는 일이 없습니다. 넘기면 504(`UPSTREAM_TIMEOUT`)입니다.

## 프로젝트 구조

```text
AI-Service/
├── app/
│   ├── core/
│   │   ├── config.py             # 환경변수 및 모델 설정
│   │   ├── security.py           # X-API-KEY 검증
│   │   ├── errors.py             # ErrorCode enum, ApiErrorResponse, GiftieHTTPException
│   │   └── exception_handlers.py # 모든 예외를 공통 오류 JSON으로 변환
│   ├── routers/
│   │   └── agent.py              # 공개 API 네 개
│   ├── schemas/
│   │   ├── agent.py              # API 요청·응답 타입 (공개 계약)
│   │   ├── recommendation.py     # 추천 입력·출력 타입
│   │   └── vision.py             # 이미지 추출 내부 타입 (HTTP 로 나가지 않음)
│   ├── services/
│   │   ├── gift_agent_service.py # 실행 순서·타임아웃·결과 병합 + 추천 실행 대상 분기
│   │   ├── qwen_service.py       # 추천 추론 (bedrock / vllm / mlx / transformers / mock)
│   │   ├── bedrock_client.py     # Bedrock 클라이언트·인증·구조화 출력·오류 해석 (추천·비전 공용)
│   │   ├── prompt.py             # 추천 프롬프트 + JSON 스키마
│   │   ├── model_response_parser.py # 모델 JSON 응답 파싱
│   │   ├── recommendation_policy.py # 가격·카테고리 안전 정책
│   │   ├── price_policy.py       # 받은 가격 -> 답례 가격 범위 계산
│   │   ├── recommendation_rationale.py # 추천 근거(rationale) 조립
│   │   ├── product_search.py     # Tavily 상품 검색·판매가 확인
│   │   ├── product_filter.py     # 검색 결과가 카테고리에 맞는 선물인지 모델 판정
│   │   ├── image_loader.py       # presigned URL 다운로드·검증·EXIF 회전·리사이즈
│   │   ├── vision_prompt.py      # 이미지 추출 프롬프트 + JSON 스키마
│   │   ├── vlm_service.py        # Bedrock/vLLM 이미지 추출 호출
│   │   ├── vision_response_parser.py # VLM 출력 정규화 (날짜·금액·중복)
│   │   ├── gift_data_policy.py   # 추출 결과 -> GiftData 안전 변환
│   │   ├── reciprocity_schedule.py # 답례일·준비일·알림 시각 규칙
│   │   ├── record_summary.py     # 여러 건을 사람이 읽는 문구로 요약
│   │   ├── confirmation_service.py # 사용자 승인 이후의 확정 처리
│   │   ├── calendar_mcp_client.py # Google Calendar MCP 클라이언트
│   │   ├── clock.py              # 서비스 타임존 기준 벽시계 시각
│   │   └── tasks/
│   │       ├── image_analysis.py # 이미지 -> 선물데이터
│   │       ├── gift_record.py    # 선물 기록 JSON
│   │       ├── calendar.py       # Google MCP 캘린더
│   │       ├── notification.py   # 알림 예약 JSON
│   │       └── recommendation.py # 추천·메시지
│   └── main.py                   # FastAPI 진입점
├── mcp_servers/
│   └── google_calendar.py        # 자체 Google Calendar MCP 서버 (별도 프로세스)
├── scripts/
│   ├── export_openapi.py         # docs/openapi.json 생성·검증
│   ├── verify_bedrock.py         # Bedrock 백엔드 실호출 검증
│   └── verify_calendar.py        # Google Calendar 실연동 검증
├── docs/
│   ├── openapi.json              # 계약 스펙 (Java 클라이언트 생성용)
│   └── api-examples.http         # 실행 가능한 요청 예시
└── tests/
```

## 환경 설정

`.env.example`을 복사해 사용합니다. `.env`에는 비밀값이 들어가므로 Git에 커밋하지 않습니다.
아래 표는 자주 만지는 값만 담았습니다. 전체 목록과 각 기본값을 그렇게 고른 이유는
`app/core/config.py` 에 주석과 함께 있습니다.

```bash
cp .env.example .env
```

### 공통

| 변수 | 설명 | 로컬 예시 |
|---|---|---|
| `API_KEY` | Spring Boot와 공유하는 내부 API 키 | `local-development-key` |
| `MODEL_BACKEND` | `bedrock`, `mock`, `vllm`, `mlx`, `transformers` | `bedrock` |
| `IMAGE_ANALYSIS_TIMEOUT_SECONDS` | `/from-image` 의 이미지 분석 단계 제한 시간(초) | `45` |
| `TASK_TIMEOUT_SECONDS` | 네 후속 작업과 `/recommend` 의 제한 시간(초) | `30` |
| `IMAGE_MAX_EDGE` | 이미지 장변 리사이즈 상한(px) | `1280` |
| `IMAGE_MAX_BYTES` | 허용 이미지 최대 크기 | `12582912` |
| `STRICT_PRICE` | 금액을 못 읽었을 때 `true` 면 502, `false` 면 `gift_price` 를 비움 | `false` |

### Bedrock

| 변수 | 설명 | 로컬 예시 |
|---|---|---|
| `BEDROCK_API_STYLE` | 호출 방식, `invoke` 또는 `mantle` | `invoke` |
| `BEDROCK_REGION` | 호출할 AWS 리전 | `us-east-1` |
| `BEDROCK_MODEL_ID` | 추천에 사용할 Claude 모델 ID | `global.anthropic.claude-sonnet-4-6` |
| `BEDROCK_VISION_MODEL_ID` | 이미지 분석에 사용할 Claude 모델 ID. 지금은 추천과 같은 모델이며, 필요할 때 갈라놓을 수 있게 값만 분리해 둡니다 | `global.anthropic.claude-sonnet-4-6` |
| `BEDROCK_MAX_TOKENS` | 추천·이미지 JSON 최대 출력 토큰 | `2048` |
| `BEDROCK_TEMPERATURE` | Bedrock 전용 샘플링. 카테고리 개수·메시지 길이를 프롬프트로 요구하므로 형식 안정성을 우선합니다 | `0.4` |
| `BEDROCK_API_KEY` | Bearer API 키. IAM 방식이면 비움 | (비움) |
| `BEDROCK_AWS_PROFILE` | 로컬 AWS 프로필. API 키 방식이면 비움 | (비움) |
| `RECOMMENDATION_SPLIT_CALLS` | 추천 생성을 카테고리 / 이유·요약 / 감사 메시지 세 호출로 나눕니다. 상품 검색이 카테고리만 나오면 출발할 수 있어 종단 지연이 줄어듭니다(실측 중앙값 13.9초 → 7.6초). Bedrock 경로에서만 동작합니다 | `false` |
| `RECOMMENDATION_LANGGRAPH` | 추천 오케스트레이션을 LangGraph 상태 그래프로 실행합니다(실험 경로). 분할 경로와 같은 세 호출·같은 프롬프트·같은 정규화라 출력은 동일하고(실호출 A/B 중앙값 차이 ±0.3초 이내), 상품 0건일 때 남은 씨앗으로 한 번 재검색하는 자기 보정이 더해집니다. Bedrock + langgraph 설치 환경에서만 동작하며 켜면 `RECOMMENDATION_SPLIT_CALLS` 보다 우선합니다. 구조와 실측은 `app/graph/recommendation_graph.py` 와 `scripts/benchmark_graph.py` 참고 | `false` |
| `LANGGRAPH_SEARCH_RETRY` | LangGraph 경로에서 상품 0건일 때의 재검색. 정상 경로 지연은 그대로이고, 0건 경로에서만 검색 한 바퀴(6~9초)와 Tavily 크레딧(최대 3회)이 추가됩니다 | `true` |

`BEDROCK_API_KEY`와 `BEDROCK_AWS_PROFILE`은 함께 설정하지 않습니다. EC2에서는 키를 파일에
넣기보다 IAM Role을 연결하고 두 값을 모두 비우는 방식을 권장합니다.

`BEDROCK_API_STYLE` 은 계정마다 열려 있는 경로가 달라 존재합니다. `invoke` 는 레거시
`bedrock-runtime`(추론 프로파일 ID), `mantle` 은 Messages 엔드포인트(`anthropic.` 접두사 ID)
입니다. **모든 모델이 403 이면 이 값을 가장 먼저 의심하세요.**

### 상품 검색 (Tavily)

키가 없거나 검색이 실패해도 API 전체를 실패시키지 않고 `products: []` 와 기존
`product_examples` 를 반환합니다.

| 변수 | 설명 | 로컬 예시 |
|---|---|---|
| `TAVILY_ENABLED` | 실제 상품 검색 활성화 여부 | `true` |
| `TAVILY_API_KEY` | Tavily API 키 | (비움) |
| `TAVILY_TIMEOUT_SECONDS` | 검색 제한 시간(초) | `15` |
| `TAVILY_MAX_RESULTS` | 검색 1회가 가져올 결과 수. 결과 수와 무관하게 1회 = 1크레딧 | `12` |
| `TAVILY_EXTRACT_LIMIT` | 판매가 확정에 Extract 를 쓸 URL 상한 | `5` |
| `TAVILY_EXTRACT_BATCH_SIZE` | Extract 한 요청에 담을 URL 수 | `3` |
| `TAVILY_EXTRACT_TIMEOUT_SECONDS` | Extract 묶음 제한 시간(초) | `3` |
| `PRODUCT_CANDIDATE_LIMIT` | 가격으로 고르기 전에 모아 두는 후보 수. `TAVILY_MAX_RESULTS` 와 함께 올립니다 | `12` |
| `PRODUCT_PRICE_FETCH_ENABLED` | 상품 페이지를 직접 받아 판매가를 먼저 읽을지 | `true` |
| `PRODUCT_PRICE_LOOKUP_ENABLED` | 이미지에 금액이 없을 때 상품명으로 판매가를 검색할지 | `true` |
| `PRODUCT_LLM_FILTER_ENABLED` | 카테고리 적합성 판정을 모델에 맡길지. 끄면 키워드 사전 | `true` |
| `PRODUCT_FILTER_TEMPERATURE` | 상품 판정 샘플링. 판정은 창작이 아니므로 greedy 고정 | `0.0` |
| `PRODUCT_PRICE_LOOKUP_LIMIT` | 상품 페이지를 열어 판매가를 확인할 후보 수. 여기 안 든 후보는 가격 미상이라 노출되지 않습니다 | `5` |
| `PRODUCT_SUGGESTION_LIMIT` | 최종 노출 상품 수 상한 | `3` |
| `PRODUCT_PRICE_SLACK_RATIO` | 예산 안 상품이 모자랄 때 보충을 허용하는 이탈 폭 | `0.15` |

### 캘린더·알림

| 변수 | 설명 | 로컬 예시 |
|---|---|---|
| `CALENDAR_MCP_URL` | Google Calendar MCP 서버 주소 | `http://localhost:8300/mcp` |
| `GOOGLE_ACCESS_TOKEN` | 서버 기본 OAuth access token. `/confirm` 요청에 토큰이 없을 때만 사용 | (비움) |
| `CALENDAR_AUTO_REGISTER` | `true`면 준비 단계에서 바로 등록. 승인 UI 없는 개발 단계 전용, 운영에서는 `false` | `false` |
| `GOOGLE_CALENDAR_ID` | 대상 캘린더 | `primary` |
| `CALENDAR_DEFAULT_LEAD_DAYS` | `target_date` 가 없을 때 답례일까지의 기본 간격(일) | `30` |
| `NOTIFICATION_LEAD_DAYS` | 답례일 며칠 전에 알릴지 | `7` |

### 로컬 모델 (vLLM / MLX / Transformers)

| 변수 | 설명 | 로컬 예시 |
|---|---|---|
| `VLLM_BASE_URL` | 공용 vLLM 서버 주소. FastAPI 가 8000 을 쓰므로 8001 로 띄웁니다 | `http://localhost:8001` |
| `VLLM_MODEL` | vLLM `--served-model-name` 값 | `gemma4-12b-qat` |
| `VLLM_TIMEOUT_SECONDS` | vLLM 호출 제한 시간 | `90` |
| `LOCAL_MODEL_ID` | Apple Silicon MLX 모델 | `mlx-community/Qwen3-4B-Instruct-2507-4bit` |
| `MODEL_ID` | GPU Transformers 모델 | `Qwen/Qwen3-4B` |
| `PRELOAD_MODEL` | 서버 시작 시 모델 사전 적재 여부 | `false` |
| `MAX_NEW_TOKENS` | 모델 최대 생성 토큰 | `600` |
| `VISION_MAX_NEW_TOKENS` | 이미지 추출 최대 생성 토큰 | `900` |

## 실행

### Bedrock (권장)

GPU나 로컬 모델 다운로드 없이 추천과 이미지 분석을 모두 실제 실행합니다.

```env
MODEL_BACKEND=bedrock
BEDROCK_API_STYLE=invoke
BEDROCK_REGION=us-east-1
BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-4-6
BEDROCK_VISION_MODEL_ID=global.anthropic.claude-sonnet-4-6
AWS_BEARER_TOKEN_BEDROCK=발급받은-키
```

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Swagger: http://127.0.0.1:8000/docs
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json

설정이 실제로 동작하는지는 앱을 띄우지 않고도 확인할 수 있습니다. 앱과 같은 코드 경로를
통과하므로 여기서 통과하면 API 도 통과합니다.

```bash
python scripts/verify_bedrock.py              # 전체
python scripts/verify_bedrock.py --preflight  # 연결·권한만
```

### Mock

모델 다운로드 없이 API 흐름만 테스트할 때 씁니다. 요청·응답 형태는 완전히 동일하고 AI 응답만
고정값입니다.

```bash
MODEL_BACKEND=mock uvicorn app.main:app --reload --port 8000
```

### vLLM

추천과 이미지 분석이 함께 쓰는 서버입니다. FastAPI 가 8000 을 쓰므로 8001 로 띄웁니다.

```bash
docker run --rm --gpus all -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface --ipc=host \
  vllm/vllm-openai:v0.27.1-x86_64-cu129 \
  --model google/gemma-4-12B-it-qat-w4a16-ct \
  --served-model-name gemma4-12b-qat \
  --max-model-len 16384 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 2}'
```

```env
MODEL_BACKEND=vllm
VLLM_BASE_URL=http://localhost:8001
VLLM_MODEL=gemma4-12b-qat
```

MTP(Multi-Token Prediction)를 켜더라도 OpenAI 호환 API 는 그대로이므로 이 서비스 코드는
바뀌지 않습니다. 서버 기동 플래그만 달라집니다.

### Apple Silicon MLX

```bash
pip install -r requirements-mac.txt
MODEL_BACKEND=mlx uvicorn app.main:app --port 8000
```

`mlx` 와 `transformers` 는 추천용 텍스트 모델만 지원하므로 `/from-image` 의 이미지 추출은
mock 으로 동작합니다. 실제 이미지 분석은 Bedrock 또는 vLLM 을 쓰세요.

### Google Calendar MCP 서버

캘린더 등록은 `mcp_servers/google_calendar.py` 가 노출하는 MCP 툴을 통해 이뤄집니다.
별도 프로세스로 띄웁니다.

```bash
python -m mcp_servers.google_calendar     # streamable-http, :8300/mcp
```

노출하는 툴은 `create_event`, `update_event`, `get_event`, `delete_event`, `list_events`
다섯 가지이고 모두 사용자별 `access_token` 을 인자로 받습니다. 필요한 OAuth 스코프는
`https://www.googleapis.com/auth/calendar.events` 입니다. 토큰은 로그에 남기지 않습니다.

공개된 Google Calendar MCP 서버 대부분은 서버 자신이 OAuth 플로우를 돌리고 토큰 파일 하나로
단일 계정만 다룹니다. Giftie 는 Spring Security 가 들고 있는 사용자별 토큰을 써야 해서 직접
만들었습니다.

실제 Google 계정으로 확인하려면 `.env` 에 `GOOGLE_ACCESS_TOKEN` 을 넣고 검증 스크립트를
실행합니다. 생성 → 조회 → 알림 확인 → 승인 후 등록 → 삭제 순으로 돌기 때문에 캘린더에 흔적이
남지 않습니다.

```bash
python scripts/verify_calendar.py
```

MCP 서버가 죽어 있어도 캘린더 작업은 `ERROR` 가 아니라 초안과 `registerError` 를 함께
돌려주므로 나머지 세 작업 결과는 그대로 유지됩니다.

## 인증

모든 API 는 `X-API-KEY` 헤더가 필요합니다.

```http
X-API-KEY: local-development-key
```

키가 없거나 틀리면 `401 Unauthorized` 입니다. 운영에서는 프론트엔드가 아니라 Spring Boot만
이 키를 보유하고 FastAPI를 호출해야 합니다.

## 오류 응답

인증 실패, 요청 검증 실패, 외부 서비스 실패, 내부 오류는 모두 동일한 JSON 구조로 반환합니다.
HTTP 상태 코드는 그대로 유지하고 `error_code` 로 프로그램에서 오류 종류를 구분합니다.
`detail` 은 항상 한글이고 `error_code` 는 안정적인 영문 코드입니다.

```json
{
  "status": "ERROR",
  "error_code": "INVALID_API_KEY",
  "detail": "유효하지 않은 AI 서비스 API 키입니다."
}
```

요청 필드 검증 오류에는 문제가 된 필드의 `errors` 배열이 추가됩니다.

| error_code | 의미 |
|---|---|
| `INVALID_API_KEY` | `X-API-KEY`가 누락됐거나 서버 설정과 다름 |
| `VALIDATION_ERROR` | 요청 JSON 필드의 형식·범위가 잘못됨 |
| `GIFT_INPUT_INVALID` | 입력에서 유효한 선물데이터를 만들 수 없음 |
| `IMAGE_ANALYSIS_FAILED` | 이미지 다운로드 또는 이미지 분석 실패 |
| `RECOMMENDATION_FAILED` | 추천·메시지 생성 실패 |
| `CONFIRMATION_FAILED` | 사용자 확정 및 후속 처리 실패 |
| `AGENT_EXECUTION_FAILED` | 에이전트 전체 실행 중 예상하지 못한 오류 |
| `UPSTREAM_TIMEOUT` | 제한 시간 초과 |
| `UPSTREAM_SERVICE_ERROR` | 별도로 분류되지 않은 외부 서비스 오류 |
| `INTERNAL_SERVER_ERROR` | 처리되지 않은 서버 내부 오류 |

전체 목록은 `app/core/errors.py` 의 `ErrorCode` enum 과 Swagger 의 `ErrorCode` 스키마에
있습니다.

## API 1: 선물데이터 직접 전달

```bash
curl -X POST http://127.0.0.1:8000/api/v1/agent/from-gift-data \
  -H 'Content-Type: application/json' \
  -H 'X-API-KEY: local-development-key' \
  -d '{
    "gift_data": {
      "gift_name": "스타벅스 케이크",
      "gift_price": 35000,
      "age": 29,
      "gender": "female",
      "person_name": "김민수",
      "relationship": "친구",
      "received_at": "2026-08-19",
      "target_date": "2026-09-10"
    }
  }'
```

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `gift_name` | string | O | 받은 선물 이름, 1~200자 |
| `gift_price` | integer/null | X | 받은 선물 가격, 1~100,000,000원. 모르면 생략하거나 `null` |
| `age` | integer/null | X | 상대방 나이, 0~120 |
| `gender` | string/null | X | `male` 또는 `female`. 모르면 생략/null |
| `person_name` | string/null | X | 상대방 이름 |
| `relationship` | string/null | X | 상대방과의 관계 |
| `received_at` | date/null | X | 받은 날짜 |
| `target_date` | date/null | X | 답례 예정일 |

날짜는 정상적인 `YYYY-MM-DD` 만 사용합니다. `""`, `null`, 형식이 잘못된 문자열은 오류로 처리하지
않고 모두 미입력(`null`)으로 정규화합니다. 성별도 생략·`""`·`null`·`unknown` 이면 미입력으로
처리하며 값이 있을 때만 나이와 함께 추천에 반영합니다. `target_date` 가 없으면 캘린더는 오늘부터
30일 뒤, 알림은 그 날짜의 7일 전 오전 10시를 사용합니다.

### 여러 건이 들어 있는 입력

이미지 한 장에 여러 건이 들어 있는 경우도 다룹니다. 계좌 거래내역 5건, 선물함 목록 4건,
영수증 3품목 같은 입력입니다. `GiftData` 의 기존 평면 필드는 대표 1건(받은 금액이 가장 큰 건)을
그대로 담고 전체는 `records` 배열에 들어갑니다. 기존 필드는 손대지 않았으므로 이를 모르는 코드도
그대로 동작합니다.

```json
{
  "gift_name": "축의금", "gift_price": 200000, "person_name": "최은비",
  "records": [
    {"record_id": "r0", "person_name": "김도윤", "price": 100000, "direction": "received", "selected": true},
    {"record_id": "r1", "person_name": "박서준", "price":  50000, "direction": "received", "selected": true},
    {"record_id": "r2", "person_name": "최은비", "price": 200000, "direction": "received", "selected": true},
    {"record_id": "r3", "person_name": "카카오페이", "price": 38900, "direction": "sent", "selected": true}
  ],
  "recordCount": 4, "receivedCount": 3, "totalAmount": 350000,
  "summary": "김도윤님 외 2명에게 받은 축의금 (총 350,000원)"
}
```

- `recordCount` 는 저장할 기록 수, `receivedCount` 는 답례 대상 수입니다. 거래내역의 출금 건은
  기록으로는 남기되 답례 대상과 금액 합계에서는 빠집니다.
- `selected` 를 `false` 로 바꿔 `/confirm` 에 보내면 그 건은 저장·합계·명단에서 제외됩니다.
- 캘린더 일정은 건마다 만들지 않고 하나로 묶습니다. 대상자 명단은 일정 설명에 담습니다.
- 모델 신뢰도가 낮거나 이름·날짜·금액을 읽지 못한 항목은 `needs_review: true` 와
  `review_reasons` 가 붙어 나옵니다. 확인 화면에서 강조해 주세요.

### 금액을 읽을 수 없는 경우

청첩장처럼 금액이 없는 이미지도 502로 실패시키지 않습니다. 대신 **값을 지어내지도 않습니다.**

1. 상품명과 브랜드로 실제 판매가를 검색해 채웁니다. 여러 용량·구성이 섞이므로 찾은 가격의
   **중앙값**을 쓰고 `price_basis` 를 `searched` 로 표시합니다.
2. 검색으로도 못 찾으면 `gift_price` 를 `null`, `price_basis` 를 `unknown` 으로 둡니다.
   이때 추천만 `SKIPPED` 가 되고 나머지 세 작업은 정상 진행됩니다.
3. `STRICT_PRICE=true` 로 두면 비우는 대신 502를 반환합니다.

카테고리로 추정하지 않습니다. 브랜드를 모르는 추정가는 실제와 몇 배씩 어긋나는데 사용자는 그
값을 사실로 받아들입니다. `gift_name` 에 `(금액 미상)` 같은 표시도 붙이지 않습니다.

## API 2: 이미지 전달

```bash
curl -X POST http://127.0.0.1:8000/api/v1/agent/from-image \
  -H 'Content-Type: application/json' \
  -H 'X-API-KEY: local-development-key' \
  -d '{
    "image_url": "https://example-bucket.s3.amazonaws.com/gift.png",
    "category": "gift"
  }'
```

`category` 는 업로드 화면에서 사용자가 고른 값입니다. 선택 사항이며 보내지 않으면 이미지 분석
결과로 판단하므로 기존 연동은 고치지 않아도 됩니다.

| 값 | 동작 |
|---|---|
| `gift` | 답례 선물 추천을 만듭니다 |
| `occasion` | 추천을 만들지 않고 `recommend_gift_info.status` 를 `SKIPPED` 로 돌려줍니다 |
| 생략 | 이미지에서 읽은 기록 종류로 판단합니다 |

사용자가 고른 `category` 가 모델의 이미지 분류보다 우선합니다.

## API 3: 확정

```bash
curl -X POST http://127.0.0.1:8000/api/v1/agent/confirm \
  -H 'Content-Type: application/json' \
  -H 'X-API-KEY: local-development-key' \
  -d '{ "workflow_id": "9f1c...", "gift_data": { ... }, "google_access_token": "ya29...." }'
```

| 필드 | 필수 | 설명 |
|---|---|---|
| `workflow_id` | O | 준비 응답의 값을 그대로 |
| `gift_data` | O | 사용자가 수정한 기록. `records[].selected` 로 저장할 건을 고릅니다 |
| `calendar` | X | 사용자가 수정한 일정. **생략하면 수정된 `gift_data` 로 다시 계산합니다** |
| `approved` | X | `false` 면 아무것도 등록하지 않습니다 (기본 `true`) |
| `register_calendar` | X | `false` 면 초안만 확정하고 등록은 건너뜁니다 (기본 `true`) |
| `google_access_token` | X | 사용자 OAuth access token. 없으면 서버 설정값 사용 |

응답의 `calendar_info.payload` 에 `registered: true`, `eventId`, `htmlLink` 가 채워집니다.
등록에 실패해도 HTTP는 `200` 이며 `registered: false` 와 `registerError` 가 함께 옵니다.
캘린더가 막혔다고 기록과 알림까지 잃을 이유는 없습니다.

## API 4: 추천 단독

```bash
curl -X POST http://127.0.0.1:8000/api/v1/agent/recommend \
  -H 'Content-Type: application/json' \
  -H 'X-API-KEY: local-development-key' \
  -d '{ "gift_name": "스타벅스 케이크", "gift_price": 35000, "age": 29, "relationship": "직장 동료" }'
```

`gift_price` 나 `budget_min`/`budget_max` 중 하나가 반드시 있어야 하며 셋 다 없으면 422 입니다.
답례 가격대는 받은 금액의 **80~120%** 로 정해지므로 기준이 없으면 추천이 성립하지 않습니다.
`interests`, `dislikes`, `categories`, `event`, `gender`, `person_name` 도 선택으로 받습니다.
`categories` 를 주면 그 안에서만 고릅니다.

## 응답

준비 단계(`/from-image`, `/from-gift-data`)의 응답은 네 작업 결과를 하나로 묶고
`requires_confirmation: true` 를 붙입니다.

```json
{
  "gift_data":       { "status": "SUCCESS", "payload": { ... } },
  "calendar_info":   { "status": "SUCCESS", "payload": { "registered": false, ... } },
  "noti_info":       { "status": "SUCCESS", "payload": { ... } },
  "recommend_gift_info": {
    "status": "SUCCESS",
    "recommend_gift": {
      "input_gift_name": "스타벅스 케이크",
      "input_gift_price": 35000,
      "input_age": 29,
      "recommended_price_min": 28000,
      "recommended_price_max": 42000,
      "categories": [
        {
          "category": "식품·디저트",
          "score": 88,
          "reason": "케이크와 같은 결의 디저트로 답례하면 자연스럽습니다.",
          "product_examples": ["프리미엄 쿠키 선물세트", "디저트 기프트박스"],
          "search_query": "식품·디저트 답례 선물 28000원 42000원"
        }
      ],
      "products": [
        {
          "title": "[삼청동 소샌드 흑임자 12개입] 프리미엄 쿠키 선물",
          "url": "https://gift.kakao.com/product/...",
          "source": "카카오 선물하기",
          "category": "식품·디저트",
          "price": 39000,
          "price_verified": true,
          "kind": "product",
          "reason": "식품·디저트 선물로 고른 카카오 선물하기 상품. 판매가 39,000원으로 제안 가격대 안입니다",
          "snippet": null
        }
      ],
      "summary": "…",
      "rationale": {
        "price_range_basis": "받은 금액 35,000원의 80%(28,000원) ~ 120%(42,000원)를 …",
        "inputs_used": ["나이 29세", "관계 직장 동료", "받은 금액 35,000원"],
        "category_basis": "연령대·성별·관계를 고려해 …",
        "product_basis": "카카오 선물하기, 컬리에서 찾았습니다. 2개 중 2개는 상품 페이지에서 판매가를 확인했고, 2개가 28,000원 ~ 42,000원 안에 듭니다.",
        "warnings": []
      },
      "model": "global.anthropic.claude-sonnet-4-6",
      "source": "BEDROCK_CLAUDE"
    },
    "message": {
      "tone": "따뜻하고 구체적이며 부담 없는 말투",
      "content": "김민수님, 맛있는 케이크를 보내주셔서 정말 감사합니다. …",
      "generated_by": "BEDROCK_CLAUDE",
      "message_source": "MODEL"
    }
  },
  "workflow_id": "9f1c...",
  "requires_confirmation": true
}
```

손대지 않은 원문 응답은 `docs/api-examples.http` 에, 전체 스키마는 Swagger 와
`docs/openapi.json` 에 있습니다. `/confirm` 을 거치면 `provider` 가 `GOOGLE_MCP` 로 바뀌고
`registered: true` 와 함께 `eventId`, `htmlLink` 가 채워집니다.

`age` 에 `0`, `"0"`, 빈 문자열, `null` 을 보내면 나이 정보가 없는 것으로 처리하며, 최종
응답에서는 값이 `null` 인 선택 필드가 생략될 수 있습니다.

### 작업별 status

| 값 | 뜻 |
|---|---|
| `SUCCESS` | 정상 처리 |
| `ERROR` | 실패. `error` 에 사용자에게 보여 줄 문구가 들어갑니다 |
| `SKIPPED` | 실패가 아니라 "이 입력에는 필요 없음". `reason` 에 사유가 들어갑니다 |

부분 실패 시 HTTP 응답 자체는 `200` 이며 실패한 작업만 `{"status": "ERROR", "error": "…"}` 로
표시됩니다. 선물데이터 생성 자체가 실패하거나 이미지 분석이 실패하면 네 작업을 시작할 수 없으므로
`422` 또는 `502` 를 반환합니다.

### SKIPPED — 오류가 아닙니다

`SKIPPED` 는 현재 `recommend_gift_info` 에서만 나옵니다.

| 사유 | 설명 |
|---|---|
| 답례 대상이 아님 | 현금·부조금(`money`)과 영수증(`receipt`). 여러 명이 서로 다른 금액을 낸 축의금 명단에 하나의 가격대를 권하면 한쪽에는 과하고 다른 쪽에는 모자랍니다 |
| 금액을 모름 | `gift_price` 가 `null`. 답례 가격대는 받은 금액 기준이라 금액 없이는 성립하지 않습니다 |

```json
{
  "recommend_gift_info": {
    "status": "SKIPPED",
    "reason": "경조사로 선택하셔서 답례 선물은 추천하지 않았습니다. 받은 금액을 기준으로 답례 규모를 정해 보세요."
  }
}
```

두 경우 모두 모델을 호출하지 않으므로 응답이 그만큼 빠릅니다. **화면에 오류로 표시하지 마세요.**
`reason` 은 사용자에게 그대로 보여 줄 수 있는 문장이고 사유마다 문구가 다릅니다. 사용자가 대상을
고르거나 금액을 입력한 뒤 `POST /api/v1/agent/recommend` 를 호출하는 흐름으로 이어 주면 됩니다.

추천이 실행되는 기록 종류는 `gift` 와 `event_invitation`(청첩장·부고장) 뿐입니다. 청첩장은
답례품이 아니라 축의금 적정 수준을 안내하므로 포함합니다.

### message — 두 필드를 구분해서 읽으세요

`recommend_gift_info.message` 의 두 필드는 **서로 다른 것**을 말합니다. 같은 것으로 읽으면 품질
지표가 뒤집힙니다.

| 필드 | 무엇을 말하는가 | 값 |
|---|---|---|
| `generated_by` | 추천 **전체**를 만든 백엔드. `recommend_gift.source` 와 같은 값 | `BEDROCK_CLAUDE` / `BEDROCK_CLAUDE_FALLBACK` / `GEMMA_VLLM` / `QWEN_MLX` / `MOCK` |
| `message_source` | `content` **한 필드**를 누가 썼는지 | `MODEL` / `TEMPLATE_TOO_SHORT` / `TEMPLATE_NO_OUTPUT` |

메시지 교체는 추천 백엔드와 **완전히 별개로** 일어납니다. 모델 응답을 정상적으로 읽고
카테고리·가격까지 모델이 정했는데 메시지 문장만 길이 미달로 템플릿에 교체될 수 있습니다. 그래서
`generated_by: "BEDROCK_CLAUDE"` 와 `message_source: "TEMPLATE_TOO_SHORT"` 가 한 응답에 함께
나오며, 그것이 정상입니다.

**"모델이 쓴 문장인가"는 `message_source == "MODEL"` 하나로만 판정하세요.** `MODEL` 이 아닌 값은
전부 템플릿입니다.

| `message_source` | 뜻 |
|---|---|
| `MODEL` | 모델이 쓴 문장이 그대로 나갔습니다(이름·조사 교정만 적용) |
| `TEMPLATE_TOO_SHORT` | 모델이 쓰긴 했지만 길이 미달로 폐기하고 템플릿으로 대체했습니다 |
| `TEMPLATE_NO_OUTPUT` | 모델 문장이 아예 없었습니다. JSON 파싱 실패, 필드 누락, `MODEL_BACKEND=mock` 이 여기입니다 |

### products — 0건일 수 있습니다

`products` 는 Tavily 가 찾은 결과 중 **허용된 쇼핑 도메인의 개별 상품 상세페이지만** 포함합니다.
검색 결과·카테고리·기획전·기사 페이지는 사용자에게 실제 상품을 보여 주지 못하므로 제외합니다.
모바일/PC 주소가 달라도 같은 상품 ID면 하나로 합칩니다.

판매가는 검색 스니펫이 아니라 상품 페이지에서 확인하며, 확인된 값만 `price_verified: true` 로
표시합니다. 선별은 예산 안을 먼저 채우고 자리가 남을 때만 `PRODUCT_PRICE_SLACK_RATIO`(경계값
기준 ±15%) 안에 드는 것으로 보충합니다.

**가격을 전혀 모르는 상품은 어떤 경로로도 노출하지 않습니다.** 채울 것이 없으면 적게, 없으면
0건으로 나갑니다. 1건이라도 예산 안인 편이 3건 예산 밖보다 낫습니다. 최대 3건
(`PRODUCT_SUGGESTION_LIMIT`)이며, 0건이어도 `product_examples` 는 항상 안전한 대체 추천으로
유지됩니다.

0건일 때 "검색이 비었다"와 "찾았지만 가격대에 맞는 게 없다"는 다른 말이고,
`rationale.product_basis` 가 둘을 구분해 말합니다.

| 상황 | `product_basis` |
|---|---|
| 검색 자체가 비었음 | `상품 검색 결과가 없어 카테고리와 가격대만 제안했습니다.` |
| 후보는 찾았지만 가격이 안 맞음 | `상품 후보 9개를 찾았지만 8,000원 ~ 12,000원에 맞는 판매가를 확인하지 못했습니다.` |

한계: Tavily 는 쇼핑 API 가 아니라 범용 웹 검색이라 가격대로 결과를 거를 수 없고 가격을 사후에
확인합니다. 구조화된 쇼핑 API(네이버 쇼핑, 쿠팡 파트너스)를 쓰면 이 한계가 사라집니다.

### rationale — 왜 그 추천인가

카테고리별 이유는 모델이 쓰지만 `rationale` 값들은 규칙에서 결정론적으로 나오므로 사용자에게
그대로 보여 줘도 됩니다. `inputs_used` 에는 **실제로 반영된 입력만** 들어갑니다. 나이를 안 줬으면
나이가 나오지 않습니다. 없는 근거를 있는 것처럼 보여서는 안 됩니다.

## 백엔드 연동

```text
프론트엔드 ──사용자 인증──> Spring Boot ──X-API-KEY──> Giftie FastAPI
```

```env
AI_SERVICE_URL=http://giftie-ai:8000
AI_SERVICE_API_KEY=FastAPI의-API_KEY와-같은-값
```

Spring Boot 는 입력에 맞는 하나를 호출합니다.

- `POST {AI_SERVICE_URL}/api/v1/agent/from-gift-data`
- `POST {AI_SERVICE_URL}/api/v1/agent/from-image`

HTTP 타임아웃은 **90초 이상**으로 잡아 주세요. 서버가 스스로 끊는 최악 지연이 75초라서
90초면 백엔드가 먼저 끊는 일이 없습니다.

### 알아 둘 것

- `/from-image` 와 `/from-gift-data` 는 캘린더에 등록하지 않습니다. 초안까지만 만들고
  `requires_confirmation: true` 로 표시합니다. 실제 등록은 `/confirm` 에서만 일어납니다.
- 이 서비스는 상태를 보관하지 않습니다. 백엔드가 준비 응답을 들고 있다가 사용자 수정본과 함께
  `/confirm` 으로 되돌려주면 됩니다.
- 부분 실패는 200 입니다. 네 작업 중 하나가 죽어도 그 항목만 `status: "ERROR"` 이고 나머지는
  유지됩니다.
- `recommend_gift_info.status` 가 `SKIPPED` 면 `recommend_gift` 가 없습니다. 항상 있다고
  가정하지 마세요.
- Google access token 은 `/confirm` 요청 본문에 실어 주세요. 사용자별 토큰이므로 서버 설정값이
  아니라 요청마다 달라야 합니다.

### 계약 문서

- Swagger: `http://<AI-서버-주소>:8000/docs`
- OpenAPI 스펙: `docs/openapi.json` (Java 클라이언트 생성에 쓸 수 있습니다)
- 실행 가능한 요청 예시: `docs/api-examples.http` (IntelliJ / VS Code REST Client)

스펙 파일은 코드에서 뽑습니다. **계약을 바꾸면 다시 뽑아 주세요.**

```bash
python scripts/export_openapi.py          # 갱신
python scripts/export_openapi.py --check  # 코드와 다르면 종료코드 1 (CI 용)
```

### 모델 없이 흐름만 확인

```bash
docker compose up --build   # :8000 AI Service, :8300 Calendar MCP
```

## 테스트

```bash
source .venv/bin/activate
pytest -q
```

모든 테스트는 실제 Bedrock·vLLM·S3·Google 호출 없이 돕니다(respx 로 가로챔). 다루는 범위는
API 계약(공개 API 네 개, 인증, 입력 정규화, 부분 실패 보존), 이미지 추출 종단, 답례일·알림 규칙과
캘린더 MCP 왕복, 확정 흐름, 그리고 추천 파이프라인(가격 범위, 카테고리 정책, 추천 대상 분기,
상품 검색·판정·계절 필터, 근거 문구)입니다. 파일 단위 구성은 `tests/` 를 보세요.

## 배포

```env
MODEL_BACKEND=bedrock
BEDROCK_API_STYLE=invoke
BEDROCK_REGION=us-east-1
BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-4-6
CALENDAR_MCP_URL=http://calendar-mcp:8300/mcp
API_KEY=운영용-긴-비밀키
```

```bash
docker build -t giftie-ai .
docker run --rm -p 8000:8000 --env-file .env giftie-ai
```

`MODEL_BACKEND=bedrock` 이면 이 컨테이너에 GPU나 모델 파일이 필요하지 않습니다. EC2 IAM Role 에
Bedrock 모델 호출 권한을 부여하면 장기 API 키 없이 실행할 수 있습니다.

`transformers` 백엔드로 모델을 이 프로세스에 직접 올리는 경우에는 GPU 하나당 Uvicorn worker 를
하나만 실행해야 합니다. worker 를 늘리면 각 프로세스가 모델을 별도로 적재해 GPU 메모리를 중복
사용합니다. Docker 헬스체크는 `/openapi.json` 을 사용합니다.
