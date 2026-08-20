"""설정된 백엔드(Bedrock / vLLM / MLX / Transformers)로 답례 선물을 추천합니다."""

import logging
from threading import Lock
from typing import Any

from app.core.config import settings
from app.schemas.recommendation import (
    CategoryRecommendation,
    SimpleGiftRecommendationRequest,
    SimpleGiftRecommendationResponse,
)
from app.services import bedrock_client
from app.services.prompt import build_recommendation_schema, build_simple_messages
from app.services.model_response_parser import ModelResponseParseError, parse_json_object
from app.services.recommendation_policy import normalize_recommendation
from app.services.price_policy import calculate_recommended_price_range

logger = logging.getLogger(__name__)


class RecommendationGenerationError(RuntimeError):
    """모델 적재, 추론 또는 모델 응답 파싱이 실패했을 때 발생합니다."""


class QwenRecommendationService:
    """설정된 백엔드로 Qwen 추론을 실행하는 단일 모델 서비스.

    한 프로세스에서 모델을 한 번만 적재합니다. MLX 추론은 모델 객체가 동시에
    사용되지 않도록 lock으로 보호합니다. FastAPI의 이벤트 루프를 막지 않도록
    호출 측에서는 이 동기 서비스를 ``asyncio.to_thread``로 실행합니다.
    """

    def __init__(self) -> None:
        """아직 적재되지 않은 모델과 동시성 제어 lock을 초기화합니다."""
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_lock = Lock()
        self._generate_lock = Lock()
        # Claude 최신 모델은 temperature/top_p 를 받지 않습니다. 한 번 거부당하면 내립니다.
        self._bedrock_accepts_sampling = True
        # 구조화 출력도 같은 이유로 모델·호출 경로마다 다릅니다. 거부당하면 내리고
        # 프롬프트만으로 형식을 요구하던 예전 동작으로 돌아갑니다.
        self._bedrock_accepts_structured_output = True

    @property
    def is_loaded(self) -> bool:
        """모델과 토크나이저가 메모리에 적재됐는지 반환합니다."""
        return self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        """환경설정에 맞는 Qwen 모델을 최초 한 번만 적재합니다.

        Raises:
            RecommendationGenerationError: 모델 다운로드 또는 적재 실패 시.
        """
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            try:
                if settings.model_backend == "mlx":
                    from mlx_lm import load

                    self._model, self._tokenizer = load(settings.local_model_id)
                    return

                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(settings.model_id)
                self._model = AutoModelForCausalLM.from_pretrained(
                    settings.model_id,
                    torch_dtype="auto",
                    device_map="auto",
                )
                self._model.eval()
            except Exception as exc:
                raise RecommendationGenerationError(
                    f"Qwen 모델을 불러오지 못했습니다: {exc}"
                ) from exc

    def recommend_simple(
        self,
        request: SimpleGiftRecommendationRequest,
    ) -> SimpleGiftRecommendationResponse:
        """선물명·가격·선택적 나이를 받아 추천 결과를 생성합니다.

        Args:
            request: 검증이 끝난 추천 입력 모델.

        Returns:
            가격 범위, 최대 세 카테고리, 추천 이유가 포함된 결과.

        Raises:
            RecommendationGenerationError: 백엔드 설정 또는 추론 실패 시.
        """
        if settings.model_backend == "mock":
            return self._mock_recommend(request)
        # Bedrock 과 vLLM 은 모델을 이 프로세스에 적재하지 않으므로 load() 를 거치지 않습니다.
        if settings.model_backend == "bedrock":
            return self._generate_with_bedrock(request)
        # vLLM 은 모델을 이 프로세스에 적재하지 않으므로 load() 를 거치지 않습니다.
        # 이미지 분석과 같은 서버·같은 모델을 씁니다.
        if settings.model_backend == "vllm":
            return self._generate_with_vllm(request)
        if settings.model_backend not in {"mlx", "transformers"}:
            raise RecommendationGenerationError(
                f"지원하지 않는 MODEL_BACKEND입니다: {settings.model_backend}"
            )

        self.load()
        if settings.model_backend == "mlx":
            return self._generate_with_mlx(request)
        return self._generate_with_transformers(request)

    def _generate_with_bedrock(
        self,
        request: SimpleGiftRecommendationRequest,
    ) -> SimpleGiftRecommendationResponse:
        """Amazon Bedrock 의 Claude 로 추천을 생성합니다.

        vLLM 경로가 ``response_format`` 을 쓰듯 여기서도 ``output_config`` 로 출력
        형식을 강제합니다. 이유는 :func:`bedrock_client.output_config` 참고. 다만
        스키마가 온전히 실리지 않으므로 프롬프트의 스키마 지시문도 함께 보냅니다.

        그래도 파싱은 방어합니다. ``BEDROCK_MODEL_ID`` 는 바꿔 가며 쓰는 값이고
        구조화 출력을 거부하는 모델이면 프롬프트만 남기 때문입니다. 실패하면 MLX
        경로와 동일하게 안전 추천으로 대체합니다. 추천 하나 때문에 기록·캘린더·
        알림까지 깨뜨리지 않기 위함입니다.

        ``recommend_simple`` 은 호출 측에서 ``asyncio.to_thread`` 로 실행되므로
        여기서는 동기 클라이언트를 씁니다.
        """
        import anthropic

        messages = build_simple_messages(request)
        # Anthropic Messages API 는 system 을 messages 가 아니라 별도 인자로 받습니다.
        system = next(m["content"] for m in messages if m["role"] == "system")
        # Bedrock 에는 response_format 이 없으므로 스키마를 프롬프트로 못박습니다.
        system += "\n\n" + bedrock_client.schema_instruction(build_recommendation_schema())
        user_messages = [m for m in messages if m["role"] != "system"]

        payload = {
            "model": settings.bedrock_model_id,
            "max_tokens": settings.bedrock_max_tokens,
            "system": system,
            "messages": user_messages,
        }
        # Claude 는 temperature 와 top_p 를 동시에 받지 않으므로(둘 중 하나만) 문장
        # 다양성을 좌우하는 temperature 만 보냅니다. 또한 Opus 4.6+ / Sonnet 5 등은
        # 샘플링 파라미터 자체를 400 으로 거부합니다. BEDROCK_MODEL_ID 는 바꿔 가며
        # 쓰는 값이므로, 거부당하면 한 번만 감지해 빼고 다시 보냅니다.
        #
        # settings.temperature(1.0)가 아니라 bedrock_temperature 를 씁니다. 1.0 은
        # Gemma 권장값이고 vLLM 경로는 response_format 이 JSON 을 강제하지만, 이
        # 경로는 위에서 보듯 형식 준수를 프롬프트에만 의존합니다.
        if self._bedrock_accepts_sampling:
            payload["temperature"] = settings.bedrock_temperature
        if self._bedrock_accepts_structured_output:
            payload["output_config"] = bedrock_client.output_config(
                build_recommendation_schema()
            )

        try:
            response = self._call_bedrock(payload)
        except bedrock_client.BedrockClientError as exc:
            raise RecommendationGenerationError(str(exc)) from exc
        except anthropic.AnthropicError as exc:
            raise RecommendationGenerationError(
                bedrock_client.describe_failure(exc)
            ) from exc

        if response.stop_reason == "max_tokens":
            logger.warning(
                "Bedrock 응답이 max_tokens(%s)에서 잘렸습니다. BEDROCK_MAX_TOKENS 를 늘리세요.",
                settings.bedrock_max_tokens,
            )
        text = bedrock_client.extract_text(response)
        try:
            parsed = parse_json_object(text)
        except ModelResponseParseError:
            logger.warning("Bedrock 응답 JSON 파싱 실패. 안전 추천으로 대체합니다.")
            parsed = {}

        return SimpleGiftRecommendationResponse(
            **normalize_recommendation(request, parsed),
            input_gift_name=request.gift_name,
            input_gift_price=request.gift_price,
            input_age=request.age,
            model=settings.bedrock_model_id,
            source="BEDROCK_CLAUDE" if parsed else "BEDROCK_CLAUDE_FALLBACK",
        )

    def _call_bedrock(self, payload: dict[str, Any]) -> Any:
        """Bedrock 을 호출하되 모델이 거부한 파라미터를 빼고 한 번만 재시도합니다.

        거부 대상은 둘입니다. 최신 모델은 샘플링 파라미터를 400 으로 막고, 일부
        모델·호출 경로는 구조화 출력을 막습니다. 어느 쪽인지는 오류 본문에만
        드러나므로 그걸 보고 하나만 골라 내립니다.
        """
        import anthropic

        client = bedrock_client.get_client()
        try:
            return client.messages.create(**payload)
        except anthropic.BadRequestError as exc:
            dropped = self._drop_rejected(payload, exc)
            if not dropped:
                raise
            logger.warning(
                "%s 가 %s 를 거부해 빼고 재시도합니다: %s",
                settings.bedrock_model_id,
                ", ".join(dropped),
                bedrock_client.upstream_message(exc),
            )
            return client.messages.create(**payload)

    def _drop_rejected(self, payload: dict[str, Any], exc: Exception) -> list[str]:
        """거부된 파라미터를 payload 에서 빼고 그 이름을 돌려줍니다.

        뺄 것이 없으면 빈 목록입니다. 호출 측은 그때 원래 오류를 그대로 올립니다.
        """
        message = bedrock_client.upstream_message(exc)
        if "output_config" in message and "output_config" in payload:
            self._bedrock_accepts_structured_output = False
            payload.pop("output_config")
            return ["구조화 출력"]
        sampling = [k for k in ("temperature", "top_p", "top_k") if k in payload]
        if sampling:
            self._bedrock_accepts_sampling = False
            for key in sampling:
                payload.pop(key)
            return [f"샘플링 파라미터({', '.join(sampling)})"]
        return []

    def _generate_with_vllm(
        self,
        request: SimpleGiftRecommendationRequest,
    ) -> SimpleGiftRecommendationResponse:
        """이미지 분석과 같은 vLLM 서버에서 추천을 생성합니다.

        GPU 한 장에 모델을 두 벌 올리지 않기 위한 경로입니다. vLLM 의 연속 배칭 덕분에
        추천 요청과 이미지 분석 요청이 동시에 들어와도 한 엔진에서 함께 처리됩니다.

        ``recommend_simple`` 은 호출 측에서 ``asyncio.to_thread`` 로 실행되므로
        여기서는 동기 HTTP 클라이언트를 씁니다.
        """
        import httpx

        body = {
            "model": settings.vllm_model,
            "messages": build_simple_messages(request),
            "max_tokens": settings.max_new_tokens,
            # Gemma 공식 권장 샘플링. 추출과 달리 여기는 문장 다양성이 품질이다.
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "top_k": settings.top_k,
            # 이미지 추출과 마찬가지로 구조화 출력을 강제한다. 카테고리를 enum 으로 못박아
            # 두면 모델이 목록 밖의 값을 만들 수 없고, JSON 파싱 실패도 원천 차단된다.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "gift_recommendation",
                    "schema": build_recommendation_schema(),
                    "strict": True,
                },
            },
        }
        try:
            with httpx.Client(
                base_url=settings.vllm_base_url,
                timeout=settings.vllm_timeout_seconds,
                headers={"Authorization": f"Bearer {settings.vllm_api_key}"},
            ) as client:
                response = client.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise RecommendationGenerationError(
                f"vLLM 서버에 연결할 수 없습니다: {exc}"
            ) from exc

        if response.status_code != 200:
            raise RecommendationGenerationError(
                f"vLLM 응답 오류 HTTP {response.status_code}: {response.text[:300]}"
            )

        try:
            text = response.json()["choices"][0]["message"]["content"] or ""
            normalized = normalize_recommendation(request, parse_json_object(text))
        except ModelResponseParseError as exc:
            raise RecommendationGenerationError(
                f"vLLM 응답을 JSON 으로 읽지 못했습니다: {exc}"
            ) from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise RecommendationGenerationError(
                f"vLLM 응답 형식이 올바르지 않습니다: {exc}"
            ) from exc

        return SimpleGiftRecommendationResponse(
            **normalized,
            input_gift_name=request.gift_name,
            input_gift_price=request.gift_price,
            input_age=request.age,
            model=settings.vllm_model,
            source="GEMMA_VLLM",
        )

    def _generate_with_transformers(
        self,
        request: SimpleGiftRecommendationRequest,
    ) -> SimpleGiftRecommendationResponse:
        """CUDA 서버용 Transformers 백엔드에서 동기 추론을 실행합니다."""
        try:
            prompt = self._tokenizer.apply_chat_template(
                build_simple_messages(request),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                do_sample=False,
            )
            generated = output_ids[0][inputs.input_ids.shape[1] :]
            text = self._tokenizer.decode(generated, skip_special_tokens=True)
            parsed = normalize_recommendation(request, parse_json_object(text))
            return SimpleGiftRecommendationResponse(
                **parsed,
                input_gift_name=request.gift_name,
                input_gift_price=request.gift_price,
                input_age=request.age,
                model=settings.model_id,
                source="QWEN_TRANSFORMERS",
            )
        except RecommendationGenerationError:
            raise
        except Exception as exc:
            raise RecommendationGenerationError(
                f"Transformers Qwen 추천 생성에 실패했습니다: {exc}"
            ) from exc

    def _generate_with_mlx(
        self,
        request: SimpleGiftRecommendationRequest,
    ) -> SimpleGiftRecommendationResponse:
        """Apple Silicon용 MLX 백엔드에서 동기 추론을 실행합니다.

        첫 응답의 JSON 문법이 잘못되면 오류 내용을 대화에 추가하고 한 번 재시도합니다.
        """
        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            messages = build_simple_messages(request)
            parsed: dict[str, Any] | None = None
            for attempt in range(2):
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                with self._generate_lock:
                    text = generate(
                        self._model,
                        self._tokenizer,
                        prompt=prompt,
                        max_tokens=settings.max_new_tokens,
                        sampler=make_sampler(temp=0.0),
                        verbose=False,
                    )
                try:
                    parsed = parse_json_object(text)
                    break
                except ModelResponseParseError:
                    if attempt == 1:
                        # 추천 하나가 실패했다고 기록·캘린더·알림까지 사용자 경험을
                        # 깨뜨리지 않습니다. 가격 정책과 안전 카테고리, 장문 메시지
                        # 템플릿으로 결정론적 fallback을 만듭니다.
                        logger.warning(
                            "MLX 응답 JSON 파싱 2회 실패. 안전 추천으로 대체합니다."
                        )
                        parsed = {}
                        break
                    messages.extend(
                        [
                            {"role": "assistant", "content": text},
                            {
                                "role": "user",
                                "content": (
                                    "위 응답은 JSON 문법이 잘못되었습니다. 설명이나 코드 "
                                    "블록 없이 요구된 스키마의 JSON 객체만 다시 출력하세요."
                                ),
                            },
                        ]
                    )

            if parsed is None:
                raise RecommendationGenerationError("Qwen 응답이 비어 있습니다.")
            normalized = normalize_recommendation(request, parsed)
            return SimpleGiftRecommendationResponse(
                **normalized,
                input_gift_name=request.gift_name,
                input_gift_price=request.gift_price,
                input_age=request.age,
                model=settings.local_model_id,
                source="QWEN_MLX" if parsed else "QWEN_MLX_FALLBACK",
            )
        except RecommendationGenerationError:
            raise
        except Exception as exc:
            raise RecommendationGenerationError(
                f"MLX Qwen 추천 생성에 실패했습니다: {exc}"
            ) from exc

    @staticmethod
    def _mock_recommend(
        request: SimpleGiftRecommendationRequest,
    ) -> SimpleGiftRecommendationResponse:
        """모델 없이 API 흐름을 시험할 수 있는 결정적 mock 추천을 반환합니다."""
        minimum, maximum = calculate_recommended_price_range(request.gift_price)
        # 메시지는 정책 템플릿에서 나옵니다. mock 은 모델을 부르지 않으므로
        # message_source 도 정책이 내는 값(TEMPLATE_NO_OUTPUT)을 그대로 씁니다.
        # 여기서 손으로 적으면 정책과 갈라져 mock 만 다른 말을 하게 됩니다.
        normalized = normalize_recommendation(request, {})
        age_hint = f"{request.age}세 연령대를 고려하고 " if request.age is not None else ""
        return SimpleGiftRecommendationResponse(
            input_gift_name=request.gift_name,
            input_gift_price=request.gift_price,
            input_age=request.age,
            recommended_price_min=minimum,
            recommended_price_max=maximum,
            categories=[
                CategoryRecommendation(
                    category="식품·디저트",
                    score=90,
                    reason=f"{age_hint}받은 선물과 비슷한 부담으로 답례하기 좋습니다.",
                    product_examples=["프리미엄 디저트 세트", "커피·티 세트"],
                    search_query=f"답례 디저트 {minimum}원 {maximum}원",
                ),
                CategoryRecommendation(
                    category="생활용품",
                    score=82,
                    reason="취향을 크게 타지 않으면서 실용적으로 사용할 수 있습니다.",
                    product_examples=["홈 프래그런스", "고급 타월 세트"],
                    search_query=f"답례 생활용품 {minimum}원 {maximum}원",
                ),
            ],
            summary=f"{request.gift_name}의 가격을 참고해 부담 없는 답례 범위를 정했습니다.",
            suggested_message=normalized["suggested_message"],
            message_source=normalized["message_source"],
            model=settings.local_model_id,
            source="MOCK",
        )


# 애플리케이션 전체에서 모델을 한 번만 적재하도록 singleton 인스턴스를 공유합니다.
qwen_service = QwenRecommendationService()
