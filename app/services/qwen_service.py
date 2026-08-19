"""MLX/Transformers Qwen 모델을 이용해 답례 선물을 추천합니다."""

from threading import Lock
from typing import Any

from app.core.config import settings
from app.schemas.recommendation import (
    CategoryRecommendation,
    SimpleGiftRecommendationRequest,
    SimpleGiftRecommendationResponse,
)
from app.services.prompt import build_simple_messages
from app.services.model_response_parser import ModelResponseParseError, parse_json_object
from app.services.recommendation_policy import normalize_recommendation


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
            "temperature": settings.temperature,
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
                        raise
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
                source="QWEN_MLX",
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
        minimum = int(request.gift_price * 0.8 / 1000) * 1000
        maximum = int(request.gift_price * 1.2 / 1000) * 1000
        age_hint = f"{request.age}세 연령대를 고려하고 " if request.age is not None else ""
        return SimpleGiftRecommendationResponse(
            input_gift_name=request.gift_name,
            input_gift_price=request.gift_price,
            input_age=request.age,
            recommended_price_min=max(minimum, 1_000),
            recommended_price_max=max(maximum, 1_000),
            categories=[
                CategoryRecommendation(
                    category="식품·디저트",
                    score=90,
                    reason=f"{age_hint}받은 선물과 비슷한 부담으로 답례하기 좋습니다.",
                    product_examples=["프리미엄 디저트 세트", "커피·티 세트"],
                ),
                CategoryRecommendation(
                    category="생활용품",
                    score=82,
                    reason="취향을 크게 타지 않으면서 실용적으로 사용할 수 있습니다.",
                    product_examples=["홈 프래그런스", "고급 타월 세트"],
                ),
            ],
            summary=f"{request.gift_name}의 가격을 참고해 부담 없는 답례 범위를 정했습니다.",
            suggested_message=normalize_recommendation(request, {})[
                "suggested_message"
            ],
            model=settings.local_model_id,
            source="MOCK",
        )


# 애플리케이션 전체에서 모델을 한 번만 적재하도록 singleton 인스턴스를 공유합니다.
qwen_service = QwenRecommendationService()
