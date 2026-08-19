"""Bedrock Claude 또는 vLLM VLM으로 이미지에서 선물 기록을 추출합니다.

Bedrock이 기본 실제 실행 경로이며 추천과 이미지 분석에 같은 Claude 설정을 사용합니다.
자체 GPU를 쓰는 경우에는 추천용 ``qwen_service``와 같은 vLLM 서버·모델을 공유합니다.

어느 경로든 모델을 이 프로세스에 적재하지 않고 HTTP로 요청합니다.

vLLM 서버 기동 예시(FastAPI 가 8000 을 쓰므로 8001 로 매핑):

    docker run --rm --gpus all -p 8001:8000 \\
      -v ~/.cache/huggingface:/root/.cache/huggingface --ipc=host \\
      vllm/vllm-openai:v0.27.1-x86_64-cu129 \\
      --model google/gemma-4-12B-it-qat-w4a16-ct --served-model-name gemma4-12b-qat \\
      --max-model-len 16384 --gpu-memory-utilization 0.90 \\
      --limit-mm-per-prompt '{"image": 2}'

MTP(Multi-Token Prediction)를 켜는 경우에도 OpenAI 호환 API 는 그대로이므로
이 클라이언트 코드는 바뀌지 않습니다. 서버 기동 플래그만 달라집니다.
"""

import base64
import logging
import re
from dataclasses import dataclass, field
from datetime import date

import httpx

from app.core.config import settings
from app.services.clock import service_today
from app.services.image_loader import LoadedImage
from app.services.model_response_parser import ModelResponseParseError, parse_json_object
from app.services.vision_prompt import (
    SYSTEM_PROMPT,
    build_extraction_prompt,
    build_extraction_schema,
)

logger = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class VisionAnalysisError(RuntimeError):
    """Bedrock/vLLM 호출 또는 응답 파싱이 실패했을 때 발생합니다."""


@dataclass
class VisionResult:
    """VLM 이 돌려준 원시 추출 결과."""

    payload: dict
    prompt_tokens: int = 0
    completion_tokens: int = 0
    warnings: list[str] = field(default_factory=list)


_MOCK_PAYLOAD = {
    "image_kind": "kakao_gift",
    "records": [
        {
            "record_type": "gift",
            "direction": "received",
            "counterpart_name": "김수현",
            "occurred_date": "2026-03-14",
            "event_date": None,
            "item_name": "아이스 카페 아메리카노 T",
            "brand": "스타벅스",
            "category": "기프티콘/음료",
            "event": "생일",
            "amount": 12300,
            "memo": "생일 축하해!",
            "confidence": 0.95,
        }
    ],
}


class VlmExtractionService:
    """이미지 한 장에서 선물·부조금 기록을 뽑아내는 서비스."""

    def __init__(self) -> None:
        """Bedrock 모델의 샘플링 파라미터 지원 여부를 기억합니다."""
        self._bedrock_accepts_sampling = True

    @property
    def uses_real_model(self) -> bool:
        """실제 VLM 을 호출하는지. False 면 이미지를 내려받을 필요도 없습니다."""
        return settings.model_backend in {"vllm", "bedrock"}

    async def extract(
        self,
        image: LoadedImage | None,
        today: date | None = None,
        category: str | None = None,
    ) -> VisionResult:
        """이미지를 VLM 에 넣어 구조화된 기록을 추출합니다.

        Args:
            image: ``image_loader`` 가 정규화한 이미지. mock 동작에서는 ``None`` 입니다.
            today: 연도가 보이지 않을 때 기준으로 삼을 날짜. 기본값은 오늘.
            category: 사용자가 업로드 화면에서 고른 종류(``gift`` / ``occasion``).
                프롬프트에 힌트로 실어 보냅니다. 비용도 지연도 늘지 않습니다.

        Returns:
            ``image_kind`` 와 ``records`` 를 담은 원시 결과.

        Raises:
            VisionAnalysisError: 호출 또는 응답 파싱이 실패한 경우.
        """
        if self.uses_real_model:
            if image is None:
                raise VisionAnalysisError(
                    f"model_backend={settings.model_backend} 인데 이미지가 전달되지 않았습니다."
                )
            if settings.model_backend == "bedrock":
                return await self._extract_with_bedrock(image, today or service_today(), category)
            return await self._extract_with_vllm(image, today or service_today(), category)

        # mlx / transformers 는 텍스트 전용 로컬 백엔드라 이미지를 볼 수 없습니다.
        # 개발이 막히지 않도록 실패시키지 않고 mock 으로 떨어뜨리되 경고를 남깁니다.
        warning = f"model_backend={settings.model_backend} 이라 이미지 분석이 mock 으로 동작했습니다"
        if settings.model_backend != "mock":
            logger.warning(warning)
        return VisionResult(payload=_MOCK_PAYLOAD, warnings=[warning])

    async def _extract_with_bedrock(
        self,
        image: LoadedImage,
        today: date,
        category: str | None = None,
    ) -> VisionResult:
        """Amazon Bedrock 의 Claude 로 이미지에서 기록을 추출합니다.

        Bedrock 은 구조화 출력을 지원하지 않으므로 스키마를 프롬프트에 실어 보내고
        관대한 파서로 읽습니다. 이미지는 URL 소스도 지원되지 않아 base64 로 넣습니다.
        """
        import anthropic

        from app.services import bedrock_client

        instruction = (
            f"{build_extraction_prompt(today.year, category)}\n\n"
            f"{bedrock_client.schema_instruction(build_extraction_schema())}"
        )
        payload = {
            # 이미지 판독은 추천보다 어려워 별도 모델을 씁니다(BEDROCK_VISION_MODEL_ID).
            "model": settings.bedrock_vision_model_id,
            # Bedrock은 JSON Schema까지 프롬프트에 포함하므로 vLLM 이미지 예산보다 넉넉히 둡니다.
            "max_tokens": settings.bedrock_max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image.mime,
                                "data": base64.b64encode(image.data).decode(),
                            },
                        },
                        {"type": "text", "text": instruction},
                    ],
                }
            ],
        }
        # 추출은 창의성이 필요 없어 temperature 0 이지만, Opus 4.6+ 등은 샘플링
        # 파라미터를 400 으로 거부합니다. 거부당하면 한 번만 감지해 빼고 재시도합니다.
        if self._bedrock_accepts_sampling:
            payload["temperature"] = settings.vision_temperature

        try:
            response = await self._call_bedrock(payload)
        except bedrock_client.BedrockClientError as exc:
            raise VisionAnalysisError(str(exc)) from exc
        except anthropic.AnthropicError as exc:
            raise VisionAnalysisError(bedrock_client.describe_failure(exc)) from exc

        text = _THINK_BLOCK.sub("", bedrock_client.extract_text(response)).strip()
        try:
            payload = parse_json_object(text)
        except ModelResponseParseError as exc:
            raise VisionAnalysisError(f"VLM 응답을 JSON 으로 읽지 못했습니다: {exc}") from exc

        warnings: list[str] = []
        if response.stop_reason == "max_tokens":
            warnings.append(
                f"출력이 max_tokens({settings.bedrock_max_tokens})에 걸려 잘렸을 수 있습니다"
            )
        return VisionResult(
            payload=payload,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            warnings=warnings,
        )

    async def _call_bedrock(self, payload: dict) -> object:
        """이미지 분석을 호출하고 temperature가 거부되면 제거해 한 번 재시도합니다."""
        import anthropic

        from app.services import bedrock_client

        client = bedrock_client.get_async_client()
        try:
            return await client.messages.create(**payload)
        except anthropic.BadRequestError:
            if "temperature" not in payload:
                raise
            logger.warning(
                "%s가 이미지 분석 temperature를 거부해 빼고 재시도합니다.",
                # 비전 경로는 추천과 다른 모델을 씁니다. 여기에 bedrock_model_id 를
                # 찍으면 실제로 거부한 모델과 다른 이름이 로그에 남습니다.
                settings.bedrock_vision_model_id,
            )
            self._bedrock_accepts_sampling = False
            payload.pop("temperature")
            return await client.messages.create(**payload)

    async def _extract_with_vllm(
        self,
        image: LoadedImage,
        today: date,
        category: str | None = None,
    ) -> VisionResult:
        """vLLM OpenAI 호환 엔드포인트에 구조화 출력을 강제해 요청합니다."""
        data_url = f"data:{image.mime};base64,{base64.b64encode(image.data).decode()}"
        body = {
            "model": settings.vllm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": build_extraction_prompt(today.year, category)},
                    ],
                },
            ],
            "max_tokens": settings.vision_max_new_tokens,
            "temperature": settings.vision_temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "gift_records",
                    "schema": build_extraction_schema(),
                    "strict": True,
                },
            },
        }

        try:
            async with httpx.AsyncClient(
                base_url=settings.vllm_base_url,
                timeout=settings.vllm_timeout_seconds,
                headers={"Authorization": f"Bearer {settings.vllm_api_key}"},
            ) as client:
                response = await client.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise VisionAnalysisError(f"vLLM 서버에 연결할 수 없습니다: {exc}") from exc

        if response.status_code != 200:
            raise VisionAnalysisError(
                f"vLLM 응답 오류 HTTP {response.status_code}: {response.text[:300]}"
            )

        completion = response.json()
        choice = completion["choices"][0]
        text = _THINK_BLOCK.sub("", choice["message"].get("content") or "").strip()

        try:
            payload = parse_json_object(text)
        except ModelResponseParseError as exc:
            raise VisionAnalysisError(f"VLM 응답을 JSON 으로 읽지 못했습니다: {exc}") from exc

        warnings: list[str] = []
        if choice.get("finish_reason") == "length":
            warnings.append(
                f"출력이 max_tokens({settings.vision_max_new_tokens})에 걸려 잘렸을 수 있습니다"
            )

        usage = completion.get("usage") or {}
        return VisionResult(
            payload=payload,
            prompt_tokens=usage.get("prompt_tokens") or 0,
            completion_tokens=usage.get("completion_tokens") or 0,
            warnings=warnings,
        )


vlm_extraction_service = VlmExtractionService()
