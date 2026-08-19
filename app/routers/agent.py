"""Spring Boot가 호출하는 두 개의 Giftie 에이전트 HTTP API."""

import logging
from typing import Awaitable

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import verify_api_key
from app.schemas.agent import GiftAgentResponse, GiftDataRequest, ImageRequest
from app.services.gift_agent_service import (
    GiftInputAnalysisError,
    ImageAnalysisError,
    gift_agent_service,
)
from app.services.qwen_service import RecommendationGenerationError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["gift-agent"],
    dependencies=[Depends(verify_api_key)],
)


@router.post(
    "/from-gift-data",
    response_model=GiftAgentResponse,
    response_model_exclude_none=True,
)
async def prepare_from_gift_data(request: GiftDataRequest) -> GiftAgentResponse:
    """구조화된 선물데이터로 네 가지 준비 작업을 실행합니다.

    Args:
        request: 백엔드가 전달한 ``gift_data`` 요청 본문.

    Returns:
        선물 기록, 캘린더, 알림, 추천/메시지를 합친 응답.
    """
    return await _execute(gift_agent_service.run_from_gift_data(request.gift_data))


@router.post(
    "/from-image",
    response_model=GiftAgentResponse,
    response_model_exclude_none=True,
)
async def prepare_from_image(request: ImageRequest) -> GiftAgentResponse:
    """S3 이미지 주소를 선물데이터로 변환한 뒤 네 작업을 실행합니다.

    현재 이미지 변환 함수는 mock이며 추후 이미지 분석 담당 구현으로 교체됩니다.
    """
    return await _execute(gift_agent_service.run_from_image(str(request.image_url)))


async def _execute(operation: Awaitable[GiftAgentResponse]) -> GiftAgentResponse:
    """서비스 예외를 외부에 노출할 안전한 HTTP 오류로 변환합니다."""
    try:
        return await operation
    except GiftInputAnalysisError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except (ImageAnalysisError, RecommendationGenerationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("처리되지 않은 에이전트 API 오류")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="요청 처리 중 내부 오류가 발생했습니다.",
        ) from exc
