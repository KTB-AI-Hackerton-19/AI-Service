"""Spring Boot가 호출하는 Giftie 에이전트 HTTP API."""

import logging
from typing import Annotated, Awaitable

from fastapi import APIRouter, Body, Depends, status

from app.core.errors import ApiErrorResponse, ErrorCode, GiftieHTTPException
from app.core.security import verify_api_key
from app.schemas.agent import (
    ConfirmRequest,
    ConfirmResponse,
    GiftAgentResponse,
    GiftDataRequest,
    ImageRequest,
    RecommendRequest,
    RecommendResponse,
)
from app.services.confirmation_service import confirmation_service
from app.services.gift_agent_service import (
    GiftInputAnalysisError,
    ImageAnalysisError,
    gift_agent_service,
)
from app.services.qwen_service import RecommendationGenerationError
from app.services.tasks.recommendation import recommendation_preparation_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["gift-agent"],
    dependencies=[Depends(verify_api_key)],
    responses={
        401: {"model": ApiErrorResponse, "description": "API 키 인증 실패"},
        422: {"model": ApiErrorResponse, "description": "요청 데이터 검증 실패"},
        500: {"model": ApiErrorResponse, "description": "내부 처리 오류"},
        502: {"model": ApiErrorResponse, "description": "Bedrock 등 외부 서비스 오류"},
    },
)


@router.post(
    "/from-gift-data",
    response_model=GiftAgentResponse,
    response_model_exclude_none=True,
)
async def prepare_from_gift_data(
    request: Annotated[
        GiftDataRequest,
        Body(
            openapi_examples={
                "singleGift": {
                    "summary": "단건 선물데이터",
                    "description": "일반적인 직접 입력입니다. records를 보내지 않습니다.",
                    "value": {
                        "gift_data": {
                            "gift_name": "스타벅스 케이크",
                            "gift_price": 35000,
                            "age": 29,
                            "gender": "female",
                            "person_name": "김민수",
                            "relationship": "대학 동기",
                            "received_at": "2026-08-19",
                            "target_date": "2026-09-10",
                        }
                    },
                },
                "multipleRecords": {
                    "summary": "여러 건의 기록",
                    "description": "계좌 거래내역처럼 여러 건이면 records를 함께 보냅니다.",
                    "value": {
                        "gift_data": {
                            "gift_name": "축의금",
                            "gift_price": 200000,
                            "person_name": "최은비",
                            "received_at": "2026-08-19",
                            "records": [
                                {
                                    "record_id": "r0",
                                    "record_type": "money",
                                    "direction": "received",
                                    "person_name": "김도윤",
                                    "gift_name": "축의금",
                                    "price": 100000,
                                    "received_at": "2026-08-19",
                                    "confidence": 1.0,
                                    "selected": True,
                                },
                                {
                                    "record_id": "r1",
                                    "record_type": "money",
                                    "direction": "received",
                                    "person_name": "박서준",
                                    "gift_name": "축의금",
                                    "price": 50000,
                                    "received_at": "2026-08-19",
                                    "confidence": 1.0,
                                    "selected": True,
                                },
                            ],
                        }
                    },
                },
            }
        ),
    ]
) -> GiftAgentResponse:
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
async def prepare_from_image(
    request: Annotated[
        ImageRequest,
        Body(
            openapi_examples={
                "s3PresignedUrl": {
                    "summary": "S3 presigned 이미지 URL",
                    "value": {
                        "image_url": "https://example-bucket.s3.ap-northeast-2.amazonaws.com/u1/gift.png?X-Amz-Signature=example"
                    },
                },
                "gift": {
                    "summary": "사용자가 '선물'을 고른 경우 — 답례 선물 추천 실행",
                    "value": {
                        "image_url": "https://example-bucket.s3.ap-northeast-2.amazonaws.com/u1/gift.png",
                        "category": "gift",
                    },
                },
                "occasion": {
                    "summary": "사용자가 '경조사'를 고른 경우 — 추천은 SKIPPED",
                    "value": {
                        "image_url": "https://example-bucket.s3.ap-northeast-2.amazonaws.com/u1/ledger.jpg",
                        "category": "occasion",
                    },
                }
            }
        ),
    ]
) -> GiftAgentResponse:
    """S3 이미지 주소를 선물데이터로 변환한 뒤 네 작업을 실행합니다.

    ``MODEL_BACKEND=bedrock`` 또는 ``vllm``에서는 실제 이미지 분석을 수행하고,
    ``mock``에서는 외부 모델 없이 고정 결과로 전체 연동 흐름을 검증합니다.
    """
    return await _execute(gift_agent_service.run_from_image(
            str(request.image_url), request.category
        ))


@router.post(
    "/confirm",
    response_model=ConfirmResponse,
    response_model_exclude_none=True,
)
async def confirm(
    request: Annotated[
        ConfirmRequest,
        Body(
            openapi_examples={
                "confirmDraft": {
                    "summary": "사용자 검토 후 확정",
                    "value": {
                        "workflow_id": "from-previous-response-workflow-id",
                        "gift_data": {
                            "gift_name": "스타벅스 케이크",
                            "gift_price": 35000,
                            "person_name": "김민수",
                            "relationship": "대학 동기",
                            "received_at": "2026-08-19",
                            "target_date": "2026-09-10",
                        },
                        "calendar": None,
                        "approved": True,
                        "register_calendar": True,
                    },
                }
            }
        ),
    ]
) -> ConfirmResponse:
    """사용자가 확인 화면에서 검토·수정한 결과를 확정하고 캘린더에 등록합니다.

    ``/from-image`` 나 ``/from-gift-data`` 는 일정을 등록하지 않고 초안까지만 만듭니다.
    잘못 추출된 일정이 사용자 캘린더에 바로 박히면 되돌리기 어렵기 때문입니다.

    이 서비스는 상태를 보관하지 않으므로, 백엔드가 직전 응답을 들고 있다가
    사용자 수정본과 함께 그대로 되돌려주면 됩니다.

    Args:
        request: 사용자 수정본이 반영된 확정 요청.

    Returns:
        확정된 기록, 캘린더 등록 결과, 알림 예약.
    """
    try:
        return await confirmation_service.confirm(request)
    except Exception as exc:
        logger.exception("확정 처리 실패 workflow=%s", request.workflow_id)
        raise GiftieHTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=ErrorCode.CONFIRMATION_FAILED,
            detail="확정 처리 중 내부 오류가 발생했습니다.",
        ) from exc


@router.post(
    "/recommend",
    response_model=RecommendResponse,
    response_model_exclude_none=True,
)
async def recommend(
    request: Annotated[
        RecommendRequest,
        Body(
            openapi_examples={
                "recommendOnly": {
                    "summary": "추천 조건만 다시 실행",
                    "value": {
                        "age": 32,
                        "gender": "male",
                        "budget_min": 18000,
                        "budget_max": 27000,
                        "categories": ["꽃·식물"],
                        "gift_name": "꽃",
                        "gift_price": 23333,
                        "person_name": "김영삼",
                        "relationship": "친구",
                    },
                }
            }
        ),
    ]
) -> RecommendResponse:
    """추천만 단독으로 실행합니다.

    나이·가격대·카테고리·성별만으로도 추천이 나옵니다. 사용자가 확인 화면에서
    조건을 바꿔 다시 추천받을 때, 이미지 분석과 캘린더·알림을 다시 돌릴 이유가 없습니다.

    Args:
        request: 추천 조건.

    Returns:
        추천 가격대·카테고리·실제 상품·메시지와 그 근거.
    """
    try:
        info = await recommendation_preparation_service.recommend_only(request)
    except RecommendationGenerationError as exc:
        raise GiftieHTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code=ErrorCode.RECOMMENDATION_FAILED,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("추천 실패")
        raise GiftieHTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=ErrorCode.RECOMMENDATION_FAILED,
            detail="추천 처리 중 내부 오류가 발생했습니다.",
        ) from exc
    return RecommendResponse(recommend_gift_info=info)


async def _execute(operation: Awaitable[GiftAgentResponse]) -> GiftAgentResponse:
    """서비스 예외를 외부에 노출할 안전한 HTTP 오류로 변환합니다."""
    try:
        return await operation
    except GiftInputAnalysisError as exc:
        raise GiftieHTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code=ErrorCode.GIFT_INPUT_INVALID,
            detail=str(exc),
        ) from exc
    except ImageAnalysisError as exc:
        raise GiftieHTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code=ErrorCode.IMAGE_ANALYSIS_FAILED,
            detail=str(exc),
        ) from exc
    except RecommendationGenerationError as exc:
        raise GiftieHTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code=ErrorCode.RECOMMENDATION_FAILED,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("처리되지 않은 에이전트 API 오류")
        raise GiftieHTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=ErrorCode.AGENT_EXECUTION_FAILED,
            detail="요청 처리 중 내부 오류가 발생했습니다.",
        ) from exc
