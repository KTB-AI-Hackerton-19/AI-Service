"""Giftie FastAPI 애플리케이션 생성과 라우터 등록 진입점."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.exception_handlers import register_exception_handlers
from app.core.logging_config import RequestLoggingMiddleware, configure_logging
from app.routers.agent import router as agent_router
from app.services.qwen_service import qwen_service

configure_logging()

logger = logging.getLogger(__name__)


def _warm_bedrock() -> None:
    """Bedrock 클라이언트를 첫 요청 전에 만들어 둡니다. 네트워크는 타지 않습니다.

    bedrock 백엔드에서 첫 요청이 유독 느린 이유는 모델이 아니라 준비 비용입니다.
    ``import anthropic`` 과 클라이언트 생성이 지금은 첫 요청 **안에서** 일어납니다
    (qwen_service / vlm_service 안의 지연 import, bedrock_client 의 지연 생성).
    데모의 첫 호출이 가장 눈에 띄는 호출이므로 여기서 미리 치릅니다.

    하는 일은 모듈 임포트와 클라이언트 객체 생성까지입니다. 호출은 하지 않으므로
    AWS 로 나가는 요청도, 크레딧도 없습니다. 자격증명이 없거나 설정이 잘못돼
    생성이 실패해도 서버 기동은 계속합니다. 그 오류는 실제 요청에서 지금처럼
    502 로 드러나야 하고, 기동 자체가 죽으면 헬스체크가 컨테이너를 재시작합니다.
    """
    try:
        from app.services import bedrock_client

        bedrock_client.get_client()  # 추천(동기)
        bedrock_client.get_async_client()  # 이미지 분석(비동기)
    except Exception as exc:  # 어떤 이유로도 기동을 막지 않습니다.
        logger.warning("Bedrock 클라이언트 예열 실패. 첫 요청에서 다시 시도합니다: %s", exc)
    else:
        logger.info("Bedrock 클라이언트 예열 완료 style=%s", settings.bedrock_api_style)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """설정된 경우 서버 시작 시 모델·클라이언트를 미리 준비합니다."""
    if settings.model_backend in {"transformers", "mlx"} and settings.preload_model:
        qwen_service.load()
    if settings.model_backend == "bedrock":
        _warm_bedrock()
    yield


app = FastAPI(
    title="Giftie AI 추천 서비스",
    version="0.1.0",
    description="받은 선물과 관계 맥락을 바탕으로 답례 가격대와 카테고리를 추천합니다.",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
register_exception_handlers(app)

# 외부 백엔드에 공개하는 업무 API는 agent 라우터의 두 개뿐입니다.
app.include_router(agent_router, prefix="/api/v1")
