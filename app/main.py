"""Giftie FastAPI 애플리케이션 생성과 라우터 등록 진입점."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.exception_handlers import register_exception_handlers
from app.routers.agent import router as agent_router
from app.services.qwen_service import qwen_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    """설정된 경우 서버 시작 시 모델을 미리 적재합니다."""
    if settings.model_backend in {"transformers", "mlx"} and settings.preload_model:
        qwen_service.load()
    yield


app = FastAPI(
    title="Giftie AI 추천 서비스",
    version="0.1.0",
    description="받은 선물과 관계 맥락을 바탕으로 답례 가격대와 카테고리를 추천합니다.",
    lifespan=lifespan,
)

register_exception_handlers(app)

# 외부 백엔드에 공개하는 업무 API는 agent 라우터의 두 개뿐입니다.
app.include_router(agent_router, prefix="/api/v1")
