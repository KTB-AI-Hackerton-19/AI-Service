import asyncio
"""S3 선물 이미지를 공통 선물데이터로 변환하는 작업."""

import logging

from app.core.config import settings
from app.schemas.agent import GiftData
from app.schemas.vision import ExtractionResult
from app.services import product_search as product_search_module
from app.services.clock import service_today
from app.services.gift_data_policy import GiftDataPolicyError, build_gift_data
from app.services.image_loader import ImageLoadError, image_loader
from app.services.vision_response_parser import parse_extraction
from app.services.vlm_service import VisionAnalysisError, vlm_extraction_service

logger = logging.getLogger(__name__)


class ImageAnalysisService:
    """presigned URL 을 받아 VLM 으로 선물 정보를 추출하는 서비스.

    처리 단계는 셋입니다.

    1. ``image_loader``      presigned URL 다운로드, 크기·형식 검증, 장변 1280 리사이즈
    2. ``vlm_extraction``    vLLM(Gemma4-12B-QAT)에 구조화 출력을 강제해 기록 배열 추출
    3. ``gift_data_policy``  정규화된 기록 목록에서 대표 1건을 골라 ``GiftData`` 로 변환

    이미지 한 장에 여러 건이 있는 경우(계좌 거래내역, 선물함 목록, 영수증)는
    2단계까지 전부 추출하지만, 현재 ``GiftData`` 계약이 1건만 표현할 수 있어
    3단계에서 대표 1건만 남습니다. 남은 건수는 경고 로그로 남깁니다.
    """

    async def analyze(self, image_url: str) -> GiftData:
        """S3 이미지 주소를 분석해 선물명·가격·나이를 반환합니다.

        Args:
            image_url: Spring Boot가 전달한 S3 HTTP(S) 주소 또는 presigned URL.

        Returns:
            후속 네 작업이 공통으로 사용할 ``GiftData``.

        Raises:
            ImageLoadError: 이미지를 내려받거나 열지 못한 경우.
            VisionAnalysisError: vLLM 호출 또는 응답 파싱이 실패한 경우.
            GiftDataPolicyError: 기록을 하나도 찾지 못한 경우.
                셋 모두 오케스트레이터가 안전한 502 응답으로 변환합니다.
        """
        # mock 동작에서는 네트워크를 타지 않습니다. 테스트와 오프라인 개발이 그대로 돌아가야 합니다.
        image = await image_loader.load(image_url) if vlm_extraction_service.uses_real_model else None
        result = await vlm_extraction_service.extract(image)

        today = service_today()
        extraction = parse_extraction(result.payload, today)
        extraction.warnings.extend(result.warnings)
        await self._fill_missing_prices(extraction)

        build = build_gift_data(extraction)

        for warning in build.warnings:
            logger.warning("이미지 분석 주의: %s", warning)
        logger.info(
            "이미지 분석 완료 kind=%s 추출=%d건 대표=%s %s원(%s) 토큰=%d/%d",
            extraction.image_kind,
            len(extraction.records),
            build.gift_data.person_name or "이름미상",
            f"{build.gift_data.gift_price:,}" if build.gift_data.gift_price else "금액미상",
            build.price_basis.value,
            result.prompt_tokens,
            result.completion_tokens,
        )
        return build.gift_data

    @staticmethod
    async def _fill_missing_prices(extraction: ExtractionResult) -> None:
        """이미지에 금액이 없는 기록을 상품명 검색으로 채웁니다.

        카테고리 추정가는 브랜드를 모릅니다. 실측에서 TWG Tea 티백 선물이 "음료"로
        분류돼 10,000원이 됐지만 실제로는 3~7만원대였습니다. 답례 가격대가 이 값에서
        나오므로 3~7배 오차는 추천을 통째로 어긋나게 합니다.

        상품명이 있는 기록만, 여러 건이면 동시에 찾습니다. 못 찾으면 그대로 두어
        기존 카테고리 추정가로 넘어갑니다.
        """
        if not settings.product_price_lookup_enabled:
            return
        targets = [
            record
            for record in extraction.records
            if record.amount is None and (record.item_name or "").strip()
        ]
        if not targets:
            return

        found = await asyncio.gather(
            *(
                product_search_module.lookup_price(record.item_name or "", record.brand)
                for record in targets
            ),
            return_exceptions=True,
        )
        for record, price in zip(targets, found):
            if isinstance(price, int) and price > 0:
                record.amount = price
                record.price_searched = True



image_analysis_service = ImageAnalysisService()

# 오케스트레이터가 잡아 502 로 바꾸는 예외들을 한곳에서 확인할 수 있게 재노출합니다.
__all__ = [
    "ImageAnalysisService",
    "image_analysis_service",
    "ImageLoadError",
    "VisionAnalysisError",
    "GiftDataPolicyError",
]
