"""S3 선물 이미지를 공통 선물데이터로 변환하는 작업."""

import asyncio
import logging

from app.core.config import settings
from app.schemas.agent import GiftData, InputCategory
from app.schemas.vision import ExtractedRecord, ExtractionResult, RecordType
from app.services import product_search as product_search_module
from app.services.clock import service_today
from app.services.gift_data_policy import GiftDataPolicyError, build_gift_data, select_primary
from app.services.image_loader import ImageLoadError, image_loader
from app.services.vision_response_parser import parse_extraction, refresh_review_flags
from app.services.vlm_service import VisionAnalysisError, vlm_extraction_service

logger = logging.getLogger(__name__)

# 한 번의 이미지 분석에서 판매가를 검색할 최대 기록 수.
# Tavily 는 검색 1회가 1크레딧이고 월 한도가 1000 인데, 스키마상 기록은 20건까지
# 나올 수 있습니다(vision_prompt 의 maxItems). 상한이 없으면 이미지 한 장에
# 20크레딧이 나갑니다. 대표가 될 가능성이 높은 기록부터 이 수만큼만 찾습니다.
MAX_PRICE_LOOKUPS = 3

# 답례 "선물" 추천이 나올 수 있는 기록 종류. gift_agent_service.RECOMMENDABLE_KINDS
# 와 같은 기준입니다(순환 임포트를 피하려고 내부 타입으로 따로 둡니다).
# 추천을 만들지 않을 이미지에는 판매가 검색 크레딧을 쓰지 않습니다.
_RECOMMENDABLE_TYPES = frozenset({RecordType.GIFT, RecordType.EVENT_INVITATION})


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

    async def analyze(
        self,
        image_url: str,
        category: InputCategory | None = None,
    ) -> GiftData:
        """S3 이미지 주소를 분석해 선물명·가격·나이를 반환합니다.

        Args:
            image_url: Spring Boot가 전달한 S3 HTTP(S) 주소 또는 presigned URL.
            category: 사용자가 업로드 화면에서 고른 종류. 추출 프롬프트에 힌트로
                실어 보내고, 판매가 검색이 필요한지 판단하는 데도 씁니다.

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
        result = await vlm_extraction_service.extract(
            image,
            category=category.value if category else None,
        )

        today = service_today()
        extraction = parse_extraction(result.payload, today)
        extraction.warnings.extend(result.warnings)
        await self._fill_missing_prices(extraction, category)
        # 금액이 채워졌으면 "금액을 확인하지 못했습니다"는 더 이상 사실이 아니고,
        # 검색으로 채운 값이면 새로 확인이 필요합니다. 그래서 채운 뒤에 다시 봅니다.
        refresh_review_flags(extraction, today)

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

    @classmethod
    async def _fill_missing_prices(
        cls,
        extraction: ExtractionResult,
        category: InputCategory | None = None,
    ) -> None:
        """이미지에 금액이 없는 기록을 상품명 검색으로 채웁니다.

        카테고리 추정가는 브랜드를 모릅니다. 실측에서 TWG Tea 티백 선물이 "음료"로
        분류돼 10,000원이 됐지만 실제로는 3~7만원대였습니다. 답례 가격대가 이 값에서
        나오므로 3~7배 오차는 추천을 통째로 어긋나게 합니다.

        검색은 유료(Tavily 크레딧)이므로 두 가지로 제한합니다.

        - 추천을 만들지 않을 이미지(사용자가 경조사를 골랐거나, 현금·영수증뿐인
          이미지)에는 아예 검색하지 않습니다.
        - 검색하더라도 대표가 될 가능성이 높은 순서로 ``MAX_PRICE_LOOKUPS`` 건까지만.

        못 찾으면 그대로 두어 금액 미상으로 넘어갑니다.
        """
        if not settings.product_price_lookup_enabled:
            return
        if cls._skips_recommendation(extraction, category):
            logger.info("추천 대상이 아니라 판매가 검색을 건너뜁니다 kind=%s", extraction.image_kind)
            return

        targets = [
            record
            for record in extraction.records
            if record.amount is None and (record.item_name or "").strip()
        ]
        if not targets:
            return

        primary = select_primary(extraction.records)
        targets.sort(key=lambda record: cls._lookup_priority(record, primary))
        if len(targets) > MAX_PRICE_LOOKUPS:
            logger.info(
                "금액 없는 기록 %d건 중 %d건만 판매가를 검색합니다",
                len(targets),
                MAX_PRICE_LOOKUPS,
            )
            targets = targets[:MAX_PRICE_LOOKUPS]

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

    @staticmethod
    def _lookup_priority(record: ExtractedRecord, primary: ExtractedRecord | None) -> tuple:
        """검색 순서. 대표 기록, 답례 추천 대상 종류, 확신이 높은 기록 순입니다."""
        return (
            record is not primary,
            record.record_type not in _RECOMMENDABLE_TYPES,
            -record.confidence,
        )

    @staticmethod
    def _skips_recommendation(
        extraction: ExtractionResult,
        category: InputCategory | None,
    ) -> bool:
        """답례 선물 추천이 나올 수 없는 입력인지. 맞으면 판매가 검색이 무의미합니다."""
        if category is InputCategory.OCCASION:
            return True
        if category is InputCategory.GIFT:
            return False
        return not any(record.record_type in _RECOMMENDABLE_TYPES for record in extraction.records)


image_analysis_service = ImageAnalysisService()

# 오케스트레이터가 잡아 502 로 바꾸는 예외들을 한곳에서 확인할 수 있게 재노출합니다.
__all__ = [
    "ImageAnalysisService",
    "image_analysis_service",
    "ImageLoadError",
    "VisionAnalysisError",
    "GiftDataPolicyError",
    "MAX_PRICE_LOOKUPS",
]
