"""언어모델의 텍스트 응답에서 JSON 객체를 안전하게 추출합니다."""

import json
import re
from typing import Any


class ModelResponseParseError(ValueError):
    """모델 응답이 유효한 JSON 객체가 아닐 때 발생합니다."""


def parse_json_object(text: str) -> dict[str, Any]:
    """코드 펜스를 제거하고 첫 JSON 객체를 파이썬 dict로 변환합니다."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ModelResponseParseError("모델 응답에서 JSON을 찾지 못했습니다.")
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ModelResponseParseError(
                "모델이 유효하지 않은 JSON을 반환했습니다."
            ) from exc
    if not isinstance(parsed, dict):
        raise ModelResponseParseError("모델 응답은 JSON 객체여야 합니다.")
    return parsed
