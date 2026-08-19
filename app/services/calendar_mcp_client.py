"""Google Calendar MCP 서버(:8300)에 붙는 클라이언트.

MCP Python SDK 2.0 의 고수준 ``Client`` 를 씁니다. URL 문자열을 넘기면 Streamable HTTP 로,
서버 인스턴스를 넘기면 인메모리로 붙습니다. 후자 덕분에 네트워크 없이 테스트할 수 있습니다.

세션을 요청마다 새로 엽니다. 사용자별 access token 을 툴 인자로 넘기기 때문에
세션을 재사용하면 서로 다른 사용자의 요청이 섞일 여지가 생깁니다.
"""

import json
import logging
from typing import Any

from mcp import Client

from app.core.config import settings

logger = logging.getLogger(__name__)


class CalendarMcpError(RuntimeError):
    """MCP 서버 연결 또는 툴 호출이 실패했을 때 발생합니다."""


def _text_of(result: Any) -> str:
    """CallToolResult 의 텍스트 블록들을 이어 붙입니다."""
    parts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _unwrap(result: Any) -> dict:
    """CallToolResult 에서 dict 결과를 꺼냅니다.

    Args:
        result: ``Client.call_tool`` 이 돌려준 결과.

    Returns:
        툴이 반환한 dict.

    Raises:
        CalendarMcpError: 툴이 오류를 반환했거나 응답을 해석할 수 없는 경우.
    """
    if getattr(result, "is_error", False):
        raise CalendarMcpError(_text_of(result) or "MCP 툴이 오류를 반환했습니다.")

    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        # dict 반환값이 {"result": {...}} 로 한 겹 감싸여 오는 경우를 벗겨 냅니다.
        inner = structured.get("result")
        if set(structured) == {"result"} and isinstance(inner, dict):
            return inner
        return structured

    text = _text_of(result)
    if not text:
        raise CalendarMcpError("MCP 응답이 비어 있습니다.")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CalendarMcpError(f"MCP 응답을 파싱할 수 없습니다: {text[:200]}") from exc
    if not isinstance(parsed, dict):
        raise CalendarMcpError("MCP 응답이 JSON 객체가 아닙니다.")
    return parsed


class CalendarMcpClient:
    """MCP 프로토콜로 Google Calendar 툴을 호출합니다."""

    def __init__(self, server: Any = None) -> None:
        """클라이언트를 만듭니다.

        Args:
            server: 인메모리로 붙을 ``MCPServer`` 인스턴스. ``None`` 이면 설정된 URL 로 붙습니다.
                테스트에서 실제 서버 객체를 넘겨 네트워크 없이 검증할 때 사용합니다.
        """
        self._server = server

    def _target(self) -> Any:
        return self._server if self._server is not None else settings.calendar_mcp_url

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """MCP 서버의 툴 하나를 호출합니다.

        Args:
            name: 툴 이름. create_event / update_event / delete_event / list_events.
            arguments: 툴 인자. access_token 을 포함합니다.

        Returns:
            툴이 돌려준 dict.

        Raises:
            CalendarMcpError: 연결, 초기화 또는 호출이 실패한 경우.
        """
        try:
            async with Client(
                self._target(),
                read_timeout_seconds=settings.calendar_mcp_timeout_seconds,
            ) as client:
                result = await client.call_tool(name, arguments)
        except CalendarMcpError:
            raise
        except Exception as exc:
            raise CalendarMcpError(
                f"MCP 서버에 연결할 수 없습니다({settings.calendar_mcp_url}): {exc}"
            ) from exc
        return _unwrap(result)

    async def create_event(
        self,
        *,
        access_token: str,
        summary: str,
        start_date: str,
        description: str = "",
        start_time: str = "10:00",
        duration_minutes: int = 30,
        reminders_minutes: list[int] | None = None,
        calendar_id: str | None = None,
    ) -> dict:
        """답례 준비 일정을 등록합니다.

        Returns:
            event_id, html_link, summary, start, status 를 담은 dict.
        """
        return await self.call_tool(
            "create_event",
            {
                "access_token": access_token,
                "summary": summary,
                "description": description,
                "start_date": start_date,
                "start_time": start_time,
                "duration_minutes": duration_minutes,
                "all_day": False,
                "timezone": settings.default_timezone,
                "reminders_minutes": reminders_minutes if reminders_minutes is not None else [0, 24 * 60],
                "calendar_id": calendar_id or settings.google_calendar_id,
            },
        )

    async def list_tool_names(self) -> list[str]:
        """헬스체크용. MCP 서버가 노출하는 툴 이름을 돌려줍니다."""
        try:
            async with Client(
                self._target(),
                read_timeout_seconds=settings.calendar_mcp_timeout_seconds,
            ) as client:
                listed = await client.list_tools()
        except Exception as exc:
            raise CalendarMcpError(
                f"MCP 서버에 연결할 수 없습니다({settings.calendar_mcp_url}): {exc}"
            ) from exc
        return [tool.name for tool in listed.tools]


calendar_mcp_client = CalendarMcpClient()
