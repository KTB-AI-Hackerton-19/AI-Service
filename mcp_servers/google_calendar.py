"""Giftie Google Calendar MCP 서버.

왜 직접 만드는가
- 공개된 Google Calendar MCP 서버는 대부분 서버 자신이 OAuth 플로우를 돌리고
  토큰 파일 하나로 단일 계정만 다룹니다. Giftie 는 Spring Security 가 보유한
  "사용자별" access token 을 써야 하므로 그 구조로는 다중 사용자를 받을 수 없습니다.
- 그래서 access_token 을 툴 인자로 받습니다. 토큰은 로그에 남기지 않습니다.

실행

    python -m mcp_servers.google_calendar     # streamable-http, :8300/mcp

필요한 OAuth 스코프

    https://www.googleapis.com/auth/calendar.events
"""

import logging
import os
from datetime import date, datetime, time, timedelta
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("giftie.calendar-mcp")

GOOGLE_CALENDAR_API = "https://www.googleapis.com/calendar/v3"
HTTP_TIMEOUT = float(os.getenv("GOOGLE_TIMEOUT_SECONDS", "20"))

# Google 은 알림을 시작 시각 기준 "몇 분 전"으로만 받습니다. 음수는 허용되지 않습니다.
_REMINDER_MIN = 0
_REMINDER_MAX = 40_320  # 4주

# MCP Python SDK 2.0 에서 FastMCP 가 MCPServer 로 바뀌었습니다. 데코레이터 API 는 동일합니다.
mcp = MCPServer("giftie-calendar", version="0.1.0")


class CalendarApiError(RuntimeError):
    """Google Calendar API 가 오류를 반환했을 때 발생합니다."""


def _build_event_body(
    *,
    summary: str,
    description: str,
    all_day: bool,
    start_date: str,
    end_date: str | None,
    start_time: str | None,
    duration_minutes: int,
    timezone: str,
    reminders_minutes: list[int] | None,
) -> dict[str, Any]:
    """Google Calendar events 리소스 본문을 만듭니다.

    종일 일정과 시간 지정 일정은 start/end 표현이 다릅니다. 종일 일정의 ``end.date``
    는 배타적이므로 하루짜리 일정이면 다음 날을 넣어야 합니다.
    """
    body: dict[str, Any] = {"summary": summary, "description": description}

    if all_day:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date) if end_date else start
        body["start"] = {"date": start.isoformat()}
        body["end"] = {"date": (end + timedelta(days=1)).isoformat()}
    else:
        start_dt = datetime.combine(
            date.fromisoformat(start_date),
            time.fromisoformat(start_time or "10:00"),
        )
        end_dt = start_dt + timedelta(minutes=max(1, duration_minutes))
        body["start"] = {"dateTime": start_dt.isoformat(), "timeZone": timezone}
        body["end"] = {"dateTime": end_dt.isoformat(), "timeZone": timezone}

    if reminders_minutes is not None:
        overrides = [
            {"method": "popup", "minutes": minutes}
            for minutes in reminders_minutes
            if isinstance(minutes, int) and _REMINDER_MIN <= minutes <= _REMINDER_MAX
        ]
        body["reminders"] = {"useDefault": False, "overrides": overrides}

    return body


async def _call_google(
    method: str,
    path: str,
    access_token: str,
    *,
    json_body: dict | None = None,
    params: dict | None = None,
) -> dict:
    """Google Calendar API 를 호출합니다. 토큰은 예외 메시지에 넣지 않습니다."""
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.request(
            method,
            f"{GOOGLE_CALENDAR_API}{path}",
            headers=headers,
            json=json_body,
            params=params,
        )

    if response.status_code >= 400:
        logger.warning("Google Calendar API %s %s -> %s", method, path, response.status_code)
        raise CalendarApiError(f"Google Calendar API {response.status_code}: {response.text[:400]}")
    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _summarize(event: dict) -> dict:
    """Giftie 가 쓰는 필드만 추려서 돌려줍니다."""
    return {
        "event_id": event.get("id"),
        "html_link": event.get("htmlLink"),
        "summary": event.get("summary"),
        "start": event.get("start"),
        "status": event.get("status"),
    }


@mcp.tool()
async def create_event(
    access_token: str,
    summary: str,
    start_date: str,
    description: str = "",
    all_day: bool = False,
    end_date: str | None = None,
    start_time: str | None = "10:00",
    duration_minutes: int = 30,
    timezone: str = "Asia/Seoul",
    reminders_minutes: list[int] | None = None,
    calendar_id: str = "primary",
) -> dict:
    """Google Calendar 에 일정을 등록합니다.

    Args:
        access_token: 사용자 Google OAuth access token (calendar.events 스코프).
        summary: 일정 제목.
        start_date: 시작 날짜 YYYY-MM-DD.
        description: 일정 설명.
        all_day: 종일 일정 여부. False 면 start_time 기준 시간 지정 일정.
        end_date: 종일 일정의 마지막 날짜(포함). None 이면 하루짜리.
        start_time: 시간 지정 일정의 시작 시각 HH:MM.
        duration_minutes: 시간 지정 일정의 길이(분).
        timezone: IANA 타임존.
        reminders_minutes: 시작 시각 기준 몇 분 전에 알릴지. 0 은 정각, 1440 은 하루 전.
        calendar_id: 대상 캘린더. 보통 primary.

    Returns:
        등록된 일정의 event_id, html_link, summary, start, status.
    """
    body = _build_event_body(
        summary=summary,
        description=description,
        all_day=all_day,
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        duration_minutes=duration_minutes,
        timezone=timezone,
        reminders_minutes=reminders_minutes,
    )
    created = await _call_google("POST", f"/calendars/{calendar_id}/events", access_token, json_body=body)
    return _summarize(created)


@mcp.tool()
async def update_event(
    access_token: str,
    event_id: str,
    summary: str | None = None,
    description: str | None = None,
    start_date: str | None = None,
    start_time: str | None = None,
    duration_minutes: int = 30,
    all_day: bool = False,
    end_date: str | None = None,
    timezone: str = "Asia/Seoul",
    reminders_minutes: list[int] | None = None,
    calendar_id: str = "primary",
) -> dict:
    """등록된 일정을 수정합니다. 넘기지 않은 필드는 그대로 둡니다."""
    patch: dict[str, Any] = {}
    if summary is not None:
        patch["summary"] = summary
    if description is not None:
        patch["description"] = description
    if start_date is not None:
        timing = _build_event_body(
            summary=summary or "",
            description=description or "",
            all_day=all_day,
            start_date=start_date,
            end_date=end_date,
            start_time=start_time,
            duration_minutes=duration_minutes,
            timezone=timezone,
            reminders_minutes=reminders_minutes,
        )
        patch["start"] = timing["start"]
        patch["end"] = timing["end"]
        if "reminders" in timing:
            patch["reminders"] = timing["reminders"]

    updated = await _call_google(
        "PATCH", f"/calendars/{calendar_id}/events/{event_id}", access_token, json_body=patch
    )
    return _summarize(updated)


@mcp.tool()
async def get_event(access_token: str, event_id: str, calendar_id: str = "primary") -> dict:
    """등록된 일정 하나를 조회합니다. 알림이 제대로 걸렸는지 확인할 때 씁니다.

    Args:
        access_token: 사용자 Google OAuth access token.
        event_id: 조회할 일정 ID.
        calendar_id: 대상 캘린더.

    Returns:
        일정 요약 정보와 reminders 설정.
    """
    event = await _call_google("GET", f"/calendars/{calendar_id}/events/{event_id}", access_token)
    summary = _summarize(event)
    summary["reminders"] = event.get("reminders")
    summary["description"] = event.get("description")
    return summary


@mcp.tool()
async def delete_event(access_token: str, event_id: str, calendar_id: str = "primary") -> dict:
    """등록된 일정을 삭제합니다. 사용자가 승인을 철회했을 때 사용합니다."""
    await _call_google("DELETE", f"/calendars/{calendar_id}/events/{event_id}", access_token)
    return {"deleted": True, "event_id": event_id}


@mcp.tool()
async def list_events(
    access_token: str,
    time_min: str,
    time_max: str,
    query: str | None = None,
    max_results: int = 20,
    calendar_id: str = "primary",
) -> dict:
    """기간 안의 일정을 조회합니다. 같은 답례 일정을 두 번 등록하지 않으려고 확인할 때 씁니다.

    Args:
        access_token: 사용자 Google OAuth access token.
        time_min: RFC3339 시작 시각. 예: 2026-06-01T00:00:00+09:00
        time_max: RFC3339 종료 시각.
        query: 제목·설명 검색어.
        max_results: 최대 건수.
        calendar_id: 대상 캘린더.

    Returns:
        events 목록.
    """
    params: dict[str, Any] = {
        "timeMin": time_min,
        "timeMax": time_max,
        "maxResults": max_results,
        "singleEvents": "true",
        "orderBy": "startTime",
    }
    if query:
        params["q"] = query

    data = await _call_google("GET", f"/calendars/{calendar_id}/events", access_token, params=params)
    return {"events": [_summarize(item) for item in data.get("items", [])]}


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=os.getenv("CALENDAR_MCP_HOST", "0.0.0.0"),
        port=int(os.getenv("CALENDAR_MCP_PORT", "8300")),
    )
