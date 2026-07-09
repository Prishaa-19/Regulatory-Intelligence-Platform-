"""GDELT Cloud search, exposed as an OpenAI-style callable tool.

Reuses the same MCP connection pattern as fetch_stories.py (search_stories
via the gdelt_cloud_tool_call wrapper), but as a single on-demand call
instead of a batch job.
"""

import asyncio
import json
import os

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = "https://gdelt-cloud-mcp.fastmcp.app/mcp"

SEARCH_STORIES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_gdelt_stories",
        "description": (
            "Search GDELT Cloud for real, recent news stories matching a query. "
            "Returns story records with title, date, significance score, country, "
            "and source article links. Use this whenever the user asks about current "
            "events, recent news, or real-world developments you can't be certain of "
            "from training data alone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query describing the news/topic to look for.",
                },
                "days": {
                    "type": "integer",
                    "description": "How many days back to search.",
                    "default": 14,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of stories to return.",
                    "default": 10,
                },
                "sort": {
                    "type": "string",
                    "enum": ["significance", "date"],
                    "default": "significance",
                },
            },
            "required": ["query"],
        },
    },
}


def _extract_stories(raw_content: list[dict]) -> list[dict]:
    stories: list[dict] = []
    for item in raw_content:
        text = item.get("text")
        if not text:
            continue
        brace_index = text.find("{")
        if brace_index == -1:
            continue
        try:
            payload = json.loads(text[brace_index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates = payload.get("data") or payload.get("stories") or payload.get("results")
            if isinstance(candidates, list):
                stories.extend(candidates)
        elif isinstance(payload, list):
            stories.extend(payload)
    return stories


async def _search_async(query: str, days: int, limit: int, sort: str) -> list[dict]:
    api_key = os.environ.get("GDELT_CLOUD_API_KEY")
    if not api_key:
        raise RuntimeError("GDELT_CLOUD_API_KEY is not set in .env")

    headers = {"Authorization": f"Bearer {api_key}"}
    async with streamablehttp_client(MCP_URL, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "gdelt_cloud_tool_call",
                {
                    "tool_name": "search_stories",
                    "tool_arguments": {
                        "query": query,
                        "days": days,
                        "limit": limit,
                        "sort": sort,
                    },
                },
            )
            content = [c.model_dump() for c in result.content]
            return _extract_stories(content)


def search_gdelt_stories(
    query: str, days: int = 14, limit: int = 10, sort: str = "significance"
) -> list[dict]:
    return asyncio.run(_search_async(query, days, limit, sort))


def compact_story(story: dict) -> dict:
    geo = story.get("geo") or {}
    return {
        "title": story.get("title"),
        "url": story.get("url"),
        "date": story.get("story_date"),
        "significance": (story.get("metrics") or {}).get("significance"),
        "country": geo.get("country"),
        "sources": [a.get("domain") for a in story.get("top_articles") or [] if a.get("domain")],
    }
