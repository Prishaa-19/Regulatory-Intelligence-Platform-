"""
Fetch GDELT Cloud story digests for a configurable set of topics.

Connects directly to the GDELT Cloud MCP server over HTTP (independent of
Claude.ai), calls search_stories for each configured topic via the
gdelt_cloud_tool_call wrapper, and writes a timestamped JSON + Markdown
digest to the output/ folder.

Usage:
    python fetch_stories.py
    python fetch_stories.py --topics my_topics.json --output output
"""

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import OpenAI

MCP_URL = "https://gdelt-cloud-mcp.fastmcp.app/mcp"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"

SUMMARY_SYSTEM_PROMPT = (
    "You are a news analyst summarizing GDELT story data. Write a 3-5 sentence "
    "summary of the key trend(s) in the provided stories for this topic. Base your "
    "summary strictly on the story records given — do not invent facts, sources, or "
    "events not present in the data. Note if multiple stories cover the same "
    "underlying event."
)


def load_topics(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def fetch_topic(session: ClientSession, topic: dict) -> dict:
    result = await session.call_tool(
        "gdelt_cloud_tool_call",
        {
            "tool_name": "search_stories",
            "tool_arguments": {
                "query": topic["query"],
                "days": topic.get("days", 14),
                "limit": topic.get("limit", 15),
                "sort": topic.get("sort", "significance"),
            },
        },
    )
    content = [c.model_dump() for c in result.content]
    return {"topic": topic, "raw_content": content}


async def run(topics_path: Path, output_dir: Path) -> None:
    api_key = os.environ.get("GDELT_CLOUD_API_KEY")
    if not api_key:
        raise SystemExit(
            "GDELT_CLOUD_API_KEY is not set. Add it to a local .env file "
            "(see .env.example) or export it in your shell before running."
        )

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to a local .env file "
            "(see .env.example) or export it in your shell before running."
        )

    topics = load_topics(topics_path)
    headers = {"Authorization": f"Bearer {api_key}"}

    async with streamablehttp_client(MCP_URL, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            results = []
            for topic in topics:
                print(f"Fetching: {topic['name']}...")
                results.append(await fetch_topic(session, topic))

    write_outputs(results, output_dir, openai_api_key)


def extract_stories(raw_content: list[dict]) -> list[dict]:
    """Extract story records from tool result content.

    The text content is a human-readable preamble followed by an embedded
    JSON object (fields: success, sort, pagination, data). This strips the
    preamble by locating the first '{' and parses from there.
    """
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


def _story_for_summary(story: dict) -> dict:
    geo = story.get("geo") or {}
    return {
        "title": story.get("title"),
        "story_date": story.get("story_date"),
        "significance": (story.get("metrics") or {}).get("significance"),
        "country": geo.get("country"),
        "sources": [a.get("domain") for a in story.get("top_articles") or [] if a.get("domain")],
    }


def summarize_topic(client: OpenAI, topic_name: str, stories: list[dict]) -> str:
    """One-shot, non-streaming summary of a topic's stories via the OpenAI API.

    Grounded strictly in the provided story records (see SUMMARY_SYSTEM_PROMPT).
    """
    if not stories:
        return "No stories were returned for this topic."

    payload = json.dumps([_story_for_summary(s) for s in stories], indent=2, ensure_ascii=False)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Topic: {topic_name}\n\nStory records (JSON):\n{payload}"},
        ],
    )
    return response.choices[0].message.content or ""


def group_by_region(stories: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for story in stories:
        geo = story.get("geo") or {}
        region = geo.get("country") or geo.get("region") or geo.get("admin1") or geo.get("location") or "Unknown"
        groups.setdefault(region, []).append(story)
    return groups


def write_outputs(results: list[dict], output_dir: Path, openai_api_key: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_client = OpenAI(api_key=openai_api_key)

    structured = []
    for entry in results:
        stories = extract_stories(entry["raw_content"])
        print(f"Summarizing: {entry['topic']['name']}...")
        summary = summarize_topic(llm_client, entry["topic"]["name"], stories)
        structured.append(
            {
                "topic": entry["topic"],
                "story_count": len(stories),
                "summary": summary,
                "stories": stories,
            }
        )

    json_path = output_dir / f"stories_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)

    md_lines = [f"# GDELT Story Digest — {ts}", ""]
    for entry in structured:
        topic = entry["topic"]
        md_lines.append(f"## {topic['name']}")
        md_lines.append(f"_Query: {topic['query']}_  ")
        md_lines.append(f"_{entry['story_count']} stories_")
        md_lines.append("")
        md_lines.append(entry["summary"])
        md_lines.append("")

        groups = group_by_region(entry["stories"])
        for region in sorted(groups):
            md_lines.append(f"### {region}")
            for story in groups[region]:
                title = story.get("title") or "(untitled)"
                url = story.get("url") or ""
                story_date = story.get("story_date") or ""
                significance = (story.get("metrics") or {}).get("significance")
                line = f"- **{title}**"
                if url:
                    line += f" ([story]({url}))"
                meta_bits = [b for b in [story_date, f"significance {significance}" if significance is not None else None] if b]
                if meta_bits:
                    line += f" — {', '.join(meta_bits)}"
                md_lines.append(line)
                seen_article_urls: set[str] = set()
                for article in story.get("top_articles") or []:
                    a_title = article.get("title") or ""
                    a_url = article.get("url") or ""
                    a_domain = article.get("domain") or ""
                    if a_url and a_url not in seen_article_urls:
                        seen_article_urls.add(a_url)
                        md_lines.append(f"  - [{a_title}]({a_url}) ({a_domain})")
            md_lines.append("")

        if not groups:
            md_lines.append("_No stories returned or response shape did not match expected fields "
                             "— see JSON output for raw content._")
            md_lines.append("")

    md_path = output_dir / f"stories_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GDELT Cloud story digests for configured topics.")
    parser.add_argument("--topics", default="topics.json", help="Path to topics config file (default: topics.json)")
    parser.add_argument("--output", default="output", help="Output directory (default: output)")
    args = parser.parse_args()

    load_dotenv()
    asyncio.run(run(Path(args.topics), Path(args.output)))


if __name__ == "__main__":
    main()
