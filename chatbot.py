"""Standalone terminal chatbot using an Azure AI Foundry chat-completions endpoint.

Usage:
    python chatbot.py                # uses default model key below
    python chatbot.py --model grok-nr
"""

import argparse
import json
import os
import sys

import httpx
from dotenv import load_dotenv

from gdelt_tool import SEARCH_STORIES_TOOL_SCHEMA, compact_story, search_gdelt_stories

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(override=True)

MODELS = {
    "grok-nr": {"provider": "azure_openai_v1", "model": "grok-4-20-non-reasoning"},
}

DEFAULT_MODEL_KEY = "grok-nr"

DEBUG = os.environ.get("CHATBOT_DEBUG") == "1"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks about current events, recent "
    "news, or anything that needs up-to-date real-world information, call the "
    "search_gdelt_stories tool to look up real GDELT news before answering, and "
    "cite the specific stories/sources you used. For everything else, answer directly. "
    "When presenting structured or multi-item data (e.g. lists of stories, comparisons, "
    "or anything with consistent fields like title/date/source), format it as a "
    "GitHub-flavored markdown table rather than a bulleted list."
)

TOOLS = [SEARCH_STORIES_TOOL_SCHEMA]
MAX_TOOL_ROUNDS = 5


def _run_tool_call(tool_call: dict) -> str:
    name = tool_call["function"]["name"]
    try:
        args = json.loads(tool_call["function"].get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}

    if name != "search_gdelt_stories":
        return json.dumps({"error": f"unknown tool '{name}'"})

    try:
        stories = search_gdelt_stories(
            query=args.get("query", ""),
            days=args.get("days", 14),
            limit=args.get("limit", 10),
            sort=args.get("sort", "significance"),
        )
        compact = [compact_story(s) for s in stories]
        return json.dumps({"story_count": len(compact), "stories": compact}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def build_caller(provider: str):
    if provider == "azure_openai_v1":
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            sys.exit(
                "Missing env vars for azure_openai_v1 provider — set "
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env"
            )

        client = httpx.Client(timeout=60)

        def call_once(model_name: str, messages: list) -> dict:
            headers = {
                "Content-Type": "application/json",
                "api-key": api_key,
            }
            body = {
                "model": model_name,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
            }

            if DEBUG:
                print(f"[DEBUG] >>> POST {endpoint}")
                print(f"[DEBUG] >>> body: {body}")

            resp = client.post(endpoint, headers=headers, json=body)

            if DEBUG:
                print(f"[DEBUG] <<< status: {resp.status_code}")
                print(f"[DEBUG] <<< body: {resp.text[:2000]}")

            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]

        def call(model_name: str, messages: list) -> str:
            for _ in range(MAX_TOOL_ROUNDS):
                message = call_once(model_name, messages)
                tool_calls = message.get("tool_calls")
                if not tool_calls:
                    return message.get("content") or ""

                messages.append(message)
                for tool_call in tool_calls:
                    if DEBUG:
                        print(f"[DEBUG] tool call: {tool_call}")
                    result = _run_tool_call(tool_call)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result,
                        }
                    )

            return "(gave up after multiple tool-call rounds without a final answer)"

        return call

    sys.exit(f"Unknown provider: {provider}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=MODELS.keys())
    parser.add_argument("--system", default=None, help="Optional system prompt")
    args = parser.parse_args()

    config = MODELS[args.model]
    call = build_caller(config["provider"])
    model_name = config["model"]

    messages = [{"role": "system", "content": args.system or DEFAULT_SYSTEM_PROMPT}]

    print(f"Chatting with '{args.model}' ({model_name}). Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})

        try:
            reply = call(model_name, messages)
        except Exception as e:
            print(f"[error] {e}\n")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})
        print(f"\n{args.model}> {reply}\n")


if __name__ == "__main__":
    main()
