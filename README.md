# GDELT News Monitor (private, local)

A standalone project that talks to the GDELT Cloud MCP server directly over
HTTP — independent of the Claude.ai web/chat interface. It pulls story
digests for configurable topics and writes JSON + Markdown summaries locally.

This project is **not** connected to any git remote and should stay local
until you explicitly decide to push it somewhere.

## What it does

Connects to `https://gdelt-cloud-mcp.fastmcp.app/mcp`, calls the GDELT Cloud
`search_stories` tool (via the `gdelt_cloud_tool_call` wrapper) for each
topic listed in [topics.json](topics.json), merges the results, and writes:

- `output/stories_<timestamp>.json` — structured, full results
- `output/stories_<timestamp>.md` — human-readable summary grouped by
  country/region

Default topics (edit or replace `topics.json`, or pass `--topics` to use a
different file):

1. New education policy announcement / government schools reform
2. AI curriculum / teacher training / AI in the classroom

## Setup

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Provide your GDELT Cloud API key. Copy `.env.example` to `.env` and fill
   in the value — `.env` is gitignored and never read into any committed
   file:

   ```
   cp .env.example .env
   # then edit .env and set GDELT_CLOUD_API_KEY=<your key>
   ```

3. (Optional, for VS Code MCP integration / Copilot agent mode) Open this
   folder in VS Code and run **MCP: List Servers** — it will prompt you for
   the API key (stored as a VS Code secret input, not written to disk) and
   connect using [.vscode/mcp.json](.vscode/mcp.json).

## Running the script

```
python fetch_stories.py
```

Optional flags:

```
python fetch_stories.py --topics my_topics.json --output my_output_dir
```

## Notes

- The API key is only ever read from the `GDELT_CLOUD_API_KEY` environment
  variable (via `.env`, loaded with `python-dotenv`). It is never printed or
  written to any output file.
- `output/` is gitignored by default (except a `.gitkeep` placeholder) since
  digests are timestamped, regenerable artifacts.
- The Markdown grouping logic in `fetch_stories.py` (`extract_stories` /
  `group_by_region`) is a best-effort parse of the tool response. Once you've
  made a live call and can see the actual `search_stories` response shape,
  tell me and I'll tighten the field names it looks for (title/url/summary/
  country) to match exactly.
