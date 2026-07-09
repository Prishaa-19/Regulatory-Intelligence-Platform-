# GDELT News Monitor — Project Context

## Overview

A local Python project that connects directly to the GDELT Cloud MCP server over HTTP (independent of Claude.ai). It pulls news story digests for configurable topics and writes timestamped JSON + Markdown summaries locally. It also includes a chatbot UI and an RBI regulatory scraper.

## Project Structure

```
gdelt-news-monitor/
├── fetch_stories.py          # Batch fetcher: calls GDELT search_stories per topic, writes output/
├── gdelt_tool.py             # Single-call GDELT search, exposed as an OpenAI-callable tool
├── chatbot.py                # Terminal chatbot (Azure/NVIDIA LLM + GDELT search tool)
├── webchat.py                # Flask web UI wrapping chatbot.py
├── rbi_scraper.py            # Scrapes RBI "What's New" page + linked docs → JSON/MD/PDF
├── topics.json               # Topic configs: name, query, days, limit, sort
├── requirements.txt          # Python deps
├── .env                      # Secrets (gitignored — see below)
├── output/                   # Timestamped digests (gitignored)
│   ├── stories_<ts>.json
│   ├── stories_<ts>.md
│   ├── rbi_whatsnew_<ts>.json
│   ├── rbi_whatsnew_<ts>.md
│   └── rbi_whatsnew_<ts>.pdf
└── connectors/gdelt_cloud/
    └── module.py             # GDELT Cloud MCP connector registration (for larger MCP framework)
```

## Key Dependencies (`requirements.txt`)

- `mcp>=1.2.0` — MCP client (Streamable HTTP transport)
- `openai>=1.30.0` — LLM calls (OpenAI-compatible endpoints: Azure, NVIDIA)
- `flask>=3.0.0` — web UI for chatbot
- `httpx>=0.27.0` — HTTP client (RBI scraper)
- `beautifulsoup4>=4.12.0` — HTML parsing (RBI scraper)
- `pypdf>=4.0.0` — PDF text extraction
- `fpdf2>=2.7.0` — PDF generation
- `python-dotenv>=1.0.1` — `.env` loading

Install: `pip install -r requirements.txt`

## Environment Variables (`.env`)

| Variable | Used by | Purpose |
|---|---|---|
| `GDELT_CLOUD_API_KEY` | `fetch_stories.py`, `gdelt_tool.py`, connector | GDELT Cloud MCP Bearer token |
| `OPENAI_API_KEY` | `fetch_stories.py` | LLM for story summarization |
| `AZURE_OPENAI_ENDPOINT` | `chatbot.py` | Azure AI Foundry chat endpoint |
| `NVIDIA_BASE_URL` | `fetch_stories.py` | NVIDIA inference base URL |
| `NVIDIA_MODEL` | `fetch_stories.py` | NVIDIA model name |

Secrets are never printed or written to output files.

## Core Entry Points

### `fetch_stories.py` — Batch digest

```bash
python fetch_stories.py
python fetch_stories.py --topics my_topics.json --output my_output_dir
```

Flow: loads `topics.json` → connects to GDELT Cloud MCP (`https://gdelt-cloud-mcp.fastmcp.app/mcp`) via `streamablehttp_client` → calls `gdelt_cloud_tool_call(tool_name="search_stories", ...)` per topic → summarizes with LLM → writes `output/stories_<ts>.json` and `output/stories_<ts>.md`.

### `rbi_scraper.py` — RBI regulatory scraper

```bash
python rbi_scraper.py
python rbi_scraper.py --limit 5 --delay 2 --output output
```

Scrapes `https://www.rbi.org.in/scripts/NewLinkDetails.aspx` (two levels deep), extracts press releases / notifications / master directions, downloads PDFs locally, and writes JSON + Markdown + a compiled PDF with TOC.

### `chatbot.py` — Terminal chatbot

```bash
python chatbot.py
python chatbot.py --model grok-nr
```

Azure/NVIDIA-backed chat loop. Automatically calls `search_gdelt_stories` tool when the user asks about current events.

### `webchat.py` — Web UI

```bash
python webchat.py
# open http://127.0.0.1:5000
```

Flask UI wrapping `chatbot.py`. Proxies messages server-side so the API key never reaches the browser.

## Topic Configuration (`topics.json`)

Array of objects, each with:

```json
{
  "name": "Display name",
  "query": "search query string",
  "days": 14,
  "limit": 15,
  "sort": "significance"   // or "date"
}
```

Add/edit topics here to change what `fetch_stories.py` monitors.

## MCP Connection Pattern

All GDELT calls use `mcp.client.streamable_http.streamablehttp_client` with a Bearer token header:

```python
headers = {"Authorization": f"Bearer {api_key}"}
async with streamablehttp_client(MCP_URL, headers=headers) as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("gdelt_cloud_tool_call", {
            "tool_name": "search_stories",
            "tool_arguments": { "query": "...", "days": 14, "limit": 15 }
        })
```

Response parsing: the tool returns text with a JSON object starting at the first `{`. The `data` (or `stories`/`results`) field holds the story array.

## Output Format

Stories are grouped by `geo.country` in Markdown. Each story entry includes title, URL, date, significance score, and top article links. The JSON output preserves the full raw story objects.

## Running Tests / Validation

No test suite currently. To validate a live run:

```bash
python fetch_stories.py --topics topics.json --output output
# Check output/stories_<latest>.md for story entries
```

## VS Code MCP Integration

`.vscode/mcp.json` is configured to connect to the GDELT Cloud MCP server. Run **MCP: List Servers** in VS Code and enter the API key when prompted (stored as a VS Code secret).

## Notes for Adding Skills

- The main async pattern for MCP calls is in `fetch_stories.py:run()` and `gdelt_tool.py:_search_async()`
- Story response parsing lives in `extract_stories()` / `_extract_stories()` in each file
- To add a new data source, follow the connector pattern in `connectors/gdelt_cloud/module.py`
- Output files are timestamped UTC (`%Y%m%dT%H%M%SZ`) and gitignored
- The `compact_story()` function in `gdelt_tool.py` normalizes story records for LLM consumption
