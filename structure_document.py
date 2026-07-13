"""
Nexus Step 2 — Structural Segmentation.

Step 1 answered "what is the document and where did it come from" (Markdown +
metadata). Step 2 answers "how is the document organized": it parses the
Step-1 Markdown into a nested Document -> Part -> Section -> Subsection ->
Clause tree, with Table/Annex/Footnote nodes attached where they occur.
Nothing is semantically interpreted yet (no summarization, no obligation
extraction) — this step only recovers structure.

Approach (hybrid, matching how document-intelligence platforms do this):
  1. Deterministic pass — every heading line the Step-1 Markdown tagged
     ("## ...") is classified by numbering convention (PART/CHAPTER, plain
     "1.", decimal "1.1", lettered/roman "(a)"/"(i)", "Table"/"Annexure"/
     "Schedule", "Note"/"N.B."). This covers the vast majority of regulatory
     documents deterministically and reproducibly.
  2. LLM pass — headings that don't match any known numbering convention are
     batched into a single LLM call that assigns each one a level, using
     surrounding context. This is the only non-deterministic step, and it's
     skippable (falls back to "clause", the least structurally disruptive
     guess) via --skip-llm.
  3. Tree assembly — a stack-based pass turns the ordered, fully-classified
     heading list into a nested tree, using numbering depth (e.g. "1.1.2" is
     deeper than "1.1") to decide nesting.

Known limitation: table/footnote *boundaries* are heuristic, not
structural, since Step 1 uses pypdf (no layout/table detection) rather than
a layout-aware engine. A node is emitted at the heading line; its captured
"text" is whatever body text follows until the next heading, which may
overshoot a table's actual extent.

Usage:
    python structure_document.py --stem <file-stem>
    python structure_document.py --stem <file-stem> --nexus Nexus --skip-llm
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

import nexus_llm
from nexus_llm import call_chat

PAGE_MARKER_RE = re.compile(r"^<!--\s*Page\s+(\d+)\s*-->$")
HEADING_LINE_RE = re.compile(r"^##\s+(.*)$")

TABLE_RE = re.compile(r"^table\s*(\d+|[ivxlcdm]+)?\s*[:\-]?\s*(.*)$", re.I)
ANNEX_RE = re.compile(r"^(?:annex(?:ure)?|appendix|schedule)\s*([ivxlcdm]+|\d+)?\s*[:\-]?\s*(.*)$", re.I)
PART_RE = re.compile(r"^(?:part|chapter)\s+([ivxlcdm]+|\d+)\s*[:\-]?\s*(.*)$", re.I)
SUBSECTION_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+(.*)$")
SECTION_RE = re.compile(r"^(\d+)\.?\s+(.*)$")
CLAUSE_RE = re.compile(r"^(\(?[a-zA-Z]{1,3}\)|\(?[ivxlcdm]{1,4}\)|\d+\))\s+(.*)$")
FOOTNOTE_RE = re.compile(r"^(?:note|n\.b\.)\s*[:.]?\s+(.*)$", re.I)

LLM_LEVELS = ("part", "section", "subsection", "clause", "annex", "table", "footnote")

STRUCTURE_SYSTEM_PROMPT = (
    "You classify headings from a regulatory document into structural levels. "
    "For each numbered heading below, pick exactly one level from: "
    + ", ".join(LLM_LEVELS) + ". Use the surrounding context (previous "
    "classified heading, following text) to judge nesting depth. Respond with "
    "a single JSON array of objects {\"index\": <int>, \"type\": \"<level>\"} "
    "and nothing else, one entry per heading given, in the same order."
)


def classify_heading(text: str) -> dict | None:
    text = text.strip()
    for regex, node_type in ((TABLE_RE, "table"), (ANNEX_RE, "annex"), (PART_RE, "part")):
        m = regex.match(text)
        if m:
            return {"type": node_type, "number": m.group(1), "title": m.group(2).strip() or None}

    m = SUBSECTION_RE.match(text)
    if m:
        return {"type": "subsection", "number": m.group(1), "title": m.group(2).strip() or None}

    m = SECTION_RE.match(text)
    if m:
        return {"type": "section", "number": m.group(1), "title": m.group(2).strip() or None}

    m = CLAUSE_RE.match(text)
    if m:
        return {"type": "clause", "number": m.group(1), "title": None, "text": m.group(2).strip()}

    m = FOOTNOTE_RE.match(text)
    if m:
        return {"type": "footnote", "number": None, "title": None, "text": m.group(1).strip()}

    return None


def parse_markdown(markdown_text: str) -> list[dict]:
    """Walk the Markdown, returning an ordered list of heading events, each
    with its classification (or None if ambiguous), the raw heading text,
    the page it starts on, and the body text up to the next heading."""
    events: list[dict] = []
    current_page = 1
    pending_body: list[str] = []

    def flush_body():
        if events:
            events[-1]["body_lines"].extend(pending_body)
        pending_body.clear()

    for raw_line in markdown_text.splitlines():
        page_match = PAGE_MARKER_RE.match(raw_line.strip())
        if page_match:
            current_page = int(page_match.group(1))
            continue

        heading_match = HEADING_LINE_RE.match(raw_line)
        if heading_match:
            flush_body()
            heading_text = heading_match.group(1).strip()
            classified = classify_heading(heading_text)
            events.append({
                "raw_heading": heading_text,
                "classified": classified,
                "page_start": current_page,
                "page_end": current_page,
                "body_lines": [],
            })
            continue

        line = raw_line.strip()
        if line:
            pending_body.append(line)
        if events:
            events[-1]["page_end"] = current_page

    flush_body()
    return events


def resolve_ambiguous(events: list[dict], skip_llm: bool, log: dict) -> None:
    ambiguous = [(i, e) for i, e in enumerate(events) if e["classified"] is None]
    if not ambiguous:
        return

    if skip_llm:
        for _, e in ambiguous:
            e["classified"] = {"type": "clause", "number": None, "title": None, "text": e["raw_heading"]}
        log["warnings"].append(
            f"{len(ambiguous)} heading(s) did not match a known numbering pattern and were "
            "defaulted to 'clause' (--skip-llm was set)."
        )
        return

    load_dotenv()
    if not nexus_llm.is_configured():
        for _, e in ambiguous:
            e["classified"] = {"type": "clause", "number": None, "title": None, "text": e["raw_heading"]}
        log["warnings"].append(
            f"{len(ambiguous)} heading(s) defaulted to 'clause': no LLM provider configured "
            "(NVIDIA_* or AZURE_OPENAI_* in .env)."
        )
        return

    items = []
    for idx, (i, e) in enumerate(ambiguous):
        prev_type = events[i - 1]["classified"]["type"] if i > 0 and events[i - 1]["classified"] else None
        context = " ".join(e["body_lines"])[:150]
        items.append(f"{idx}. heading=\"{e['raw_heading']}\" prev_level={prev_type} following_text=\"{context}\"")

    messages = [
        {"role": "system", "content": STRUCTURE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(items)},
    ]
    try:
        content = call_chat(messages)
        bracket_index = content.find("[")
        resolved = json.loads(content[bracket_index: content.rfind("]") + 1])
        by_index = {item["index"]: item["type"] for item in resolved}
        for idx, (i, e) in enumerate(ambiguous):
            level = by_index.get(idx)
            if level not in LLM_LEVELS:
                level = "clause"
            e["classified"] = {"type": level, "number": None, "title": None, "text": e["raw_heading"]}
        log["steps"].append({"step": "llm_structure_classification", "status": "ok", "count": len(ambiguous)})
    except Exception as exc:
        for _, e in ambiguous:
            e["classified"] = {"type": "clause", "number": None, "title": None, "text": e["raw_heading"]}
        log["steps"].append({"step": "llm_structure_classification", "status": "failed", "detail": str(exc)})
        log["warnings"].append(f"{len(ambiguous)} heading(s) defaulted to 'clause': LLM call failed ({exc}).")


def _rank(node_type: str, number: str | None) -> int:
    if node_type == "part":
        return 1
    if node_type == "section":
        return 2
    if node_type == "subsection":
        depth = (number or "").count(".")
        return 2 + max(depth, 1)
    return 1000  # clause/table/footnote/annex: handled by attach rules below, rank unused for popping


def build_tree(events: list[dict]) -> list[dict]:
    root: list[dict] = []
    stack: list[tuple[int, dict]] = []  # (rank, node)

    for e in events:
        c = e["classified"]
        node = {
            "type": c["type"],
            "number": c.get("number"),
            "heading": c.get("title"),
            "text": c.get("text") or ("\n".join(e["body_lines"]) or None),
            "page_start": e["page_start"],
            "page_end": e["page_end"],
            "children": [],
        }
        if c["type"] not in ("clause", "table", "footnote"):
            node["text"] = "\n".join(e["body_lines"]) or None

        if c["type"] in ("part", "annex"):
            stack.clear()
            root.append(node)
            stack.append((1, node))
        elif c["type"] in ("section", "subsection"):
            r = _rank(c["type"], c.get("number"))
            while stack and stack[-1][0] >= r:
                stack.pop()
            parent = stack[-1][1]["children"] if stack else root
            parent.append(node)
            stack.append((r, node))
        elif c["type"] == "clause":
            while stack and stack[-1][1]["type"] == "clause":
                stack.pop()
            parent = stack[-1][1]["children"] if stack else root
            parent.append(node)
            stack.append((1001, node))
        else:  # table / footnote — leaves, attach under deepest open node
            parent = stack[-1][1]["children"] if stack else root
            parent.append(node)

    return root


def structure_document(stem: str, nexus_dir: Path, skip_llm: bool) -> None:
    started_at = time.monotonic()
    md_path = nexus_dir / "markdown" / f"{stem}.md"
    metadata_path = nexus_dir / "metadata" / f"{stem}_metadata.json"
    if not md_path.exists():
        raise SystemExit(f"Markdown not found: {md_path} (run extract_document.py first)")

    document_id = None
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            document_id = json.load(f).get("document_id")

    log: dict = {
        "document": stem, "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": [], "warnings": [],
    }

    markdown_text = md_path.read_text(encoding="utf-8")
    events = parse_markdown(markdown_text)
    log["steps"].append({"step": "parse_markdown", "status": "ok", "heading_count": len(events)})

    resolve_ambiguous(events, skip_llm, log)

    tree = build_tree(events)
    log["steps"].append({"step": "build_tree", "status": "ok", "top_level_nodes": len(tree)})

    (nexus_dir / "structure").mkdir(parents=True, exist_ok=True)
    structure_path = nexus_dir / "structure" / f"{stem}_structure.json"
    with open(structure_path, "w", encoding="utf-8") as f:
        json.dump({"document_id": document_id, "structure": tree}, f, indent=2, ensure_ascii=False)
    log["steps"].append({"step": "structure_write", "status": "ok", "path": str(structure_path)})

    log["finished_at"] = datetime.now(timezone.utc).isoformat()
    log["duration_seconds"] = round(time.monotonic() - started_at, 2)

    (nexus_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = nexus_dir / "logs" / f"{stem}_structure_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"Wrote {structure_path}")
    print(f"Wrote {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Step-1 Markdown into a structural JSON tree (Nexus Step 2).")
    parser.add_argument("--stem", required=True, help="File stem shared by Nexus/markdown/<stem>.md etc. (e.g. the PDF filename without extension)")
    parser.add_argument("--nexus", default="Nexus", help="Nexus root directory (default: Nexus)")
    parser.add_argument("--skip-llm", action="store_true", help="Default ambiguous headings to 'clause' instead of calling the LLM")
    args = parser.parse_args()

    structure_document(stem=args.stem, nexus_dir=Path(args.nexus), skip_llm=args.skip_llm)


if __name__ == "__main__":
    main()
