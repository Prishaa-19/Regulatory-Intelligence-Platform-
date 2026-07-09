"""
Scrape the RBI "What's New" page and every page it links to.

Two levels:
1. https://www.rbi.org.in/scripts/NewLinkDetails.aspx lists ~30 items (press
   releases, notifications, master directions). Each item's title + URL is
   read from its <table class="tablebg"> / <a class="link2"> listing.
2. Each of those pages is fetched and parsed for title, date, PDF link, and
   body text. Any RBI-domain document links found inside that page's body
   (e.g. "Computation of Net-worth dated January 16, 2015") are followed and
   parsed the same way, but not recursed into further.

Detail pages (press releases, notifications, master directions) all share
the same template: a <table class="tablebg"> with a PDF link, an optional
date row, a title row, and a body row containing a nested <table class="td">.

Usage:
    python rbi_scraper.py
    python rbi_scraper.py --limit 5 --delay 2 --output output
"""

import argparse
import difflib
import io
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fpdf import FPDF
from pypdf import PdfReader

WHATS_NEW_URL = "https://www.rbi.org.in/scripts/NewLinkDetails.aspx"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass
class PageData:
    url: str
    title: str | None = None
    date: str | None = None
    pdf_url: str | None = None
    pdf_size: str | None = None
    pdf_text: str | None = None
    pdf_path: str | None = None
    body_text: str = ""
    linked_documents: list[dict] = field(default_factory=list)
    error: str | None = None


def fetch(client: httpx.Client, url: str) -> str | None:
    try:
        response = client.get(url)
        response.raise_for_status()
        return response.text
    except httpx.HTTPError as e:
        print(f"  ! fetch failed for {url}: {e}")
        return None


def fetch_and_save_pdf(client: httpx.Client, pdf_url: str, pdf_dir: Path) -> tuple[str | None, str | None]:
    """Download a PDF once, save it to disk, and extract its text from the
    same bytes. Returns (pdf_text, local_path), either of which may be None
    on failure. Reuses an already-downloaded copy from a previous run instead
    of re-fetching, since the same document is often referenced repeatedly."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(pdf_url).path).name or f"{abs(hash(pdf_url))}.pdf"
    local_path = pdf_dir / filename

    if local_path.exists():
        content = local_path.read_bytes()
    else:
        try:
            response = client.get(pdf_url)
            response.raise_for_status()
        except httpx.HTTPError as e:
            print(f"  ! pdf fetch failed for {pdf_url}: {e}")
            return None, None

        content = response.content
        try:
            local_path.write_bytes(content)
        except OSError as e:
            print(f"  ! could not save pdf {pdf_url}: {e}")
            local_path = None

    pdf_text = None
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        pdf_text = "\n\n".join(p for p in pages if p) or None
    except Exception as e:
        print(f"  ! pdf parse failed for {pdf_url}: {e}")

    return pdf_text, (str(local_path) if local_path else None)


def fetch_whats_new(client: httpx.Client) -> list[dict]:
    html = fetch(client, WHATS_NEW_URL)
    if html is None:
        raise SystemExit("Could not fetch the What's New page.")

    soup = BeautifulSoup(html, "html.parser")
    items = []
    for a in soup.select("table.tablebg a.link2"):
        href = a.get("href", "").strip()
        title = a.get_text(" ", strip=True)
        if href:
            items.append({"title": title, "url": urljoin(WHATS_NEW_URL, href)})
    return items


def parse_detail_page(url: str, html: str) -> PageData:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="tablebg")
    if table is None:
        return PageData(url=url, error="no table.tablebg found on page")

    pdf_link = table.find("a", id=lambda v: v and v.startswith("APDF_"))
    pdf_url = pdf_link["href"].strip() if pdf_link and pdf_link.get("href") else None
    pdf_size = None
    if pdf_link:
        size_span = table.find("span", id=lambda v: v and v.startswith("SPDF_"))
        pdf_size = size_span.get_text(strip=True) if size_span else None

    title = None
    date = None
    for td in table.find_all("td", class_="tableheader"):
        text = td.get_text(" ", strip=True)
        if text.lower().startswith("date"):
            date = text.split(":", 1)[-1].strip()
        elif td.find("b"):
            candidate = td.find("b").get_text(" ", strip=True)
            if candidate:
                title = candidate

    content_row = table.find("tr", class_=lambda c: c and c.startswith("tablecontent"))
    body_table = content_row.find("table", class_="td") if content_row else None
    body_text = body_table.get_text("\n", strip=True) if body_table else ""

    linked_documents = []
    seen = set()
    if body_table:
        for a in body_table.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#"):
                continue
            full_url = urljoin(url, href)
            if full_url in seen:
                continue
            seen.add(full_url)
            linked_documents.append({"text": a.get_text(" ", strip=True), "url": full_url})

    return PageData(
        url=url,
        title=title,
        date=date,
        pdf_url=pdf_url,
        pdf_size=pdf_size,
        body_text=body_text,
        linked_documents=linked_documents,
    )


# Known RBI document-detail scripts. Anything else (data downloads, generic
# content pages, the data portal, etc.) is left in linked_documents but not
# followed as a level-2 page.
CRAWLABLE_PAGES = {
    "notificationuser.aspx",
    "bs_pressreleasedisplay.aspx",
    "bs_viewmasdirections.aspx",
    "bs_speechesview.aspx",
    "publicationsview.aspx",
}


def is_crawlable(url: str) -> bool:
    parsed = urlparse(url)
    if "rbi.org.in" not in parsed.netloc:
        return False
    page_name = parsed.path.rsplit("/", 1)[-1].lower()
    return page_name in CRAWLABLE_PAGES


def scrape(client: httpx.Client, delay: float, limit: int | None, pdf_dir: Path) -> list[dict]:
    whats_new = fetch_whats_new(client)
    if limit is not None:
        whats_new = whats_new[:limit]

    cache: dict[str, PageData] = {}

    def get_page(page_url: str) -> PageData:
        if page_url in cache:
            return cache[page_url]
        time.sleep(delay)
        html = fetch(client, page_url)
        page = parse_detail_page(page_url, html) if html is not None else PageData(url=page_url, error="fetch failed")
        if page.pdf_url:
            time.sleep(delay)
            page.pdf_text, page.pdf_path = fetch_and_save_pdf(client, page.pdf_url, pdf_dir)
        cache[page_url] = page
        return page

    results = []
    for i, item in enumerate(whats_new, start=1):
        print(f"[{i}/{len(whats_new)}] {item['title']}")
        detail = get_page(item["url"])

        linked_pages = []
        for link in detail.linked_documents:
            if not is_crawlable(link["url"]):
                continue
            print(f"    -> {link['text'] or link['url']}")
            linked_pages.append(get_page(link["url"]))

        results.append(
            {
                "whats_new_title": item["title"],
                "whats_new_url": item["url"],
                "detail": vars(detail),
                "linked_pages": [vars(p) for p in linked_pages],
            }
        )

    return results


# fpdf2's core fonts only support Latin-1, but RBI's pages use smart quotes,
# en/em dashes, and the rupee sign. Map the common ones to ASCII and drop
# anything else rather than embedding a Unicode font just for a few glyphs.
_PDF_UNICODE_REPLACEMENTS = {
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "–": "-", "—": "-",
    "…": "...",
    "•": "-",
    "₹": "Rs.",
    "\xa0": " ",
}


def sanitize_pdf_text(text: str | None) -> str:
    if not text:
        return ""
    for src, dst in _PDF_UNICODE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def pick_display_text(body_text: str, pdf_text: str | None) -> tuple[str, str | None]:
    """RBI's PDF is usually just a print-render of the same HTML page, so
    skip showing both copies when they're near-identical."""
    if not pdf_text:
        return body_text, None
    if not body_text:
        return pdf_text, None
    similarity = difflib.SequenceMatcher(None, body_text, pdf_text).ratio()
    if similarity > 0.85:
        return body_text, None
    return body_text, pdf_text


def render_toc(pdf: FPDF, outline: list) -> None:
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    for section in outline:
        pdf.set_font("Helvetica", "B" if section.level == 0 else "", 11 if section.level == 0 else 10)
        pdf.set_x(pdf.l_margin + section.level * 8)
        title = sanitize_pdf_text(section.name)
        pdf.cell(0, 7, f"{title}  ...  {section.page_number}", new_x="LMARGIN", new_y="NEXT")


def render_pdf_item(pdf: FPDF, page: dict, level: int) -> None:
    title = sanitize_pdf_text(page.get("title") or page["url"])
    pdf.start_section(title, level=level)

    pdf.set_font("Helvetica", "B", 16 if level == 0 else 13)
    pdf.multi_cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    if page.get("date"):
        pdf.multi_cell(0, 6, sanitize_pdf_text(f"Date: {page['date']}"), new_x="LMARGIN", new_y="NEXT")
    if page.get("pdf_url"):
        pdf.set_text_color(0, 0, 200)
        pdf.cell(0, 6, "View original PDF on rbi.org.in", link=page["pdf_url"], new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
    if page.get("pdf_path"):
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, f"Local copy: {page['pdf_path']}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
    if page.get("error"):
        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(0, 6, sanitize_pdf_text(f"Error: {page['error']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    body, extra = pick_display_text(page.get("body_text") or "", page.get("pdf_text"))
    pdf.set_font("Helvetica", "", 11)
    if body:
        pdf.multi_cell(0, 6, sanitize_pdf_text(body), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
    if extra:
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, "Additional detail found only in the PDF version:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, sanitize_pdf_text(extra), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)


def write_pdf(results: list[dict], output_dir: Path, ts: str) -> Path:
    pdf = FPDF()
    pdf.set_title(f"RBI What's New Digest - {ts}")
    pdf.set_auto_page_break(auto=True, margin=18)

    pdf.add_page()
    pdf.ln(70)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, "RBI What's New Digest", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, ts, new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.add_page()
    pdf.insert_toc_placeholder(render_toc, pages=1, allow_extra_pages=True)

    for entry in results:
        pdf.add_page()
        render_pdf_item(pdf, entry["detail"], level=0)
        for linked in entry["linked_pages"]:
            pdf.ln(4)
            render_pdf_item(pdf, linked, level=1)

    out_path = output_dir / f"rbi_whatsnew_{ts}.pdf"
    pdf.output(str(out_path))
    return out_path


def write_outputs(results: list[dict], output_dir: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"rbi_whatsnew_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    def render_page_meta(page: dict) -> list[str]:
        lines = [f"_{page['url']}_"]
        if page.get("date"):
            lines.append(f"Date: {page['date']}  ")
        if page.get("pdf_url"):
            lines.append(f"[PDF]({page['pdf_url']}) ({page.get('pdf_size') or 'size unknown'})  ")
        if page.get("pdf_path"):
            lines.append(f"Local copy: `{page['pdf_path']}`  ")
        if page.get("error"):
            lines.append(f"_Error: {page['error']}_")
        lines.append("")

        body, extra = pick_display_text(page.get("body_text") or "", page.get("pdf_text"))
        if body:
            lines.append(body)
            lines.append("")
        if extra:
            lines.append("**Additional detail found only in the PDF version:**")
            lines.append("")
            lines.append(extra)
            lines.append("")
        return lines

    md_lines = [f"# RBI What's New Digest — {ts}", ""]
    for entry in results:
        md_lines.append(f"## {entry['whats_new_title']}")
        md_lines.extend(render_page_meta(entry["detail"]))

        for page in entry["linked_pages"]:
            md_lines.append("---")
            md_lines.append("")
            md_lines.append(f"### {page.get('title') or page['url']}")
            md_lines.extend(render_page_meta(page))

    md_path = output_dir / f"rbi_whatsnew_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    pdf_path = write_pdf(results, output_dir, ts)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape RBI's What's New page and its linked documents.")
    parser.add_argument("--output", default="output", help="Output directory (default: output)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N What's New items")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between requests (default: 1.0)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    pdf_dir = output_dir / "pdfs"

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True) as client:
        results = scrape(client, delay=args.delay, limit=args.limit, pdf_dir=pdf_dir)

    write_outputs(results, output_dir)


if __name__ == "__main__":
    main()
