"""
Temporary variant: download PDFs directly from RBI's Notifications listing
page (https://www.rbi.org.in/scripts/NotificationUser.aspx) instead of the
What's New page.

The listing page already exposes each item's date, title, and PDF link
inline in one <table class="tablebg"> — no need to visit each item's own
detail page just to find the PDF. Downloads only the first N items in
listing order (default 5) into output/pdfs/. No JSON/MD/report is produced,
just the raw PDF files on disk.

Usage:
    python rbi_download_notifications.py
    python rbi_download_notifications.py --limit 5 --delay 1 --output output
"""

import argparse
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from rbi_scraper import USER_AGENT, fetch, fetch_and_save_pdf

NOTIFICATIONS_URL = "https://www.rbi.org.in/scripts/NotificationUser.aspx"


def list_notifications(client: httpx.Client) -> list[dict]:
    html = fetch(client, NOTIFICATIONS_URL)
    if html is None:
        raise SystemExit("Could not fetch the Notifications page.")

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="tablebg")
    if table is None:
        raise SystemExit("Could not find the notifications table on the page.")

    items = []
    current_date = None
    for row in table.find_all("tr"):
        date_cell = row.find("td", class_="tableheader")
        if date_cell:
            current_date = date_cell.get_text(strip=True)
            continue

        title_link = row.find("a", class_="link2")
        pdf_link = row.find("a", id=lambda v: v and v.startswith("APDF_"))
        if not title_link or not pdf_link:
            continue

        size_span = row.find("span", id=lambda v: v and v.startswith("SPDF_"))
        items.append(
            {
                "date": current_date,
                "title": title_link.get_text(strip=True),
                "pdf_url": pdf_link["href"].strip(),
                "pdf_size": size_span.get_text(strip=True) if size_span else None,
            }
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the first N PDFs from RBI's Notifications listing page.")
    parser.add_argument("--limit", type=int, default=5, help="Number of PDFs to download (default: 5)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between downloads (default: 1.0)")
    parser.add_argument("--output", default="output", help="Output directory (default: output)")
    args = parser.parse_args()

    pdf_dir = Path(args.output) / "pdfs"

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True) as client:
        items = list_notifications(client)[: args.limit]
        for i, item in enumerate(items, start=1):
            print(f"[{i}/{len(items)}] {item['date']} — {item['title']} ({item['pdf_size']})")
            if i > 1:
                time.sleep(args.delay)
            _, local_path = fetch_and_save_pdf(client, item["pdf_url"], pdf_dir)
            print(f"    saved to {local_path}")


if __name__ == "__main__":
    main()
