"""
Reusable layout for rendering RBI-document classification tables (Document /
Date / Source / Domain / Sub-domain / Clause(s) / Important Info) as both
Markdown and PDF, so this format and its column definitions don't need to be
redesigned or re-explained each time a new batch of documents is classified.

Each entry in ITEMS is a dict with these keys:
    title     - full document title
    date      - the date as printed inside the document (e.g. "Jun 18, 2026")
    source    - where the PDF came from (e.g. "RBI")
    domain    - broad regulatory area (e.g. "Foreign Exchange Management (FEMA)")
    subdomain - the specific regulatory theme within that domain
    clauses   - summary of the amending/operative clauses in the document
    info      - notification number, effective date, signatory, other key facts

The classification itself (domain/subdomain/clauses/info) still has to be
worked out by reading each document - this module only owns the *layout*:
column set, table structure, and MD/PDF rendering, so that step never has to
be reinvented. Replace EXAMPLE_ITEMS with the documents you want classified
and re-run.

Usage:
    python rbi_classification_template.py
    python rbi_classification_template.py --output output --name rbi_classification
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF
from fpdf.fonts import FontFace

REQUIRED_KEYS = ("title", "date", "source", "domain", "subdomain", "clauses", "info")
COLUMNS = ("Document", "Date", "Source", "Domain", "Sub-domain", "Clause(s)", "Important Info")

# fpdf2's core Helvetica font only supports Latin-1; map common Unicode
# punctuation/currency symbols to ASCII rather than embedding a Unicode font.
_PDF_UNICODE_REPLACEMENTS = {
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "–": "-", "—": "-",
    "…": "...",
    "•": "-",
    "₹": "Rs.",
    "\xa0": " ",
}


def _sanitize(text: str) -> str:
    for src, dst in _PDF_UNICODE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _validate(items: list[dict]) -> None:
    for i, item in enumerate(items, start=1):
        missing = [k for k in REQUIRED_KEYS if k not in item]
        if missing:
            raise ValueError(f"Item {i} is missing keys: {missing}")


def write_markdown(items: list[dict], path: Path, heading: str) -> None:
    lines = [f"# {heading}", "", "| # | " + " | ".join(COLUMNS) + " |",
             "|---" * (len(COLUMNS) + 1) + "|"]
    for i, item in enumerate(items, start=1):
        row = [str(i)] + [item[k].replace("|", "\\|") for k in REQUIRED_KEYS]
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_pdf(items: list[dict], path: Path, heading: str) -> None:
    pdf = FPDF(orientation="L", format="A4")
    pdf.set_title(_sanitize(heading))
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _sanitize(heading), new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 8)
    col_widths = (5, 55, 16, 12, 32, 40, 62, 55)
    headings_style = FontFace(emphasis="B", fill_color=(230, 230, 230))

    with pdf.table(
        col_widths=col_widths,
        text_align="LEFT",
        v_align="TOP",
        headings_style=headings_style,
        line_height=4.2,
        padding=1.5,
    ) as table:
        header = table.row()
        for col in ("#",) + COLUMNS:
            header.cell(col)

        for i, item in enumerate(items, start=1):
            row = table.row()
            row.cell(str(i))
            for key in REQUIRED_KEYS:
                row.cell(_sanitize(item[key]))

    pdf.output(str(path))


def write_classification_outputs(
    items: list[dict], output_dir: Path, name: str = "rbi_classification"
) -> tuple[Path, Path]:
    _validate(items)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    heading = f"RBI Notifications - Classification ({ts})"
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{name}_{ts}.md"
    pdf_path = output_dir / f"{name}_{ts}.pdf"
    write_markdown(items, md_path, heading)
    write_pdf(items, pdf_path, heading)
    return md_path, pdf_path


# Replace with the documents you want classified for this run.
EXAMPLE_ITEMS = [
    {
        "title": "Foreign Exchange Management (Deposit) (Sixth Amendment) Regulations, 2026",
        "date": "Jun 18, 2026",
        "source": "RBI",
        "domain": "Foreign Exchange Management (FEMA)",
        "subdomain": "Deposits - SNRR Account framework",
        "clauses": (
            "9 clauses: short title/commencement; inserts 'IFSC' definition; allows SNRR accounts "
            "via AD branches incl. IFSC; NRO/NRE/SNRR transfer limits tied to Remittance of Assets "
            "Regs, 2016; restates SNRR account purpose; deletes Schedule 4 paras 2,5,6,7,8; NRO to "
            "SNRR transfer rule; inserts new para 16 on SNRR-to-SNRR transactions between "
            "non-residents"
        ),
        "info": (
            "Notification No. FEMA 5(R)(6)/2026-RB, dated Jun 18, 2026. Amends FEMA 5(R)/2016-RB "
            "(5 prior amendments since 2018). Effective from Official Gazette publication. "
            "Signed: N Senthil Kumar, CGM."
        ),
    },
    {
        "title": "Master Direction - RBI (Credit Derivatives) Directions, 2026",
        "date": "Jun 25, 2026",
        "source": "RBI",
        "domain": "Financial Markets Regulation",
        "subdomain": "Credit Derivatives - CDS & TRS (OTC + exchange-traded)",
        "clauses": (
            "12 sections: Definitions; Eligible participants; OTC directions (market-makers/user "
            "classification, reference entities/obligations, settlement, hedging rules); "
            "Standardisation; Customer protection; Reporting; Determinations Committee; "
            "Exchange-traded CDS & futures on credit indices; Valuation methodology; Prudential "
            "norms; RBI information powers; Data dissemination; Violations"
        ),
        "info": (
            "RBI/FMRD/2026-27/407, dated Jun 25, 2026. Supersedes FMRD.DIRD.11/14.03.004/2021-22 "
            "& A.P. (DIR Series) Circular No. 23/2022. Issued under Sec 45W, RBI Act 1934. Effective "
            "immediately. FPI participation capped at 5% of outstanding corporate bond stock for CDS "
            "protection sold. Signed: Dimple Bhandia, CGM."
        ),
    },
    {
        "title": "Review of Circulars issued under FEMA, 1999",
        "date": "Jun 24, 2026",
        "source": "RBI",
        "domain": "Foreign Exchange Management (FEMA)",
        "subdomain": "Regulatory rationalisation - withdrawal of obsolete circulars",
        "clauses": (
            "Single operative clause: withdraws circulars issued since Jun 1, 2000 (listed in Annex) "
            "that are redundant, overlapping, or superseded. Issued under Sections 10(4) and 11(1), "
            "FEMA 1999."
        ),
        "info": (
            "RBI/2026-27/175, A.P. (DIR Series) Circular No. 18, dated Jun 24, 2026. Part of RBI's "
            "ongoing FEMA circular clean-up drive - companion circular to item #4 (same review "
            "exercise). Signed: Dr. Aditya Gaiha, CGM-in-Charge."
        ),
    },
    {
        "title": "Modification of Returns / Reporting requirements under FEMA, 1999",
        "date": "Jun 24, 2026",
        "source": "RBI",
        "domain": "Foreign Exchange Management (FEMA)",
        "subdomain": "Authorised Persons - Reporting & Returns (Money Changing / MTSS)",
        "clauses": (
            "6 paragraphs: rationalises reporting under FEM (Authorised Persons) Regulations, 2026 - "
            "revises FLM-8 (adds FX-note write-off detail, drops RBI pre-approval >USD 2000), adds "
            "new franchisee-list & MTSS sub-agent-list filings; discontinues FLM-1 to FLM-7 "
            "registers, one quarterly FX-account return, separate MTSS additional-locations list, "
            "and MTSS collateral return"
        ),
        "info": (
            "RBI/2026-27/174, A.P. (DIR Series) Circular No.17, dated Jun 24, 2026. Includes Annexes: "
            "revised FLM-8 format, List of Forex Correspondents, List of Franchisees, Fit & Proper "
            "criteria form. Signed: N Senthil Kumar, CGM."
        ),
    },
    {
        "title": "RBI (Rural Co-operative Banks - Responsible Business Conduct) Third Amendment Directions, 2026",
        "date": "Jun 24, 2026",
        "source": "RBI",
        "domain": "Banking Regulation - Co-operative Banking Supervision",
        "subdomain": "Customer protection - liability & compensation for fraudulent EBT",
        "clauses": (
            "Inserts ~10 new definitions (Card Present/Not Present, EBT, Fraudulent EBT, Negligence "
            "by customer/RCB, Shadow reversal, Third-party breach, Unauthorised EBT); replaces old "
            "liability section (paras 58-70) with new Chapter 'CA' (6 sub-parts: Policy, Alerts, "
            "Customer reporting, Liability determination, Small-value compensation, Board "
            "monitoring); adds Annex II(1)/II(2) claim forms"
        ),
        "info": (
            "RBI/2026-27/173, DOR.MCS.REC.No.136/01-01-038/2026-27, dated Jun 24, 2026. Applies to "
            "EBTs on/after Jan 1, 2027. Key rule: zero customer liability for bank-negligence/"
            "third-party-breach fraud (if reported within 5 days); small-value fraud (<=Rs.50,000) "
            "compensated at 85% or Rs.25,000 (whichever less), cost-shared between RBI/customer's "
            "bank/beneficiary bank. Signed: Veena Srivastava, CGM."
        ),
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a Domain/Sub-domain/Clause(s)/Date/Source classification table as MD + PDF."
    )
    parser.add_argument("--output", default="output", help="Output directory (default: output)")
    parser.add_argument("--name", default="rbi_classification", help="Output filename prefix")
    args = parser.parse_args()

    md_path, pdf_path = write_classification_outputs(EXAMPLE_ITEMS, Path(args.output), args.name)
    print(f"Wrote {md_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
