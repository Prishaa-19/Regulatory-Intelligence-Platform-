"""
Nexus Step 1 — Document Acquisition & Metadata Extraction.

Converts one regulator PDF (RBI, CBUAE, etc.) into three artifacts, without
interpreting the content beyond what's explicitly written in it:

    Nexus/input/<file>.pdf          copy of the source PDF (audit trail)
    Nexus/markdown/<file>.md        extracted text, lightly structured
    Nexus/metadata/<file>_metadata.json
    Nexus/logs/<file>_log.json      per-step processing log

Text extraction uses pypdf (already a project dependency) rather than a
layout-aware engine like Docling, so table/section detection is heuristic,
not structural — total_tables is reported as null rather than guessed.

Metadata fields that require reading and understanding the document
(document_id, dates, version, status, supersedes, keywords, etc.) are filled
by one LLM call over the extracted text, using the same Azure AI Foundry
endpoint as chatbot.py. The LLM is instructed to return null for anything
not explicitly stated in the text — nothing is inferred. If the LLM call
fails (missing credentials, network error), those fields are left null and
the failure is recorded in the log rather than aborting the run; the
Markdown/deterministic-metadata steps still complete.

Usage:
    python extract_document.py --pdf output/pdfs/foo.pdf
    python extract_document.py --pdf output/pdfs/foo.pdf \
        --source-url https://rbi.org.in/... --regulator RBI --country India
    python extract_document.py --pdf output/pdfs/foo.pdf --skip-llm
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

import nexus_llm
from nexus_llm import call_chat

EXTRACTION_ENGINE = "pypdf"

# Keep the LLM's context bounded; regulator circulars rarely need more than
# this to find header-level facts (id, dates, version, supersession, etc.).
MAX_LLM_CHARS = 60_000

LLM_FIELDS = [
    "document_id", "title", "regulator", "regulator_country", "source_type",
    "publication_date", "effective_date", "implementation_deadline",
    "last_updated", "version", "status", "supersedes", "superseded_by",
    "regulation_type", "regulation_number", "business_domain", "industry",
    "jurisdiction", "applicable_entities", "keywords_from_document", "language",
]

METADATA_EXTRACTION_SYSTEM_PROMPT = (
    "You extract structured metadata from a regulatory document (circular, "
    "notification, guidance, master direction, etc.). Read the document text "
    "and return ONLY facts that are explicitly stated in the text - never "
    "infer, guess, or fill in what merely seems plausible. If a field is not "
    "explicitly stated, set it to null. The field 'business_domain' must "
    "always be null unless the document explicitly labels its own business "
    "domain in those terms. Respond with a single JSON object and nothing "
    "else, using exactly these keys: " + ", ".join(LLM_FIELDS) + ". "
    "Dates must be ISO 8601 (YYYY-MM-DD) or null. applicable_entities and "
    "keywords_from_document are arrays of strings (possibly empty)."
)

_HEADING_PATTERNS = [
    re.compile(r"^(annex(ure)?|appendix|schedule|chapter|part|section)\b", re.I),
    re.compile(r"^\d+(\.\d+)*\.?\s+\S"),
    re.compile(r"^[A-Z][A-Z\s,&/()\-]{4,80}$"),
]


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    return any(p.match(stripped) for p in _HEADING_PATTERNS)


# --- Extraction-noise denoiser ----------------------------------------------
# Some regulator PDFs are bilingual or use font subsets pypdf cannot decode
# (e.g. the CBUAE circulars pair English with an Arabic column). The Arabic
# does not come out as Arabic Unicode — pypdf emits ASCII *mojibake* like
# "~1..i.ll" or "J:!.+b:iJI.J ~\", and worse, on some lines it bleeds the two
# columns together ("The objective of this Regulation is ... <....a.JU:._,l").
#
# This filter is language-AGNOSTIC on purpose (no "Arabic"/regulator special-
# casing): it keeps tokens that look like clean Latin words or plain numbers and
# drops tokens that look like decode-garbage. Working per-TOKEN (not per-line)
# is what lets a bled line keep its leading English and shed the junk tail.
# It is a lossy heuristic — a few short garbage tokens leak through and the
# occasional real token is dropped — so it is gated by --no-denoise.

_MOJIBAKE_CHARS = set("~\\|^`�{}<>_=#*")
_VOWELS = set("aeiouyAEIOUY")
_EDGE_PUNCT = ".,;:!?()[]{}'\"/-–—"


def _is_clean_token(tok: str) -> bool:
    """True if a whitespace-delimited token reads as clean English/number text."""
    if not tok or any(c in _MOJIBAKE_CHARS for c in tok) or ".." in tok:
        return False
    core = tok.strip(_EDGE_PUNCT)
    if not core:
        return False  # pure punctuation (e.g. "·", ",")
    letters = sum(c.isalpha() for c in core)
    internal_sym = sum(1 for c in core if not c.isalnum())
    if letters == 0:
        # numeric-ish: section numbers, dates, percentages (161/2018, 23/00, 25%)
        return any(c.isdigit() for c in core) and internal_sym <= 2
    if internal_sym >= 3 or letters / len(core) < 0.5:
        return False  # letters interspersed with symbols -> decode garbage
    alpha = [c for c in core if c.isalpha()]
    if len(alpha) == 1:
        return core in ("a", "A", "I", "i")  # only real one-letter English words
    if len(alpha) >= 3 and not any(c in _VOWELS for c in alpha):
        # keep short all-caps acronyms (RBI, PDF, UAE), reject vowel-less junk
        letters_only = "".join(alpha)
        return letters_only.isupper() and len(letters_only) <= 5
    return True


def denoise_pages(pages_text: list[str]) -> tuple[list[str], dict]:
    """Drop decode-garbage tokens/lines from each page, preserving paragraph
    breaks. Returns the cleaned pages and a stats dict for the processing log."""
    stats = {"lines_in": 0, "lines_dropped": 0, "tokens_in": 0, "tokens_dropped": 0}
    cleaned: list[str] = []
    for text in pages_text:
        out_lines: list[str] = []
        for line in text.splitlines():
            toks = line.split()
            stats["lines_in"] += 1
            stats["tokens_in"] += len(toks)
            if not toks:
                out_lines.append("")  # keep blank lines -> paragraph structure
                continue
            kept = [t for t in toks if _is_clean_token(t)]
            stats["tokens_dropped"] += len(toks) - len(kept)
            if kept:
                out_lines.append(" ".join(kept))
            else:
                stats["lines_dropped"] += 1  # whole line was garbage -> drop it
        cleaned.append("\n".join(out_lines))
    return cleaned, stats


def load_pdf(pdf_path: Path) -> tuple[list[str], int, PdfReader]:
    reader = PdfReader(str(pdf_path))
    pages_text = [(page.extract_text() or "").strip() for page in reader.pages]
    total_images = sum(len(page.images) for page in reader.pages)
    return pages_text, total_images, reader


def build_markdown(pages_text: list[str]) -> tuple[str, int]:
    lines_out = []
    section_count = 0
    for i, page_text in enumerate(pages_text, start=1):
        lines_out.append(f"<!-- Page {i} -->")
        for raw_line in page_text.splitlines():
            line = raw_line.rstrip()
            if _looks_like_heading(line):
                lines_out.append(f"\n## {line.strip()}\n")
                section_count += 1
            else:
                lines_out.append(line)
        lines_out.append("")
    return "\n".join(lines_out), section_count


def call_llm_metadata(document_text: str) -> dict:
    messages = [
        {"role": "system", "content": METADATA_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Document text:\n\n{document_text[:MAX_LLM_CHARS]}"},
    ]
    content = call_chat(messages)

    brace_index = content.find("{")
    if brace_index == -1:
        raise ValueError(f"LLM response had no JSON object: {content[:200]!r}")
    parsed = json.loads(content[brace_index: content.rfind("}") + 1])
    return {key: parsed.get(key) for key in LLM_FIELDS}


def extract_document(
    pdf_path: Path,
    output_dir: Path,
    source_url: str | None,
    regulator: str | None,
    regulator_country: str | None,
    source_type: str | None,
    skip_llm: bool,
    denoise: bool = True,
) -> None:
    started_at = time.monotonic()
    started_iso = datetime.now(timezone.utc).isoformat()
    log: dict = {"document": pdf_path.name, "started_at": started_iso, "steps": [], "warnings": []}

    for sub in ("input", "markdown", "metadata", "logs"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem
    input_copy = output_dir / "input" / pdf_path.name
    if input_copy.resolve() != pdf_path.resolve():
        shutil.copy2(pdf_path, input_copy)
    log["steps"].append({"step": "acquire", "status": "ok", "path": str(input_copy)})

    pdf_bytes = pdf_path.read_bytes()
    file_hash = "sha256:" + hashlib.sha256(pdf_bytes).hexdigest()

    try:
        pages_text, total_images, reader = load_pdf(pdf_path)
        log["steps"].append({"step": "pdf_load", "status": "ok", "pages": len(reader.pages)})
    except Exception as e:
        log["steps"].append({"step": "pdf_load", "status": "failed", "detail": str(e)})
        log["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_log(output_dir, stem, log)
        raise SystemExit(f"Could not read PDF: {e}")

    # Judge "scanned/OCR-needed" on the RAW extraction, before denoising removes
    # legitimately-shorter-but-clean text and skews the ratio.
    raw_full_text = "\n\n".join(pages_text)
    if len(raw_full_text) < 200 * len(pages_text):
        log["warnings"].append(
            "Extracted text is unusually short relative to page count — this PDF may be "
            "scanned/image-based and require OCR, which this pipeline does not perform."
        )

    if denoise:
        pages_text, dn_stats = denoise_pages(pages_text)
        log["steps"].append({"step": "denoise", "status": "ok", **dn_stats})
    else:
        log["steps"].append({"step": "denoise", "status": "skipped"})

    full_text = "\n\n".join(pages_text)
    markdown, section_count = build_markdown(pages_text)
    md_path = output_dir / "markdown" / f"{stem}.md"
    md_path.write_text(markdown, encoding="utf-8")
    log["steps"].append({"step": "markdown_write", "status": "ok", "path": str(md_path)})

    llm_metadata = {key: None for key in LLM_FIELDS}
    if skip_llm:
        log["steps"].append({"step": "llm_metadata_extraction", "status": "skipped"})
    else:
        load_dotenv()
        if not nexus_llm.is_configured():
            log["steps"].append({
                "step": "llm_metadata_extraction", "status": "failed",
                "detail": "No LLM provider configured (NVIDIA_* or AZURE_OPENAI_* in .env)",
            })
            log["warnings"].append("Interpreted metadata fields left null: no LLM credentials.")
        else:
            try:
                llm_metadata = call_llm_metadata(full_text)
                log["steps"].append({
                    "step": "llm_metadata_extraction", "status": "ok",
                    "providers": nexus_llm.configured_summary(),
                })
            except Exception as e:
                log["steps"].append({"step": "llm_metadata_extraction", "status": "failed", "detail": str(e)})
                log["warnings"].append(f"Interpreted metadata fields left null: {e}")

    if regulator:
        llm_metadata["regulator"] = regulator
    if regulator_country:
        llm_metadata["regulator_country"] = regulator_country
    if source_type:
        llm_metadata["source_type"] = source_type

    extraction_time = round(time.monotonic() - started_at, 2)
    extraction_status = "Success" if all(s["status"] != "failed" for s in log["steps"]) else "Partial"

    metadata = {
        # A. Source Information
        "document_id": llm_metadata["document_id"],
        "title": llm_metadata["title"],
        "regulator": llm_metadata["regulator"],
        "regulator_country": llm_metadata["regulator_country"],
        "source_type": llm_metadata["source_type"],
        "source_url": source_url,
        "source_file_name": pdf_path.name,
        "file_hash": file_hash,
        "language": llm_metadata["language"],
        "document_format": "PDF",
        # B. Publication Information
        "publication_date": llm_metadata["publication_date"],
        "effective_date": llm_metadata["effective_date"],
        "implementation_deadline": llm_metadata["implementation_deadline"],
        "last_updated": llm_metadata["last_updated"],
        "version": llm_metadata["version"],
        "status": llm_metadata["status"],
        "supersedes": llm_metadata["supersedes"],
        "superseded_by": llm_metadata["superseded_by"],
        # C. Regulatory Information
        "regulation_type": llm_metadata["regulation_type"],
        "regulation_number": llm_metadata["regulation_number"],
        "business_domain": llm_metadata["business_domain"] or "Unknown",
        "industry": llm_metadata["industry"],
        "jurisdiction": llm_metadata["jurisdiction"],
        "applicable_entities": llm_metadata["applicable_entities"],
        "keywords_from_document": llm_metadata["keywords_from_document"],
        # D. Document Statistics
        "total_pages": len(pages_text),
        "total_sections": section_count,
        "total_tables": None,
        "total_images": total_images,
        "total_words": len(full_text.split()),
        "total_characters": len(full_text),
        "detected_language": llm_metadata["language"],
        # E. Processing Information
        "extraction_time": extraction_time,
        "extraction_engine": EXTRACTION_ENGINE,
        "OCR_used": False,
        "OCR_confidence": None,
        "extraction_status": extraction_status,
        "extraction_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    metadata_path = output_dir / "metadata" / f"{stem}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log["steps"].append({"step": "metadata_write", "status": "ok", "path": str(metadata_path)})

    log["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_log(output_dir, stem, log)

    print(f"Wrote {md_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {output_dir / 'logs' / f'{stem}_log.json'}")


def _write_log(output_dir: Path, stem: str, log: dict) -> None:
    log_path = output_dir / "logs" / f"{stem}_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Markdown + metadata from a regulator PDF (Nexus Step 1).")
    parser.add_argument("--pdf", required=True, help="Path to the source PDF")
    parser.add_argument("--output", default="Nexus", help="Output directory (default: Nexus)")
    parser.add_argument("--source-url", default=None, help="Original URL the PDF was downloaded from")
    parser.add_argument("--regulator", default=None, help="Override/seed the regulator field (e.g. RBI, CBUAE)")
    parser.add_argument("--country", default=None, dest="regulator_country", help="Override/seed regulator_country")
    parser.add_argument("--source-type", default=None, help="Override/seed source_type (e.g. Circular, Notification)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM metadata extraction; interpreted fields stay null")
    parser.add_argument("--no-denoise", action="store_true", help="Keep raw pypdf text; do not strip extraction-garbage tokens (bilingual/undecodable-font mojibake)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    extract_document(
        pdf_path=pdf_path,
        output_dir=Path(args.output),
        source_url=args.source_url,
        regulator=args.regulator,
        regulator_country=args.regulator_country,
        source_type=args.source_type,
        skip_llm=args.skip_llm,
        denoise=not args.no_denoise,
    )


if __name__ == "__main__":
    main()
