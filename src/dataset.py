import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Tuple

import tiktoken
from sec_edgar_downloader import Downloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# MD&A regex — operates on PLAIN TEXT (after HTML stripping)
# 10-Q: Item 2 → Item 3   |   10-K: Item 7 → Item 8
# Updated to handle typographic quotes, hyphens, and colons
MDA_REGEX = re.compile(
    r"(?:Item|ITEM)\s*[27][A-Z]?(?:[\.\-\:]|\s)*"
    r"Management(?:['’`]?s|['’`]?\s*s|s)?\s+Discussion\s+and\s+Analysis"
    r".*?"
    r"(?=(?:Item|ITEM)\s*[38][A-Z]?(?:[\.\-\:]|\s))",
    re.IGNORECASE | re.DOTALL,
)


# ── SGML envelope parser ─────────────────────────────────────────────────────
DOCUMENT_BLOCK_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.IGNORECASE | re.DOTALL)
TYPE_RE            = re.compile(r"<TYPE>\s*(\S+)",              re.IGNORECASE)

# FIX 1: Match </TEXT> only when it appears at the start of a line (standalone
# SGML tag), so we never terminate early on a </text> inside inline HTML/XBRL.
TEXT_RE = re.compile(r"<TEXT>(.*?)(?:^</TEXT>|\Z)", re.IGNORECASE | re.DOTALL | re.MULTILINE)


def extract_primary_document(raw: str) -> str:
    """
    Parse the EDGAR SGML envelope and return the <TEXT> content of the
    primary 10-Q (or 10-K) block.  Falls back to the first <TEXT> block
    found if no explicit filing-type match exists.
    """
    first_text = ""
    for doc_match in DOCUMENT_BLOCK_RE.finditer(raw):
        block    = doc_match.group(1)
        type_m   = TYPE_RE.search(block)
        doc_type = type_m.group(1).upper().strip() if type_m else ""

        text_m = TEXT_RE.search(block)
        if not text_m:
            continue
        content = text_m.group(1)

        if not first_text:
            first_text = content

        if doc_type in ("10-Q"):
            return content          # prefer the actual filing over exhibits

    return first_text               # fallback


# ── HTML stripper ────────────────────────────────────────────────────────────

def strip_html(raw: str) -> str:
    """Remove HTML/XML tags and decode common entities (including numeric ones)."""
    # Remove script/style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    # Remove all tags
    text = re.sub(r"<[^>]+>", " ", text)

    # FIX 2: Decode named AND numeric HTML entities.
    # Numeric entities (&#160; &#8217; etc.) are very common in iXBRL filings.
    named_entities = {
        "&amp;":   "&",  "&lt;":    "<",  "&gt;":    ">",
        "&nbsp;":  " ",  "&quot;":  '"',  "&#39;":   "'",
        "&ldquo;": '"',  "&rdquo;": '"',  "&lsquo;": "'",
        # FIX: curly apostrophe — critical for "Management&#8217;s Discussion"
        "&rsquo;": "'",  "&mdash;": "—",  "&ndash;": "–",
    }
    for ent, char in named_entities.items():
        text = text.replace(ent, char)

    # Decode remaining decimal numeric entities: &#NNN;
    text = re.sub(
        r"&#(\d+);",
        lambda m: chr(int(m.group(1))) if int(m.group(1)) < 0x110000 else " ",
        text,
    )
    # Decode hex numeric entities: &#xHH;
    text = re.sub(
        r"&#x([0-9A-Fa-f]+);",
        lambda m: chr(int(m.group(1), 16)) if int(m.group(1), 16) < 0x110000 else " ",
        text,
    )

    # FIX 3: Replace non-breaking spaces (\xa0) and other Unicode spaces with
    # regular ASCII space so \s+ in the regex matches them correctly.
    text = re.sub(r"[\xa0\u2009\u200a\u202f\u205f\u3000]", " ", text)

    # Collapse runs of spaces/tabs; normalise blank lines
    text = re.sub(r"[ \t]+",  " ",  text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("—", "-").replace("–", "-")
    
    return text.strip()


# ── MD&A extractor ───────────────────────────────────────────────────────────

# def extract_mda_section(raw_file_content: str, file_path: str) -> str:
#     """
#     Full pipeline:
#       1. Unwrap SGML envelope (full-submission.txt)  → primary document
#       2. Strip HTML                                  → plain text
#       3. Regex match                                 → MD&A text
#     """
#     # Step 1 – unwrap envelope if needed
#     if "<DOCUMENT>" in raw_file_content[:4000].upper():
#         primary = extract_primary_document(raw_file_content)
#         if not primary:
#             logger.debug(f"  No <TEXT> block found in envelope: {file_path}")
#             return ""
#     else:
#         primary = raw_file_content      # already a bare HTML / plain-text file

#     # Step 2 – strip HTML
#     plain = strip_html(primary)

#     # Step 3 – primary regex (bounded: stops at next Item header)
#     match = MDA_REGEX.search(plain)
#     if match:
#         logger.debug(f"  MD&A matched (primary regex): {file_path}")
#         return match.group(0).strip()

#     # Step 4 – loose fallback: grab from MD&A header to end, cap at 15 000 chars
#     loose = re.compile(
#         r"(?:Item|ITEM)\s*[27][A-Z]?\.?\s*Management(?:'s|'\s*s|s)?\s+Discussion\s+and\s+Analysis.*",
#         re.IGNORECASE | re.DOTALL,
#     )
#     m2 = loose.search(plain)
#     if m2:
#         logger.debug(f"  MD&A matched (loose fallback, truncated): {file_path}")
#         return m2.group(0)[:15_000].strip()

#     logger.debug(f"  No MD&A match in plain text: {file_path}")
#     return ""

# def extract_mda_section(raw_file_content: str, file_path: str) -> str:
#     """
#     Modified with debug statements and text-dumping to find the failure point.
#     """
#     # Step 1 – unwrap envelope if needed
#     if "<DOCUMENT>" in raw_file_content[:4000].upper():
#         print("Unwrapping envelope")
#         primary = extract_primary_document(raw_file_content)
#         if not primary:
#             logger.error(f"  DEBUG FAIL: <TEXT> block missing or empty for {file_path}")
#             return ""
#     else:
#         primary = raw_file_content
#         logger.debug("  DEBUG: No <DOCUMENT> tag found, using raw content.")

#     logger.info(f"  DEBUG: Length of primary document: {len(primary)} characters")

#     # Step 2 – strip HTML
#     plain = strip_html(primary)
#     logger.info(f"  DEBUG: Length of stripped plain text: {len(plain)} characters")

#     # --- NEW DEBUG STEP: Save the plain text to a file so you can inspect it ---
#     debug_dump_path = file_path + ".debug_plain.txt"
#     try:
#         with open(debug_dump_path, "w", encoding="utf-8") as df:
#             df.write(plain)
#         logger.info(f"  DEBUG: Wrote pre-regex plain text to {debug_dump_path}")
#     except Exception as e:
#         logger.error(f"  DEBUG: Could not write debug file: {e}")
#     # ---------------------------------------------------------------------------

#     # # Step 3 – primary regex
#     # match = MDA_REGEX.search(plain)
#     # if match:
#     #     logger.debug(f"  MD&A matched (primary regex): {file_path}")
#     #     return match.group(0).strip()

#     # # Step 4 – loose fallback
#     # loose = re.compile(
#     #     r"(?:Item|ITEM)\s*[27][A-Z]?\.?\s*Management(?:'s|'\s*s|s)?\s+Discussion\s+and\s+Analysis.*",
#     #     re.IGNORECASE | re.DOTALL,
#     # )
#     # m2 = loose.search(plain)
#     # if m2:
#     #     logger.debug(f"  MD&A matched (loose fallback, truncated): {file_path}")
#     #     return m2.group(0)[:15_000].strip()
#     # Step 3 – primary regex (Find all, keep the longest to bypass Table of Contents)
#     matches = list(MDA_REGEX.finditer(plain))
#     if matches:
#         longest_match = max(matches, key=lambda m: len(m.group(0)))
#         logger.debug(f"  MD&A matched (primary regex, {len(matches)} found, picking longest): {file_path}")
#         return longest_match.group(0).strip()

#     # Step 4 – loose fallback
#     loose = re.compile(
#         r"(?:Item|ITEM)\s*[27][A-Z]?(?:[\.\-\:]|\s)*Management(?:['’`]?s|['’`]?\s*s|s)?\s+Discussion\s+and\s+Analysis.*",
#         re.IGNORECASE | re.DOTALL,
#     )
#     m2 = loose.search(plain)
#     if m2:
#         logger.debug(f"  MD&A matched (loose fallback, truncated): {file_path}")
#         return m2.group(0)[:15_000].strip()

#     # --- NEW DEBUG STEP: Search to see if the phrase exists at all ---
#     test_search = re.search(r"Management.{0,10}Discussion", plain, re.IGNORECASE)
#     if test_search:
#         logger.warning(f"  DEBUG: Found '{test_search.group(0)}' but regex failed to capture.")
#     else:
#         logger.warning(f"  DEBUG: The phrase 'Management's Discussion' is completely missing from the plain text!")
        
#     logger.info(f"  SKIP (no MD&A match): {file_path}")
#     return ""
def extract_mda_section(raw_file_content: str, file_path: str) -> str:
    """
    Full pipeline:
      1. Unwrap SGML envelope
      2. Strip HTML
      3. Regex match (with length sanity checks)
    """
    # Step 1 – unwrap envelope
    if "<DOCUMENT>" in raw_file_content[:4000].upper():
        primary = extract_primary_document(raw_file_content)
        if not primary:
            return ""
    else:
        primary = raw_file_content

    # Step 2 – strip HTML
    plain = strip_html(primary)

    # Step 3 – primary regex
    matches = list(MDA_REGEX.finditer(plain))
    if matches:
        longest_match = max(matches, key=lambda m: len(m.group(0))).group(0)
        
        # Prevent the ToC trap: Only accept the strict match if it looks like actual body text
        if len(longest_match.split()) > 1000:
            logger.debug(f"  MD&A matched (primary regex): {file_path}")
            return longest_match.strip()
        else:
            logger.debug(f"  Strict match too short ({len(longest_match.split())} words). Forcing loose fallback.")

    # Step 4 – loose fallback search (avoiding slow regex)
    fallback_pattern = r"Management(?:['’`]?s|['’`]?\s*s|s)?\s+Discussion\s+and\s+Analysis"
    matches = list(re.finditer(fallback_pattern, plain, re.IGNORECASE))
    
    if matches:
        best_chunk = ""
        for m in matches:
            start_idx = m.end()
            # The next section is generally "Quantitative and Qualitative Disclosures About Market Risk"
            end_idx = re.search(r"Quantitative\s+and\s+Qualitative\s+Disclosures", plain[start_idx:], re.IGNORECASE)
            
            if end_idx:
                chunk = plain[start_idx : start_idx + end_idx.start()].strip()
            else:
                chunk = plain[start_idx : start_idx + 150_000].strip()
                
            if len(chunk) > len(best_chunk):
                best_chunk = chunk
                
        if len(best_chunk.split()) > 500:
            logger.debug(f"  MD&A matched (heuristic fallback): {file_path}")
            return best_chunk

    return ""
# ── Complexity ───────────────────────────────────────────────────────────────

def calculate_complexity_phi(text: str, total_tokens: int) -> Tuple[float, int]:
    """Φ(d) = verifiable numeric claim count / total token count."""
    if total_tokens == 0:
        return 0.0, 0
    num_claims = len(re.findall(
        r"\$?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion|%|percent)?",
        text, re.IGNORECASE,
    ))
    return num_claims / total_tokens, num_claims


# ── Path helper ──────────────────────────────────────────────────────────────

def get_ticker_and_doc_from_path(file_path: Path) -> Tuple[str, str]:
    parts = file_path.parts
    # Typical path: DATA_DIR/sec-edgar-filings/TICKER/DOC_TYPE/ACCESSION/full-submission.txt
    # We look for common SEC types as markers
    sec_types = {"10-Q"}
    for i, part in enumerate(parts):
        up = part.upper()
        if up in sec_types:
            ticker = parts[i - 1] if i > 0 else "UNKNOWN"
            return ticker, up
    return "UNKNOWN", "UNKNOWN"


# ── Main processing ──────────────────────────────────────────────────────────

def process_filings(download_dir: str, output_file: str, tickers: list = None, doc_types: list = None, min_words: int = 2000, year: int = None, seed_tokens: int = 200):
    dl_path = Path(download_dir)
    if not dl_path.exists():
        logger.error(f"Download directory {dl_path} does not exist.")
        return

    encoder    = tiktoken.get_encoding("cl100k_base")
    valid_docs = []
    total_seen = 0

    logger.info(f"Walking {dl_path} for SEC filings…")

    for root, dirs, files in os.walk(dl_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext not in (".html", ".htm", ".txt"):
                continue

            file_path  = Path(root) / file
            
            if year is not None:
                acc_num = file_path.parent.name
                parts = acc_num.split('-')
                if len(parts) >= 2 and parts[1].isdigit():
                    yy = int(parts[1])
                    file_year = 2000 + yy if yy < 50 else 1900 + yy
                    if file_year != year:
                        continue

            total_seen += 1
            ticker, doc_type = get_ticker_and_doc_from_path(file_path)

            if tickers and ticker.upper() not in [t.upper() for t in tickers]:
                continue
            if doc_types and doc_type.upper() not in [d.upper() for d in doc_types]:
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_content = f.read()
            except OSError as e:
                logger.warning(f"Could not read {file_path}: {e}")
                continue

            if len(raw_content) < 500:
                logger.debug(f"  SKIP (too small): {file_path}")
                continue

            mda_text = extract_mda_section(raw_content, str(file_path))
            if not mda_text:
                logger.info(f"  SKIP (no MD&A match): {file_path}")
                continue

            word_count = len(mda_text.split())
            if word_count < min_words:
                logger.info(f"  SKIP (only {word_count} words < {min_words}): {file_path}")
                continue

            tokens       = encoder.encode(mda_text)
            total_tokens = len(tokens)
            phi_score, num_numeric = calculate_complexity_phi(mda_text, total_tokens)
            seed_prompt  = encoder.decode(tokens[:seed_tokens])

            valid_docs.append({
                "document_id":          str(file_path),
                "ticker":               ticker,
                "total_words":          word_count,
                "total_tokens":         total_tokens,
                "num_numeric_facts":    num_numeric,
                "phi_complexity_score": phi_score,
                "seed_prompt":          seed_prompt,
                "full_mda_text":        mda_text,
            })
            logger.info(f"  OK  {ticker}  words={word_count}  Φ={phi_score:.4f}  ({file_path.name})")

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in valid_docs:
            f.write(json.dumps(doc) + "\n")

    logger.info(
        f"Done. Scanned {total_seen} files → {len(valid_docs)} valid documents "
        f"saved to {output_file}."
    )


def main():
    parser = argparse.ArgumentParser(description="Download and parse SEC EDGAR 10-Q filings")
    parser.add_argument("--tickers", nargs="+",
                        default=["AAPL"], help="Tickers to process (e.g. AAPL GOOG)")
    parser.add_argument("--doc-types", nargs="+",
                        default=["10-Q"], help="Document types to process (e.g. 10-Q 10-K)")
    parser.add_argument("--download-dir",  default="./data/sec_filings")
    parser.add_argument("--output-file",   default="./data/processed_dataset.jsonl")
    parser.add_argument("--email",         required=True,
                        help="Email for SEC EDGAR rate-limiting header")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading; reprocess already-downloaded filings")
    parser.add_argument("--year", type=int, default=2023,
                        help="Only download and process documents from this year (e.g., 2025)")
    parser.add_argument("--seed-tokens", type=int, default=200,
                        help="Number of tokens to use for seed prompt")
    args = parser.parse_args()

    if not args.skip_download:
        logger.info(f"Downloading {args.doc_types} filings for {args.tickers}…")
        dl = Downloader("Research", args.email, args.download_dir)
        for ticker in args.tickers:
            for doc_type in args.doc_types:
                logger.info(f"  Fetching {ticker} {doc_type}…")
                try:
                    if args.year:
                        dl.get(doc_type, ticker, after=f"{args.year}-01-01", before=f"{args.year}-12-31")
                    else:
                        dl.get(doc_type, ticker, limit=4)
                except Exception as e:
                    logger.error(f"  Failed {ticker} {doc_type}: {e}")

    process_filings(args.download_dir, args.output_file, 
                    tickers=args.tickers, doc_types=args.doc_types,
                    year=args.year, seed_tokens=args.seed_tokens)


if __name__ == "__main__":
    main()