"""
Downloads SEC filings (10-K / 10-Q) for a given ticker from SEC EDGAR and
saves the primary document HTML to data/filings/.

SEC EDGAR requires a descriptive User-Agent header with contact info on every
request (set SEC_EDGAR_USER_AGENT in .env) — requests without one get blocked.

Usage:
    python scripts/download_filings.py --ticker C --form-type 10-K --count 1
"""
import argparse
import json
import time
from pathlib import Path

import requests

from app.config import settings

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

OUTPUT_DIR = Path("data/filings")


def _headers() -> dict:
    return {"User-Agent": settings.sec_edgar_user_agent}


def ticker_to_cik(ticker: str) -> str:
    """Look up a company's 10-digit zero-padded CIK from its ticker."""
    resp = requests.get(EDGAR_TICKER_MAP_URL, headers=_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker '{ticker}' not found in SEC ticker map.")


def get_filing_metadata(cik: str, form_type: str, count: int) -> list[dict]:
    """Return metadata for the most recent `count` filings of `form_type`."""
    resp = requests.get(
        EDGAR_SUBMISSIONS_URL.format(cik=cik), headers=_headers(), timeout=30
    )
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]

    results = []
    for i, form in enumerate(recent["form"]):
        if form == form_type:
            results.append(
                {
                    "accession_number": recent["accessionNumber"][i],
                    "filing_date": recent["filingDate"][i],
                    "primary_document": recent["primaryDocument"][i],
                }
            )
        if len(results) >= count:
            break
    return results


def download_document(cik: str, accession_number: str, primary_document: str) -> Path:
    """Download the primary document of a filing and save it locally."""
    accession_no_dashes = accession_number.replace("-", "")
    url = (
        f"{EDGAR_ARCHIVES_BASE}/{int(cik)}/{accession_no_dashes}/{primary_document}"
    )
    resp = requests.get(url, headers=_headers(), timeout=60)
    resp.raise_for_status()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{cik}_{accession_number}_{primary_document}"
    out_path.write_bytes(resp.content)
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download SEC filings from EDGAR.")
    parser.add_argument("--ticker", required=True, help="e.g. C for Citigroup")
    parser.add_argument("--form-type", default="10-K", help="10-K or 10-Q")
    parser.add_argument("--count", type=int, default=1, help="How many filings")
    args = parser.parse_args()

    print(f"Resolving CIK for ticker '{args.ticker}'...")
    cik = ticker_to_cik(args.ticker)
    print(f"CIK: {cik}")

    print(f"Fetching {args.form_type} filing metadata...")
    filings = get_filing_metadata(cik, args.form_type, args.count)
    if not filings:
        print("No matching filings found.")
        return

    for f in filings:
        print(f"Downloading {f['form_type'] if 'form_type' in f else args.form_type} "
              f"filed {f['filing_date']} ({f['accession_number']})...")
        download_document(cik, f["accession_number"], f["primary_document"])
        time.sleep(0.5)  # be polite to SEC's rate limits

    # Save metadata alongside the docs for the ingestion step to use
    meta_path = OUTPUT_DIR / f"{cik}_{args.form_type}_metadata.json"
    meta_path.write_text(json.dumps(filings, indent=2))
    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
