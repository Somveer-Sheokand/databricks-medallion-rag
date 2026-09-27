"""Downloads a small sample of arXiv PDFs into ./local_docs/<dataset>/, ready
to be uploaded to the Databricks Unity Catalog volume as Bronze input (see
databricks/README.md). Runs on your own machine, not inside a notebook, so
Free Edition's restricted outbound internet from notebooks is a non-issue.

arXiv's edge aggressively rate-limits/bot-mitigates PDF downloads (expect
maybe 20-40% of requests to succeed even with a proper User-Agent), so this
paginates through metadata and keeps trying additional candidates until
`--target-count` files have actually downloaded, rather than requesting a
fixed batch and accepting whatever fraction gets through.

Usage:
    python scripts/fetch_arxiv_sample.py --category cs.CL --target-count 20
"""
from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, TypedDict

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
# arXiv's CDN returns 406 Not Acceptable for urllib's default User-Agent.
USER_AGENT = "varnam-agent/0.1 (https://github.com/Somveer-Sheokand/databricks-medallion-rag)"


class Paper(TypedDict):
    id: str
    pdf_url: str


def _urlopen(url: str):
    # arXiv's edge (Fastly/WAF) 406s requests missing either header, not just User-Agent.
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers))


def _urlopen_with_retry(url: str, attempts: int = 5):
    """Every request to arXiv's edge is subject to the same bot-mitigation
    flakiness -- the metadata/ATOM endpoint 406s intermittently too, not just
    PDF downloads. Retried here so a rate-limited pagination request doesn't
    crash a run that already has successful downloads banked."""
    for attempt in range(attempts):
        try:
            return _urlopen(url)
        except urllib.error.HTTPError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(3 * (2**attempt), 30))


def fetch_metadata_page(category: str, start: int, batch_size: int) -> List[Paper]:
    query = (
        f"?search_query=cat:{category}&start={start}&max_results={batch_size}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    with _urlopen_with_retry(ARXIV_API + query) as resp:
        root = ET.fromstring(resp.read())

    papers: List[Paper] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        arxiv_id = entry.find("atom:id", ATOM_NS).text.rsplit("/", 1)[-1]
        papers.append({"id": arxiv_id, "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf"})
    return papers


def download_one(paper: Paper, out_dir: Path, attempts: int = 5) -> bool:
    dest = out_dir / f"{paper['id']}.pdf"
    if dest.exists():
        return True
    try:
        with _urlopen_with_retry(paper["pdf_url"], attempts=attempts) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return True
    except urllib.error.HTTPError as e:
        print(f"  skipping {paper['id']}: {e}")
        return False


def download_until(category: str, out_dir: Path, target_count: int, max_candidates: int,
                    batch_size: int = 25) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    successes = 0
    start = 0
    tried = 0
    while successes < target_count and tried < max_candidates:
        try:
            batch = fetch_metadata_page(category, start, batch_size)
        except urllib.error.HTTPError as e:
            print(f"metadata request failed after retries, stopping with what we have: {e}")
            break
        if not batch:
            break
        start += len(batch)
        for paper in batch:
            if successes >= target_count or tried >= max_candidates:
                break
            tried += 1
            print(f"[{successes}/{target_count}] downloading {paper['id']}")
            if download_one(paper, out_dir):
                successes += 1
            time.sleep(1)  # be polite to arXiv's servers
    return successes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="cs.CL", help="arXiv category, e.g. cs.CL, cs.LG")
    parser.add_argument("--target-count", type=int, default=20,
                         help="Keep trying additional papers until this many actually download")
    parser.add_argument("--max-candidates", type=int, default=120,
                         help="Safety cap on total download attempts, in case of heavy rate-limiting")
    parser.add_argument("--out-dir", default="local_docs/arxiv_sample")
    args = parser.parse_args()

    successes = download_until(
        args.category, Path(args.out_dir), args.target_count, args.max_candidates,
    )
    print(f"Downloaded {successes}/{args.target_count} target PDFs to {args.out_dir}")


if __name__ == "__main__":
    main()
