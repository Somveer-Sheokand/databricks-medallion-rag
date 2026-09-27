"""Downloads a small sample of arXiv PDFs into ./local_docs/<dataset>/, ready
to be uploaded to the Databricks Unity Catalog volume as Bronze input (see
databricks/README.md). Runs on your own machine, not inside a notebook, so
Free Edition's restricted outbound internet from notebooks is a non-issue.

Usage:
    python scripts/fetch_arxiv_sample.py --category cs.CL --max-results 25
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


def fetch_metadata(category: str, max_results: int) -> List[Paper]:
    query = (
        f"?search_query=cat:{category}&start=0&max_results={max_results}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    with _urlopen(ARXIV_API + query) as resp:
        root = ET.fromstring(resp.read())

    papers: List[Paper] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        arxiv_id = entry.find("atom:id", ATOM_NS).text.rsplit("/", 1)[-1]
        papers.append({"id": arxiv_id, "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf"})
    return papers


def download(papers: List[Paper], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for paper in papers:
        dest = out_dir / f"{paper['id']}.pdf"
        if dest.exists():
            continue
        print(f"downloading {paper['id']}")
        for attempt in range(3):
            try:
                with _urlopen(paper["pdf_url"]) as resp, open(dest, "wb") as f:
                    f.write(resp.read())
                break
            except urllib.error.HTTPError as e:
                if attempt == 2:
                    print(f"  skipping {paper['id']}: {e}")
                    break
                time.sleep(2 * (attempt + 1))
        time.sleep(1)  # be polite to arXiv's servers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="cs.CL", help="arXiv category, e.g. cs.CL, cs.LG")
    parser.add_argument("--max-results", type=int, default=25)
    parser.add_argument("--out-dir", default="local_docs/arxiv_sample")
    args = parser.parse_args()

    papers = fetch_metadata(args.category, args.max_results)
    download(papers, Path(args.out_dir))
    print(f"Downloaded {len(papers)} PDFs to {args.out_dir}")


if __name__ == "__main__":
    main()
