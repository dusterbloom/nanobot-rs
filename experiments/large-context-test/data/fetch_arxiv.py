#!/usr/bin/env python3
"""
Fetch ArXiv papers for large-context multi-agent test.
Downloads papers from cs.CL, cs.AI, cs.LG categories.
"""

import csv
import json
import urllib.request
import urllib.parse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_API = "http://export.arxiv.org/api/query"
OUTPUT_FILE = Path(__file__).parent / "arxiv_papers.csv"

# Target: ~100 papers for quick test
CATEGORIES = ["cs.CL", "cs.AI"]
PAPERS_PER_CATEGORY = 50  # ~100 total


def fetch_papers(category: str, start: int = 0, max_results: int = 100) -> list[dict]:
    """Fetch papers from ArXiv API."""
    query = f"cat:{category}"
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    print(f"Fetching: {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "nanobot-test/1.0"})
    response = urllib.request.urlopen(req, timeout=60)
    xml_data = response.read().decode("utf-8")

    root = ET.fromstring(xml_data)
    papers = []

    ns = {"atom": "http://www.w3.org/2005/Atom"}

    for entry in root.findall("atom:entry", ns):
        id_elem = entry.find("atom:id", ns)
        title_elem = entry.find("atom:title", ns)
        summary_elem = entry.find("atom:summary", ns)
        published_elem = entry.find("atom:published", ns)
        
        if None in (id_elem, title_elem, summary_elem, published_elem):
            continue
        
        # Type narrowing - all elements confirmed non-None above
        assert id_elem is not None and title_elem is not None
        assert summary_elem is not None and published_elem is not None
            
        author_names = []
        for a in entry.findall("atom:author", ns):
            name_elem = a.find("atom:name", ns)
            if name_elem is not None and name_elem.text:
                author_names.append(name_elem.text)
        
        category_terms = []
        for c in entry.findall("atom:category", ns):
            term = c.get("term")
            if term:
                category_terms.append(term)
        
        paper_id = (id_elem.text or "").split("/")[-1]
        title = (title_elem.text or "").strip().replace("\n", " ")
        abstract = (summary_elem.text or "").strip().replace("\n", " ")
        published = (published_elem.text or "")[:10]
        
        paper = {
            "id": paper_id,
            "title": title,
            "authors": "; ".join(author_names),
            "abstract": abstract,
            "published": published,
            "categories": "; ".join(category_terms),
        }
        papers.append(paper)

    return papers


def main():
    all_papers = []
    seen_ids = set()

    for category in CATEGORIES:
        print(f"\nFetching {category}...")
        for start in range(0, PAPERS_PER_CATEGORY, 100):
            batch = min(100, PAPERS_PER_CATEGORY - start)
            papers = fetch_papers(category, start, batch)

            for p in papers:
                if p["id"] not in seen_ids:
                    all_papers.append(p)
                    seen_ids.add(p["id"])

            print(f"  Got {len(papers)} papers (total: {len(all_papers)})")
            time.sleep(3)  # Be nice to ArXiv

    # Sort by date
    all_papers.sort(key=lambda x: x["published"], reverse=True)

    # Write CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "title", "authors", "abstract", "published", "categories"]
        )
        writer.writeheader()
        writer.writerows(all_papers[:500])

    # Stats
    total_chars = sum(
        len(p["title"]) + len(p["abstract"]) + len(p["authors"])
        for p in all_papers[:500]
    )

    print(f"\n{'='*50}")
    print(f"Wrote {min(500, len(all_papers))} papers to {OUTPUT_FILE}")
    print(f"Total characters: {total_chars:,}")
    print(f"Estimated tokens: {total_chars // 4:,}")
    print(f"Unique authors: {len(set(a for p in all_papers[:500] for a in p['authors'].split('; ')))}")


if __name__ == "__main__":
    main()
