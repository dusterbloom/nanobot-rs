#!/usr/bin/env python3
"""
Generate ground truth answers for the large-context test.
This provides the expected results to validate nanobot's output.
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "arxiv_papers.csv"
OUTPUT_FILE = Path(__file__).parent.parent / "results" / "ground_truth.json"

KEYWORDS = [
    "emergent capabilities",
    "emergence",
    "scaling laws",
    "scaling law",
    "phase transition",
    "emergent ability",
    "emergent behavior",
    "few-shot",
    "zero-shot",
    "chain-of-thought",
]


def load_papers() -> list[dict]:
    papers = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            papers.append(row)
    return papers


def get_top_authors(papers: list[dict], n: int = 5) -> list[tuple[str, int]]:
    """Get the N most-published authors."""
    author_counts = Counter()
    
    for paper in papers:
        authors = [a.strip() for a in paper["authors"].split(";")]
        for author in authors:
            author_counts[author] += 1
    
    return author_counts.most_common(n)


def find_keyword_matches(text: str) -> list[dict]:
    """Find all keyword matches in text with context."""
    matches = []
    text_lower = text.lower()
    
    for keyword in KEYWORDS:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for match in pattern.finditer(text):
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].strip()
            matches.append({
                "keyword": keyword,
                "context": context,
                "position": match.start(),
            })
    
    return matches


def analyze_author(author: str, papers: list[dict]) -> dict:
    """Analyze an author's papers for emergence/scaling content."""
    author_papers = [
        p for p in papers 
        if author in [a.strip() for a in p["authors"].split(";")]
    ]
    
    findings = []
    total_matches = 0
    
    for paper in author_papers:
        matches = find_keyword_matches(paper["abstract"])
        if matches:
            findings.append({
                "title": paper["title"],
                "arxiv_id": paper["id"],
                "matches": matches,
            })
            total_matches += len(matches)
    
    return {
        "author": author,
        "paper_count": len(author_papers),
        "papers_with_keywords": len(findings),
        "total_keyword_matches": total_matches,
        "findings": findings,
    }


def main():
    print("Loading papers...")
    papers = load_papers()
    print(f"Loaded {len(papers)} papers")
    
    print("\nFinding top 5 authors...")
    top_authors = get_top_authors(papers, 5)
    print("Top authors:")
    for author, count in top_authors:
        print(f"  {author}: {count} papers")
    
    print("\nAnalyzing each author for emergence/scaling keywords...")
    ground_truth = {
        "metadata": {
            "total_papers": len(papers),
            "keywords_searched": KEYWORDS,
            "top_authors": [{"name": a, "paper_count": c} for a, c in top_authors],
        },
        "authors": {},
    }
    
    for author, count in top_authors:
        print(f"  Analyzing {author}...")
        result = analyze_author(author, papers)
        ground_truth["authors"][author] = result
        
        if result["papers_with_keywords"] > 0:
            print(f"    Found {result['total_keyword_matches']} matches in {result['papers_with_keywords']} papers")
    
    # Summary stats
    authors_with_findings = sum(
        1 for a in ground_truth["authors"].values() 
        if a["papers_with_keywords"] > 0
    )
    total_matches = sum(
        a["total_keyword_matches"] for a in ground_truth["authors"].values()
    )
    
    ground_truth["summary"] = {
        "authors_discussing_emergence": authors_with_findings,
        "total_keyword_matches": total_matches,
        "most_discussed_author": max(
            ground_truth["authors"].items(),
            key=lambda x: x[1]["total_keyword_matches"]
        )[0] if total_matches > 0 else None,
    }
    
    print(f"\n{'='*50}")
    print("Ground Truth Summary:")
    print(f"  Authors discussing emergence: {authors_with_findings}/10")
    print(f"  Total keyword matches: {total_matches}")
    if ground_truth["summary"]["most_discussed_author"]:
        print(f"  Most discussed: {ground_truth['summary']['most_discussed_author']}")
    
    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    
    print(f"\nGround truth written to: {OUTPUT_FILE}")
    
    return ground_truth


if __name__ == "__main__":
    main()
