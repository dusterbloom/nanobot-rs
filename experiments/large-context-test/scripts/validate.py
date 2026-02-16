#!/usr/bin/env python3
"""
Validate nanobot output against ground truth.
"""

import json
import re
import sys
from pathlib import Path

GROUND_TRUTH = Path(__file__).parent.parent / "results" / "ground_truth.json"
NANOBOT_OUTPUT = Path(__file__).parent.parent / "results" / "nanobot_output.txt"


def load_ground_truth() -> dict:
    with open(GROUND_TRUTH, "r") as f:
        return json.load(f)


def extract_author_mentions(text: str, authors: list[str]) -> dict[str, bool]:
    """Check which ground-truth authors are mentioned in the output."""
    mentions = {}
    text_lower = text.lower()
    
    for author in authors:
        # Check for last name or full name
        last_name = author.split()[-1].lower()
        full_name = author.lower()
        
        mentions[author] = (
            last_name in text_lower or 
            full_name in text_lower or
            author in text
        )
    
    return mentions


def extract_keyword_coverage(text: str, gt: dict) -> float:
    """Check how many ground-truth findings are reflected in output."""
    total_findings = 0
    covered_findings = 0
    
    for author, data in gt["authors"].items():
        for finding in data.get("findings", []):
            total_findings += 1
            title_words = set(re.findall(r"\w+", finding["title"].lower()))
            output_words = set(re.findall(r"\w+", text.lower()))
            
            # Check if key words from title appear in output
            overlap = len(title_words & output_words) / max(len(title_words), 1)
            if overlap > 0.3:  # 30% word overlap threshold
                covered_findings += 1
    
    return covered_findings / max(total_findings, 1)


def validate(output_text: str) -> dict:
    """Validate nanobot output against ground truth."""
    gt = load_ground_truth()
    
    # 1. Author coverage
    gt_authors = list(gt["authors"].keys())
    author_mentions = extract_author_mentions(output_text, gt_authors)
    author_coverage = sum(author_mentions.values()) / len(gt_authors)
    
    # 2. Author accuracy (are we finding the right top-10?)
    # The output should identify the same top authors
    # This is fuzzy - we check if most are mentioned
    
    # 3. Keyword coverage
    keyword_coverage = extract_keyword_coverage(output_text, gt)
    
    # 4. Emergence discussion detection
    emergence_keywords = ["emergent", "emergence", "scaling law", "phase transition"]
    emergence_discussed = any(kw in output_text.lower() for kw in emergence_keywords)
    
    return {
        "author_coverage": {
            "ratio": author_coverage,
            "mentioned": [a for a, m in author_mentions.items() if m],
            "missed": [a for a, m in author_mentions.items() if not m],
        },
        "keyword_coverage": keyword_coverage,
        "emergence_discussed": emergence_discussed,
        "overall_score": (author_coverage + keyword_coverage + (1 if emergence_discussed else 0)) / 3,
        "pass": author_coverage > 0.7 and emergence_discussed,
    }


def main():
    if len(sys.argv) > 1:
        output_file = Path(sys.argv[1])
    else:
        output_file = NANOBOT_OUTPUT
    
    if not output_file.exists():
        print(f"ERROR: Output file not found: {output_file}")
        sys.exit(1)
    
    if not GROUND_TRUTH.exists():
        print(f"ERROR: Ground truth not found. Run ground_truth.py first.")
        sys.exit(1)
    
    output_text = output_file.read_text(encoding="utf-8")
    results = validate(output_text)
    
    print("=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    print(f"\nAuthor Coverage: {results['author_coverage']['ratio']:.1%}")
    print(f"  Mentioned: {len(results['author_coverage']['mentioned'])}/10")
    print(f"  Missed: {results['author_coverage']['missed']}")
    print(f"\nKeyword Coverage: {results['keyword_coverage']:.1%}")
    print(f"Emergence Discussed: {'Yes' if results['emergence_discussed'] else 'No'}")
    print(f"\nOverall Score: {results['overall_score']:.1%}")
    print(f"\nPASS: {'✓' if results['pass'] else '✗'}")
    
    # Write results
    results_file = output_file.parent / "validation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {results_file}")
    
    sys.exit(0 if results["pass"] else 1)


if __name__ == "__main__":
    main()
