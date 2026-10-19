"""Freeze a deterministic article-stratified subset from a verified test file.

Input transfer/download is separate; run this builder through remote_run.sh.
Selection never observes model scores. The resulting data remains CC BY-SA.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

SEED = "hqsb-wikitext2-mini-v1"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def build(rows, source):
    articles = []
    for index, line in enumerate(rows):
        title = re.fullmatch(r"\s*=\s+([^=]+?)\s+=\s*", line)
        if (
            title
            and (index == 0 or not rows[index - 1].strip())
            and (index + 1 == len(rows) or not rows[index + 1].strip())
        ):
            articles.append(
                {
                    "title": title.group(1).strip(),
                    "index": len(articles),
                    "paragraphs": [],
                }
            )
        elif (
            articles
            and 400 <= len(line.strip()) <= 4096
            and not line.strip().startswith("=")
        ):
            articles[-1]["paragraphs"].append((index, line.strip()))
    eligible = [article for article in articles if article["paragraphs"]]
    if len(eligible) < 8:
        raise ValueError("Not enough eligible articles")
    samples = []
    for stratum in range(8):
        bucket = eligible[
            stratum * len(eligible) // 8 : (stratum + 1) * len(eligible) // 8
        ]
        article = min(
            bucket, key=lambda a: digest(f"{SEED}|article|{a['index']}|{a['title']}")
        )
        line, paragraph = min(
            article["paragraphs"], key=lambda p: digest(f"{SEED}|paragraph|{p[0]}")
        )
        samples.append(
            {
                "id": f"wt2-test-line-{line}",
                "article": article["title"],
                "article_index": article["index"],
                "line_index": line,
                "stratum": stratum,
                "text": paragraph,
                "text_sha256": digest(paragraph),
            }
        )
    return {
        "id": "wikitext2-mini-v1",
        "source": source,
        "selection": {
            "seed": SEED,
            "algorithm": "8 equal contiguous eligible-article strata; min SHA256 article and paragraph per stratum",
            "heading_rule": "level-one heading bounded by blank rows; exclude equations embedded in prose",
            "eligible_paragraph_characters": [400, 4096],
            "article_count": len(articles),
            "eligible_article_count": len(eligible),
            "text_changes": "strip surrounding whitespace only",
            "model_scores_used": False,
        },
        "license": "CC-BY-SA; see ATTRIBUTION.md for upstream license declarations",
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--output", default="hqsb/core/data/wikitext2-mini-v1.json", type=Path
    )
    args = parser.parse_args()
    raw, source = args.input.read_bytes(), json.loads(args.source.read_text())
    if source["sha256"] != hashlib.sha256(raw).hexdigest() or source["split"] != "test":
        raise ValueError("Upstream content identity/split mismatch")
    rows = json.loads(raw)
    if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
        raise ValueError("Expected original ordered dataset text rows")
    document = build(rows, source)
    result = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result)
    print(
        json.dumps(
            {
                "sha256": hashlib.sha256(result).hexdigest(),
                "bytes": len(result),
                "articles": [s["article"] for s in document["samples"]],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
