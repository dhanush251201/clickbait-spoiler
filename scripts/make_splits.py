"""Build the train / dev / test splits for Milestone 1 from the raw Zenodo download.

- official train -> train, official validation -> dev, official test -> test
- test is kept exactly as released (it is the SemEval-2023 Task 5 test set)
- an item that also appears in a later split is dropped from the earlier one:
  dev loses items that are in test, train loses items that are in dev or test
- kept records are copied line for line, byte for byte

Also writes data/splits/removed_duplicates.tsv and packs data/spoilerqa_data.tar.gz
(the three splits, the removed-items list and data.md if it exists).

Usage: python scripts/make_splits.py
"""
import csv
import itertools
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "splits"
ARCHIVE = ROOT / "data" / "spoilerqa_data.tar.gz"
DATA_MD = ROOT / "data.md"

SOURCE_FILES = {"train": "train.jsonl", "dev": "validation.jsonl", "test": "test.jsonl"}
PRIORITY = {"train": 0, "dev": 1, "test": 2}  # the lower split gives up a shared item
MIN_ARTICLE_OVERLAP = 0.5


def normalize(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def load_items():
    items = []
    for split, name in SOURCE_FILES.items():
        with open(RAW_DIR / name, encoding="utf-8") as f:
            for line in f:
                items.append({"split": split, "line": line, "record": json.loads(line)})
    return items


def cross_split_duplicates(items):
    """Pairs of items in different splits that are the same story.

    Candidates share a normalized post text, title or article text. A candidate pair counts as a
    duplicate when the two articles' vocabularies overlap (Jaccard) by at least MIN_ARTICLE_OVERLAP.
    That catches re-crawls of the same article with a reworded post, and keeps apart recurring
    headlines such as "10 things you need to know before the opening bell", which point at a
    different article every day.
    """
    by_key = defaultdict(list)
    for i, item in enumerate(items):
        r = item["record"]
        for field, text in [("post", r["postText"][0]), ("title", r["targetTitle"]),
                            ("article", " ".join(r["targetParagraphs"]))]:
            key = normalize(text)
            if key:
                by_key[(field, key)].append(i)

    vocab = [set(normalize(" ".join(item["record"]["targetParagraphs"])).split()) for item in items]
    pairs = {}
    for ids in by_key.values():
        for i, j in itertools.combinations(ids, 2):
            if items[i]["split"] == items[j]["split"]:
                continue
            overlap = len(vocab[i] & vocab[j]) / max(1, len(vocab[i] | vocab[j]))
            if overlap >= MIN_ARTICLE_OVERLAP:
                pairs[(min(i, j), max(i, j))] = overlap
    return pairs


def main():
    items = load_items()
    pairs = cross_split_duplicates(items)

    # For every pair, the item in the lower-priority split is removed. Keep the strongest match as the reason.
    removed = {}
    for (i, j), overlap in pairs.items():
        loser, winner = (i, j) if PRIORITY[items[i]["split"]] < PRIORITY[items[j]["split"]] else (j, i)
        if loser not in removed or overlap > removed[loser][1]:
            removed[loser] = (winner, overlap)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SOURCE_FILES:
        kept = [item["line"] for k, item in enumerate(items) if item["split"] == split and k not in removed]
        with open(OUT_DIR / f"{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(kept)
        total = sum(item["split"] == split for item in items)
        print(f"{split:5s}: {total} -> {len(kept)} ({total - len(kept)} removed)")

    with open(OUT_DIR / "removed_duplicates.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["removed_uuid", "removed_from", "duplicate_of_uuid", "duplicate_in", "article_overlap", "post"])
        for loser, (winner, overlap) in sorted(removed.items()):
            writer.writerow([items[loser]["record"]["uuid"], items[loser]["split"],
                             items[winner]["record"]["uuid"], items[winner]["split"],
                             f"{overlap:.2f}", " ".join(items[loser]["record"]["postText"][0].split())])

    with tarfile.open(ARCHIVE, "w:gz") as tar:
        for name in ["train.jsonl", "dev.jsonl", "test.jsonl", "removed_duplicates.tsv"]:
            tar.add(OUT_DIR / name, arcname=f"spoilerqa_data/{name}")
        if DATA_MD.exists():
            tar.add(DATA_MD, arcname="spoilerqa_data/data.md")
    print(f"wrote {ARCHIVE.relative_to(ROOT)} ({ARCHIVE.stat().st_size / 1e6:.1f} MB)"
          + ("" if DATA_MD.exists() else " -- data.md not found, archive has no data.md yet"))


if __name__ == "__main__":
    main()
