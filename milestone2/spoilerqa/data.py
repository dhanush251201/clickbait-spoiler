"""Loading the Webis Clickbait Spoiling splits and turning spoiler positions into context offsets.

The context every model reads is the article title followed by its paragraphs, one per line:

    title \n paragraph 0 \n paragraph 1 \n ...

Putting the title first keeps the ~2% of spoilers that sit in the title (paragraph index -1) extractable.
"""
import json
import re
from pathlib import Path

TYPES = ["phrase", "passage", "multi"]
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "splits"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_split(data_dir, split):
    return load_jsonl(Path(data_dir) / f"{split}.jsonl")


def post_text(record):
    return record["postText"][0]


def spoiler_type(record):
    return record["tags"][0]


def build_context(record):
    return "\n".join([record["targetTitle"], *record["targetParagraphs"]])


def paragraph_start(record, para):
    """Character offset where paragraph `para` starts inside build_context(record); -1 is the title."""
    if para == -1:
        return 0
    paragraphs = record["targetParagraphs"]
    return len(record["targetTitle"]) + 1 + sum(len(p) + 1 for p in paragraphs[:para])


def _position_offset(record, span, position):
    # The annotated position, if it is well-formed, inside one paragraph and reproduces the gold text.
    if len(position) != 2:
        return None
    (p1, c1), (p2, c2) = position
    if p1 != p2 or not -1 <= p1 < len(record["targetParagraphs"]):
        return None
    source = record["targetTitle"] if p1 == -1 else record["targetParagraphs"][p1]
    if source[c1:c2] != span:
        return None
    start = paragraph_start(record, p1) + c1
    return start, start + len(span)


def _search_offset(context, span):
    # First exact occurrence, then first occurrence allowing any whitespace between the span's words.
    start = context.find(span)
    if start != -1:
        return start, start + len(span)
    words = span.split()
    if not words:
        return None
    match = re.search(r"\s+".join(re.escape(w) for w in words), context)
    return (match.start(), match.end()) if match else None


def spoiler_offsets(record, context=None):
    """One (start, end) character offset in the context per gold spoiler span, or None if it can't be located.

    About 98.4% of spans have a correct annotated position. For the rest (positions that cross
    paragraphs, point at other text, or are malformed) we fall back to searching for the gold text.
    Also returns how each span was located: "position", "search" or "missing".
    """
    context = build_context(record) if context is None else context
    offsets, how = [], []
    for span, position in zip(record["spoiler"], record["spoilerPositions"]):
        found = _position_offset(record, span, position)
        if found:
            offsets.append(found)
            how.append("position")
            continue
        found = _search_offset(context, span)
        offsets.append(found)
        how.append("search" if found else "missing")
    return offsets, how
