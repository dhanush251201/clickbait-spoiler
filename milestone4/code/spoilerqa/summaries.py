"""LLM summaries of clickbait articles, and whether they give the spoiler away (4-summaries notebook).

Two kinds of summary, from the same models as the spoiler extraction:
- "generic": a two-sentence summary of the article alone (the model does not see the post). Does an ordinary
  summary already spoil the clickbait?
- "spoiler": a two-sentence summary written for the post, asked to tell the reader what the post teases.

Each summary is checked three ways: does it contain the gold spoiler as text (exact, or fuzzy), does an LLM judge
think it reveals the spoiler (yes / partly / no), and, for a small sample, what human raters think.
"""
import json
import re

from rapidfuzz import fuzz

from .compare import squash
from .data import build_context, post_text
from .llm import _first_json_object, cut_article

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}

JUDGE_LABELS = ["yes", "partly", "no"]
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"reveals": {"type": "string", "enum": JUDGE_LABELS}},
    "required": ["reveals"],
    "additionalProperties": False,
}

SYSTEM_GENERIC = """You summarize news articles. Write a summary of the article in at most two sentences (about 50 words) for a reader who has not read it.

Reply with JSON only: {"summary": "..."}"""

SYSTEM_SPOILER = """You get a clickbait post and the article it links to. The post teases something without saying it. Write a summary of the article in at most two sentences (about 50 words) that also tells the reader exactly what the post is teasing, so they don't need to click.

Reply with JSON only: {"summary": "..."}"""

SYSTEM_JUDGE = """You check whether a summary gives away the spoiler of a clickbait post. You get the post, its spoiler (the information the post hides, taken from the article; a post that promises several things has a spoiler with several parts) and a summary of the article.

Answer "yes" if a reader of the summary learns the spoiler, in any wording. Answer "partly" if they learn only part of it (for example some parts of a several-part spoiler, or a vaguer version). Otherwise answer "no".

Reply with JSON only: {"reveals": "yes" | "partly" | "no"}"""

VARIANTS = {"generic": SYSTEM_GENERIC, "spoiler": SYSTEM_SPOILER}


def summary_prompt(record, variant):
    """(system prompt, messages) for one summary. The generic variant never sees the post."""
    article = cut_article(build_context(record))
    user = f"Article:\n{article}" if variant == "generic" else f"Post: {post_text(record)}\n\nArticle:\n{article}"
    return VARIANTS[variant], [{"role": "user", "content": user}]


def judge_prompt(record, summary):
    """(system prompt, messages) for the judge: the post, the gold spoiler and the summary (not the article)."""
    spoiler = "\n".join(f"- {s}" for s in record["spoiler"])
    user = f"Post: {post_text(record)}\n\nSpoiler:\n{spoiler}\n\nSummary: {summary}"
    return SYSTEM_JUDGE, [{"role": "user", "content": user}]


def parse_field(text, field):
    """One field of a JSON reply, or None if the reply can't be read."""
    for candidate in (text or "", _first_json_object(text or "")):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and isinstance(data.get(field), str):
            return data[field].strip()
    return None


FUZZY_MIN_SCORE = 85


def spoiler_in_summary(summary, gold_items, min_score=FUZZY_MIN_SCORE):
    """(exact, fuzzy): the share of gold spoiler items found in the summary.

    - exact: the item's words appear in the summary in the same order (ignoring case, punctuation and spacing);
    - fuzzy: exact, or an alignment with rapidfuzz partial_ratio >= min_score (small wording changes).
    """
    text = f" {squash(summary)} "
    exact = fuzzy = 0
    for item in gold_items:
        g = squash(item)
        if not g:
            continue
        hit = f" {g} " in text
        exact += hit
        # fuzzy only when the item fits in the summary; otherwise partial_ratio would align the summary inside it
        fuzzy += hit or (len(g) <= len(text) and fuzz.partial_ratio(g, text.strip()) >= min_score)
    n = max(sum(bool(squash(i)) for i in gold_items), 1)
    return exact / n, fuzzy / n


def words(text):
    return len(re.findall(r"\S+", text or ""))
