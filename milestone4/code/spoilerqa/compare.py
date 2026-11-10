"""Shared pieces of the comparison and error-analysis notebooks (2-compare, 3-error-analysis).

- The list of systems: the baselines and fine-tuned models (Milestones 2 and 3) and the prompted LLMs (this
  milestone), with where their test predictions and score files are.
- Per-post BLEU (the official definition) and a paired bootstrap test.
- Small helpers for the error analysis: spoiler items, text normalization, span-level error categories.
"""
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import llm
from .data import TYPES, spoiler_type
from .metrics import as_text, bleu_scores, token_f1

# ----------------------------------------------------------------------------------------------------------------
# Systems
# ----------------------------------------------------------------------------------------------------------------

# The fine-tuned systems' outputs are copied into output/finetuned/ (from the milestone folders, relative to the
# project root), so the final package holds every system's test predictions next to the prompted ones.
FINETUNED_SOURCES = {
    "simple-baseline": "milestone2/output",
    "strong-baseline": "milestone2/output",
    "deberta-qa": "milestone3/output",
    "flan-t5": "milestone3/output",
    "llama-qlora": "milestone3/output",
}

FINETUNED = [
    {"key": "simple-baseline", "stem": "simple-baseline", "family": "baseline", "size": "-",
     "label": "Simple baseline (first sentence)", "short": "Simple baseline"},
    {"key": "roberta-base-qa", "stem": "strong-baseline", "family": "fine-tuned", "size": "125M",
     "label": "RoBERTa-base QA (M2 strong baseline)", "short": "RoBERTa-base QA"},
    {"key": "deberta-qa", "stem": "deberta-qa", "family": "fine-tuned", "size": "435M",
     "label": "DeBERTa-v3-large QA", "short": "DeBERTa-v3-large QA"},
    {"key": "flan-t5", "stem": "flan-t5", "family": "fine-tuned", "size": "783M",
     "label": "Flan-T5-large", "short": "Flan-T5-large"},
    {"key": "llama-qlora", "stem": "llama-qlora", "family": "fine-tuned", "size": "8B (42M trained)",
     "label": "Llama 3.1 8B, fine-tuned (QLoRA)", "short": "Llama 8B fine-tuned"},
]

PROMPTED = [
    {"key": "llama3.1-8b", "size": "8B (4-bit)", "label": "Llama 3.1 8B, prompted", "short": "Llama 8B prompted"},
    {"key": "gpt-oss-20b", "size": "21B MoE (4-bit)", "label": "GPT-OSS 20B", "short": "GPT-OSS 20B"},
    {"key": "claude-sonnet-5.5", "size": "not disclosed", "label": "Claude Sonnet 5.5", "short": "Claude Sonnet 5.5"},
    {"key": "claude-haiku-5.5", "size": "not disclosed", "label": "Claude Haiku 5.5", "short": "Claude Haiku 5.5"},
    {"key": "gpt-6.1-sol", "size": "not disclosed", "label": "GPT-6.1 Sol", "short": "GPT-6.1 Sol"},
    {"key": "gpt-6-luna", "size": "not disclosed", "label": "GPT-6 Luna", "short": "GPT-6 Luna"},
]

# SemEval-2023 Task 5 reference numbers on the same test set (overview paper, Fröbe et al. 2023).
SEMEVAL_REFERENCES = [
    {"system": "Reference: SemEval organizer baseline (DeBERTa-large QA)", "BLEU": 0.40, "phrase": 0.65,
     "passage": 0.24, "multi": 0.12},
    {"system": "Reference: best SemEval-2023 system", "BLEU": 0.48, "phrase": 0.66, "passage": 0.32, "multi": 0.44,
     "type bal. acc.": 0.74},
]


def collect_finetuned(project_root, output_dir):
    """Copies the baselines' and fine-tuned models' test predictions and score files into output/finetuned/.

    Files already there and identical are left alone. When the milestone folders are absent (as in the packaged
    tarball), the copies already in output/finetuned/ are used. Returns the number of files copied.
    """
    pairs = []
    for stem, source in FINETUNED_SOURCES.items():
        for name in (f"{stem}-test", f"{stem}-goldtypes-test"):
            pairs.append((Path(source) / f"{name}.jsonl", Path("predictions") / f"{name}.jsonl"))
            pairs.append((Path(source) / "scores" / f"{name}.json", Path("scores") / f"{name}.json"))
        for name in (f"{stem}-sweep.json", f"{stem}-settings.json"):  # learning-rate sweeps and chosen settings
            pairs.append((Path(source) / name, Path("training") / name))
    pairs.append((Path("milestone2/output/qa-sweep.json"), Path("training/strong-baseline-sweep.json")))
    copied = 0
    for src, dst in pairs:
        src, dst = Path(project_root) / src, Path(output_dir) / "finetuned" / dst
        if src.exists() and (not dst.exists() or dst.read_bytes() != src.read_bytes()):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied += 1
    return copied


def systems(output_dir):
    """Every system with test predictions, in table order: baselines and fine-tuned models, then prompted LLMs.

    Each is a dict with key, label, short, family, size, pred (prediction file), scores (score.py JSON) and, for
    the fine-tuned ones, goldtypes_scores; the prompted ones also have k (worked examples) and run (raw-file name).
    """
    output_dir = Path(output_dir)
    found = []
    for s in FINETUNED:
        folder = output_dir / "finetuned"
        pred = folder / "predictions" / f"{s['stem']}-test.jsonl"
        if pred.exists():
            found.append({**s, "pred": pred, "scores": folder / "scores" / f"{s['stem']}-test.json",
                          "goldtypes_scores": folder / "scores" / f"{s['stem']}-goldtypes-test.json"})
    best_k_path = output_dir / "llm-best-k.json"
    best_k = json.loads(best_k_path.read_text()) if best_k_path.exists() else {}
    for s in PROMPTED:
        if s["key"] not in best_k:
            continue
        run = llm.run_name(s["key"], "article", best_k[s["key"]], "test")
        pred = output_dir / "predictions" / f"{run}.jsonl"
        if pred.exists():
            found.append({**s, "family": "prompted", "k": best_k[s["key"]], "run": run, "pred": pred,
                          "scores": output_dir / "scores" / f"{run}.json"})
    return found


def load_preds(path):
    with open(path, encoding="utf-8") as f:
        return {p["uuid"]: p for p in map(json.loads, f) if p}


def load_scores(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


# ----------------------------------------------------------------------------------------------------------------
# Per-post BLEU and the paired bootstrap
# ----------------------------------------------------------------------------------------------------------------

def per_post_bleu(records, preds):
    """Official per-post BLEU as a Series indexed by uuid, in the order of `records` (posts without a prediction
    are left out)."""
    kept = [r for r in records if r["uuid"] in preds]
    scores = bleu_scores([as_text(r["spoiler"]) for r in kept], [as_text(preds[r["uuid"]]["spoiler"]) for r in kept])
    return pd.Series(scores, index=[r["uuid"] for r in kept])


def paired_bootstrap(a, b, n_resamples=10_000, seed=0):
    """Paired bootstrap of the mean difference a - b over the same posts.

    Returns a dict: the observed difference, its 95% percentile interval, p_not_better (the share of resamples in
    which a is not better than b; one-sided) and p_two_sided.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_resamples, len(diff)))].mean(axis=1)
    p_not_better = float((means <= 0).mean())
    return {"difference": float(diff.mean()), "95% low": float(np.percentile(means, 2.5)),
            "95% high": float(np.percentile(means, 97.5)), "p (not better)": p_not_better,
            "p (two-sided)": min(1.0, 2 * min(p_not_better, float((means >= 0).mean())))}


def holm(p_values, alpha=0.05):
    """Holm-Bonferroni: which of several tests stay significant at family-wise level alpha (list of bools)."""
    p = np.asarray(p_values, dtype=float)
    significant = np.zeros(len(p), dtype=bool)
    for rank, i in enumerate(np.argsort(p)):
        if p[i] > alpha / (len(p) - rank):
            break
        significant[i] = True
    return significant.tolist()


def bootstrap_rows(records, bleu_a, bleu_b, label_a, label_b, **kwargs):
    """Paired bootstrap rows for all posts and for each gold type."""
    by_id = {r["uuid"]: r for r in records}
    common = [u for u in bleu_a.index if u in bleu_b.index]
    rows = []
    for subset in ["all"] + TYPES:
        idx = [u for u in common if subset == "all" or spoiler_type(by_id[u]) == subset]
        rows.append({"system": label_a, "compared with": label_b, "posts": subset, "n": len(idx),
                     **paired_bootstrap(bleu_a[idx], bleu_b[idx], **kwargs)})
    return rows


# ----------------------------------------------------------------------------------------------------------------
# Error-analysis helpers
# ----------------------------------------------------------------------------------------------------------------

def items(spoiler):
    """A spoiler (string or list) as a list of non-empty items."""
    spans = spoiler if isinstance(spoiler, list) else [spoiler]
    return [s for s in spans if s and s.strip()]


def squash(text):
    """Lower case, punctuation removed, spacing collapsed: for containment checks."""
    return " ".join(re.sub(r"[^\w\s]", " ", str(text).lower()).split())


def word_count(text):
    return len(str(text).split())


SPAN_CATEGORIES = ["exact", "contains gold (too long)", "inside gold (too short)", "partial overlap", "miss"]


def span_category(gold, pred):
    """How a predicted spoiler relates to a single gold span (phrase and passage posts).

    - exact: the same text (ignoring case, punctuation and spacing);
    - contains gold (too long): the gold text is inside the prediction, plus extra words;
    - inside gold (too short): the prediction is part of the gold text;
    - partial overlap: they share words (SQuAD token F1 > 0) but neither contains the other;
    - miss: no word in common.
    """
    g, p = squash(gold), squash(pred)
    if not p:
        return "miss"
    if g == p:
        return "exact"
    if g and f" {g} " in f" {p} ":
        return "contains gold (too long)"
    if f" {p} " in f" {g} ":
        return "inside gold (too short)"
    return "partial overlap" if token_f1(gold, pred) > 0 else "miss"


def multi_match(gold_items, pred_items, threshold=0.5):
    """Item-level matching for multi posts: (gold items found, predicted items that match a gold item).

    A gold item counts as found if some predicted item has a token F1 of at least `threshold` with it.
    """
    found = sum(any(token_f1(g, p) >= threshold for p in pred_items) for g in gold_items)
    correct = sum(any(token_f1(g, p) >= threshold for g in gold_items) for p in pred_items)
    return found, correct


COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def promised_count(post):
    """The number of items a post promises ("5 ways", "three reasons"), or None. Only 2-20 counts."""
    for token in re.findall(r"\b\w+\b", post.lower()):
        n = COUNT_WORDS.get(token, int(token) if token.isdigit() else None)
        if n is not None and 2 <= n <= 20:
            return n
    return None
