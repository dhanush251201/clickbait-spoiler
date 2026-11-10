"""Evaluation metrics for clickbait spoiling, following the official SemEval-2023 Task 5 evaluator.

The official evaluator is `semeval23/clickbait-spoiling-eval.py` in https://github.com/pan-webis-de/pan-code
(Docker image webis/pan-clickbait-spoiling-evaluator). BLEU, METEOR, BERTScore and the type-classification
scores below reproduce its definitions; exact match and token F1 are SQuAD-style extras of ours.
"""
import re
import shutil
import string
import subprocess
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import nltk
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from .data import TYPES

METEOR_DIR = Path(__file__).resolve().parents[1] / "tools" / "meteor-1.5"


def ensure_nltk_data():
    for resource, name in [("tokenizers/punkt_tab", "punkt_tab"), ("corpora/stopwords", "stopwords"),
                           ("corpora/wordnet", "wordnet")]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(name, quiet=True)


def as_text(spoiler):
    """A spoiler as one string, the way the official evaluator does it: list items joined by spaces, no newlines."""
    if isinstance(spoiler, list):
        spoiler = " ".join(spoiler)
    return spoiler.replace("\n", " ").strip()


# ---------------------------------------------------------------- BLEU (official definition)

_STOPWORDS = None


def _bleu_tokens(text):
    # Same steps as the official stopfilter(): drop stopwords (case-sensitive, before lowercasing),
    # then lowercase and drop tokens that are a substring of string.punctuation.
    global _STOPWORDS
    if _STOPWORDS is None:
        from nltk.corpus import stopwords
        _STOPWORDS = set(stopwords.words("english"))
    tokens = [t for t in nltk.word_tokenize(text.replace("\n", "")) if t not in _STOPWORDS]
    return [t.lower() for t in tokens if t not in string.punctuation]


def _bleu_weights(truth_tokens, pred_tokens):
    # Short spoilers get a lower maximum n-gram order, so a 2-word spoiler isn't scored on 4-grams.
    if len(truth_tokens) > 3 and len(pred_tokens) > 3:
        return (1 / 4, 1 / 4, 1 / 4, 1 / 4)
    if len(truth_tokens) > 2 and len(pred_tokens) > 2:
        return (1 / 3, 1 / 3, 1 / 3)
    if len(truth_tokens) > 1 and len(pred_tokens) > 1:
        return (1 / 2, 1 / 2)
    return (1.0, 0.0)


def bleu_scores(truths, preds):
    """Per-item BLEU exactly as the official evaluator computes it (NLTK sentence_bleu, no smoothing)."""
    from nltk.translate.bleu_score import sentence_bleu
    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # NLTK warns about zero n-gram overlaps; the score is 0 then, as intended
        for truth, pred in zip(truths, preds):
            t, p = _bleu_tokens(truth), _bleu_tokens(pred)
            scores.append(float(sentence_bleu([t], p, weights=_bleu_weights(t, p))))
    return scores


def bleu(truths, preds):
    scores = bleu_scores(truths, preds)
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------- METEOR

def meteor_available():
    return (METEOR_DIR / "meteor-1.5.jar").exists() and (METEOR_DIR / "data" / "paraphrase-en.gz").exists() \
        and shutil.which("java") is not None


def meteor(truths, preds):
    """Corpus METEOR 1.5 (-l en -norm -t adq), as in the official evaluator.

    The paraphrase table is passed with -a: it is the same file the jar loads by default, but passing it
    explicitly also works when the project path contains spaces (the jar mis-encodes those).
    Returns (score, "meteor-1.5") or, without Java/the jar, NLTK's METEOR averaged over items as an
    approximation: (score, "nltk").
    """
    if not meteor_available():
        from nltk.translate.meteor_score import meteor_score
        scores = [meteor_score([nltk.word_tokenize(t)], nltk.word_tokenize(p)) for t, p in zip(truths, preds)]
        return (sum(scores) / len(scores) if scores else 0.0), "nltk"

    with tempfile.TemporaryDirectory() as tmp:
        truth_file, pred_file = Path(tmp) / "truths.txt", Path(tmp) / "preds.txt"
        truth_file.write_text("".join(t + "\n" for t in truths), encoding="utf-8")
        pred_file.write_text("".join(p + "\n" for p in preds), encoding="utf-8")
        cmd = ["java", "-Xmx2G", "-jar", str(METEOR_DIR / "meteor-1.5.jar"), str(truth_file), str(pred_file),
               "-l", "en", "-norm", "-t", "adq", "-a", str(METEOR_DIR / "data" / "paraphrase-en.gz")]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8")
    match = re.search(r"Final score:\s*([0-9.eE+-]+)", output)
    if not match:
        raise ValueError(f"Could not read the METEOR score from:\n{output[-2000:]}")
    return float(match.group(1)), "meteor-1.5"


# ---------------------------------------------------------------- BERTScore

_BERT_SCORER = None


def _patch_bert_score_empty_strings():
    # bert_score 0.3.12 encodes an empty string with tokenizer.build_inputs_with_special_tokens([]), which
    # transformers 5 removed. Encode it the same way (special tokens only) through the regular tokenizer call.
    import bert_score.utils as bs_utils
    if getattr(bs_utils.sent_encode, "patched", False):
        return
    original = bs_utils.sent_encode

    def sent_encode(tokenizer, sent):
        if sent.strip() == "":
            return tokenizer("", add_special_tokens=True)["input_ids"]
        return original(tokenizer, sent)

    sent_encode.patched = True
    bs_utils.sent_encode = sent_encode


def bertscore_items(truths, preds):
    """Per-item BERTScore F1 with bert_score's English default (roberta-large, no idf, no rescaling), as in
    the official evaluator. Items are scored independently, so subset means equal subset scores."""
    global _BERT_SCORER
    from bert_score import BERTScorer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    _patch_bert_score_empty_strings()
    if _BERT_SCORER is None:
        _BERT_SCORER = BERTScorer(lang="en")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, f1 = _BERT_SCORER.score(preds, truths)
    return [float(x) for x in f1]


def bertscore(truths, preds):
    scores = bertscore_items(truths, preds)
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------- exact match / token F1 (SQuAD-style extras)

def _squad_normalize(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(truth, pred):
    return float(_squad_normalize(truth) == _squad_normalize(pred))


def token_f1(truth, pred):
    t, p = _squad_normalize(truth).split(), _squad_normalize(pred).split()
    if not t or not p:
        return float(t == p)
    common = sum((Counter(t) & Counter(p)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(t)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------- spoiler type classification (task 1)

def _one_vs_rest(y_true, y_pred, label):
    # Official filter_to(): only items where the gold or the predicted label is `label` count.
    kept = [(t == label, p == label) for t, p in zip(y_true, y_pred) if t == label or p == label]
    return [int(t) for t, _ in kept], [int(p) for _, p in kept]


def type_scores(y_true, y_pred):
    scores = {"balanced-accuracy": float(balanced_accuracy_score(y_true, y_pred))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label in TYPES:
            t, p = _one_vs_rest(y_true, y_pred, label)
            scores[f"precision-{label}"] = float(precision_score(t, p, zero_division=0))
            scores[f"recall-{label}"] = float(recall_score(t, p, zero_division=0))
            scores[f"f1-{label}"] = float(f1_score(t, p, zero_division=0))
    return scores
