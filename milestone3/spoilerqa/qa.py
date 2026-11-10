"""Extractive QA helpers: sliding-window features, answer labels, and decoding spans back into spoilers.

The post is the question and the context (title + paragraphs) is the passage. Articles are longer than
one model input, so each one is split into overlapping windows (384 tokens, 128-token overlap by default).
"""
import numpy as np
import torch

from .data import build_context, post_text, spoiler_offsets

# Longest answer (in tokens) we allow per spoiler type when decoding; the notebooks tune these on dev.
DEFAULT_MAX_ANSWER_TOKENS = {"phrase": 16, "passage": 96, "multi": 48}


class FeatureDataset(torch.utils.data.Dataset):
    """Wraps tokenized features (a dict of equal-length lists) for the HF Trainer."""

    def __init__(self, features, keys):
        self.features = features
        self.keys = keys

    def __len__(self):
        return len(self.features["input_ids"])

    def __getitem__(self, i):
        return {k: torch.tensor(self.features[k][i]) for k in self.keys}


def _tokenize(tokenizer, questions, contexts, max_length, stride):
    return tokenizer(
        questions, contexts,
        truncation="only_second", max_length=max_length, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length",
    )


def training_examples(records):
    """One (question, context, answer start, answer end) per spoiler span that can be located.

    Multi spoilers become one example per span. Spans we can't find in the article are skipped.
    """
    examples, skipped = [], 0
    for r in records:
        context = build_context(r)
        offsets, _ = spoiler_offsets(r, context)
        for found in offsets:
            if found is None:
                skipped += 1
                continue
            examples.append({"question": post_text(r), "context": context,
                             "answer_start": found[0], "answer_end": found[1]})
    return examples, skipped


def training_features(tokenizer, examples, max_length=384, stride=128):
    """Tokenize examples into windows and label each window with the answer's token positions.

    A window that doesn't contain the whole answer is labeled with the first token (index 0), the usual
    convention for sliding-window QA training.
    """
    enc = _tokenize(tokenizer, [e["question"] for e in examples], [e["context"] for e in examples],
                    max_length, stride)
    starts, ends = [], []
    for i, offsets in enumerate(enc["offset_mapping"]):
        example = examples[enc["overflow_to_sample_mapping"][i]]
        seq_ids = enc.sequence_ids(i)
        ctx_tokens = [j for j, s in enumerate(seq_ids) if s == 1]
        first, last = ctx_tokens[0], ctx_tokens[-1]
        a_start, a_end = example["answer_start"], example["answer_end"]
        if offsets[first][0] > a_start or offsets[last][1] < a_end:
            starts.append(0)
            ends.append(0)
            continue
        s = first
        while s <= last and offsets[s][1] <= a_start:
            s += 1
        e = last
        while e >= first and offsets[e][0] >= a_end:
            e -= 1
        starts.append(s)
        ends.append(e)
    enc["start_positions"] = starts
    enc["end_positions"] = ends
    return enc


def inference_features(tokenizer, records, max_length=384, stride=128):
    """Windows for prediction. Keeps, per window, which record it came from and the character offsets of
    context tokens (None for question and special tokens)."""
    contexts = [build_context(r) for r in records]
    enc = _tokenize(tokenizer, [post_text(r) for r in records], contexts, max_length, stride)
    enc["example_index"] = list(enc["overflow_to_sample_mapping"])
    enc["context_offsets"] = [
        [o if s == 1 else None for o, s in zip(offsets, enc.sequence_ids(i))]
        for i, offsets in enumerate(enc["offset_mapping"])
    ]
    return enc, contexts


def candidate_spans(enc, contexts, start_logits, end_logits, n_best=30, max_answer_tokens=128):
    """All plausible answer spans per record, best first: (score, start_char, end_char, n_tokens).

    Score = start logit + end logit. Each window proposes its n_best start and end tokens; spans that
    leave the context, end before they start or are too long are dropped. The same character span seen
    from two overlapping windows keeps its best score.
    """
    best = [dict() for _ in contexts]
    for i in range(len(enc["input_ids"])):
        offsets = enc["context_offsets"][i]
        spans = best[enc["example_index"][i]]
        s_logits, e_logits = start_logits[i], end_logits[i]
        for s in np.argsort(s_logits)[::-1][:n_best]:
            if offsets[s] is None:
                continue
            for e in np.argsort(e_logits)[::-1][:n_best]:
                if offsets[e] is None or e < s or e - s + 1 > max_answer_tokens:
                    continue
                key = (offsets[s][0], offsets[e][1])
                score = float(s_logits[s] + e_logits[e])
                if key not in spans or score > spans[key][0]:
                    spans[key] = (score, e - s + 1)
    return [sorted(((sc, a, b, n) for (a, b), (sc, n) in spans.items()), reverse=True) for spans in best]


def decode(candidates, context, spoiler_type, max_answer_tokens=None, k=5):
    """Turn ranked candidates into a spoiler: one span for phrase/passage, up to k non-overlapping spans
    (in reading order) for multi."""
    caps = max_answer_tokens or DEFAULT_MAX_ANSWER_TOKENS
    allowed = [c for c in candidates if c[3] <= caps[spoiler_type]] or candidates
    if not allowed:
        return [] if spoiler_type == "multi" else ""
    if spoiler_type != "multi":
        _, a, b, _ = allowed[0]
        return context[a:b].strip()
    chosen = []
    for _, a, b, _ in allowed:
        if all(b <= x or a >= y for x, y in chosen):
            chosen.append((a, b))
        if len(chosen) == k:
            break
    return [context[a:b].strip() for a, b in sorted(chosen)]
