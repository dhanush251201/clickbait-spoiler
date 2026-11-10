"""Helpers for the generative models (Flan-T5 and Llama 3.1): inputs, targets, generation, and turning the
generated text back into a typed spoiler.

Both models write the spoiler type first and then the spoiler, so one model does both subtasks:

    Flan-T5 (one line):     "phrase: Gwyneth Paltrow"            "multi: bacon | eggs | toast"
    Llama (one per line):   "type: phrase\nspoiler: Gwyneth Paltrow"
                            "type: multi\nspoiler: bacon\nspoiler: eggs\nspoiler: toast"

T5's tokenizer drops newlines, hence the one-line format there. "|" is a single T5 token and never occurs in a
gold spoiler, so it can separate the items of a multi spoiler.
"""
import re

import torch
from sklearn.metrics import balanced_accuracy_score
from transformers import Trainer

from .data import TYPES, build_context, post_text, spoiler_offsets, spoiler_type
from .metrics import as_text, bleu

T5_SEP = " | "


# ---------------------------------------------------------------- shared

def clean_span(span):
    return " ".join(span.split())


def guess_type(spans):
    """Fallback when the model wrote no valid type: several spans -> multi; at most 5 words -> phrase (that rule
    is about 99% accurate on gold spoilers); anything longer -> passage."""
    spans = [s for s in spans if s.strip()]
    if len(spans) > 1:
        return "multi"
    return "phrase" if spans and len(spans[0].split()) <= 5 else "passage"


def to_prediction(uuid, spoiler_type_, spans):
    """A prediction in the SemEval run format. Multi spoilers are a list of spans; the others one string."""
    spans = [clean_span(s) for s in spans if s.strip()]
    return {"uuid": uuid, "spoilerType": spoiler_type_,
            "spoiler": spans if spoiler_type_ == "multi" else " ".join(spans)}


def truncate_text(tokenizer, text, max_tokens):
    """The longest start of `text` that fits in max_tokens tokens (without special tokens), cut at a token end."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc["input_ids"]) <= max_tokens:
        return text
    return text[:enc["offset_mapping"][max_tokens - 1][1]]


def spoiler_visible(record, context):
    """True if at least one gold span lies completely inside `context`, a (possibly truncated) build_context."""
    offsets, _ = spoiler_offsets(record)
    return any(o is not None and o[1] <= len(context) for o in offsets)


def score_predictions(records, preds):
    """Dev BLEU and type balanced accuracy for a list of predictions (same order as records)."""
    truths = [as_text(r["spoiler"]) for r in records]
    return {"bleu": bleu(truths, [as_text(p["spoiler"]) for p in preds]),
            "type_balanced_accuracy": float(balanced_accuracy_score([spoiler_type(r) for r in records],
                                                                    [p["spoilerType"] for p in preds]))}


class GenerationEvalTrainer(Trainer):
    """A Trainer whose evaluation generates the dev spoilers and reports BLEU and type balanced accuracy.

    `predict_fn(model, records)` must return one prediction dict per record. The model writes the type itself,
    so this dev BLEU is end to end (unlike the QA models, which are evaluated with gold types during training).
    """

    def __init__(self, *args, predict_fn=None, dev_records=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_fn, self.dev_records = predict_fn, dev_records

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        preds = self.predict_fn(self.model, self.dev_records)
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in score_predictions(self.dev_records, preds).items()}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


def _autocast(model, bf16):
    device_type = "cuda" if next(model.parameters()).is_cuda else "cpu"
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=bf16)


# ---------------------------------------------------------------- Flan-T5

def t5_source(post, context):
    return f"question: {post} context: {context}"


def t5_input(tokenizer, record, max_tokens):
    """The source text for one post, with the article cut so the whole input fits in max_tokens tokens.
    Returns (source, kept_context)."""
    post = post_text(record)
    overhead = len(tokenizer(t5_source(post, ""))["input_ids"])  # prefix + end-of-sequence token
    context = truncate_text(tokenizer, build_context(record), max_tokens - overhead)
    return t5_source(post, context), context


def t5_target(record, spoiler_type_=None):
    spans = [clean_span(s) for s in record["spoiler"]]
    return f"{spoiler_type_ or spoiler_type(record)}: " + T5_SEP.join(spans)


def parse_t5(text):
    """'multi: a | b' -> ('multi', ['a', 'b']). Without a valid type prefix, the type is guessed."""
    match = re.match(r"\s*(phrase|passage|multi)\s*:(.*)", text, flags=re.S | re.I)
    spoiler_type_, body = (match.group(1).lower(), match.group(2)) if match else (None, text)
    spans = [s.strip() for s in body.split(T5_SEP.strip()) if s.strip()]
    return spoiler_type_ or guess_type(spans), spans


@torch.no_grad()
def seq2seq_generate(model, tokenizer, sources, max_source_tokens, max_new_tokens, num_beams=4, batch_size=16,
                     decoder_prefix=None, bf16=False):
    """Generate one output per source (longest sources first, so memory problems show up at once).

    decoder_prefix (e.g. "phrase:") forces the start of every output; used to decode with gold types.
    """
    model.eval()
    device = next(model.parameters()).device
    prefix = None
    if decoder_prefix:
        prefix = [model.config.decoder_start_token_id] + tokenizer(decoder_prefix, add_special_tokens=False)["input_ids"]
    order = sorted(range(len(sources)), key=lambda i: len(sources[i]), reverse=True)
    outputs = [None] * len(sources)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tokenizer([sources[i] for i in idx], max_length=max_source_tokens, truncation=True, padding=True,
                        return_tensors="pt").to(device)
        extra = {"decoder_input_ids": torch.tensor([prefix] * len(idx), device=device)} if prefix else {}
        with _autocast(model, bf16):
            generated = model.generate(**enc, **extra, max_new_tokens=max_new_tokens, num_beams=num_beams,
                                       use_cache=True)
        for i, text in zip(idx, tokenizer.batch_decode(generated, skip_special_tokens=True)):
            outputs[i] = text
    return outputs


# ---------------------------------------------------------------- Llama (chat models)

LLAMA_SYSTEM = (
    "You spoil clickbait. You get a clickbait post and the article it links to. Reply with the spoiler type "
    "(phrase, passage or multi) and the spoiler: the text from the article that gives away what the post teases, "
    "copied word for word. Write 'type: <type>' on the first line and the spoiler on a line starting with "
    "'spoiler: '. For multi, write one 'spoiler: ' line per item."
)


def llama_messages(post, context):
    return [{"role": "system", "content": LLAMA_SYSTEM},
            {"role": "user", "content": f"Post: {post}\n\nArticle:\n{context}"}]


def llama_prompt_ids(tokenizer, record, article_tokens):
    """Token ids of the chat prompt (ending with the assistant header) for one post, with the article cut to
    article_tokens tokens. Returns (ids, kept_context)."""
    context = truncate_text(tokenizer, build_context(record), article_tokens)
    text = tokenizer.apply_chat_template(llama_messages(post_text(record), context), tokenize=False,
                                         add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)["input_ids"], context


def llama_answer_prefix(spoiler_type_):
    return f"type: {spoiler_type_}\nspoiler:"


def llama_target(record):
    return f"type: {spoiler_type(record)}\n" + "\n".join(f"spoiler: {clean_span(s)}" for s in record["spoiler"])


def parse_llama(text):
    """'type: multi\\nspoiler: a\\nspoiler: b' -> ('multi', ['a', 'b']). Lenient about case and extra lines."""
    spoiler_type_, spans = None, []
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"type\s*:\s*(\w+)", line, flags=re.I)
        if match:
            if spoiler_type_ is None and match.group(1).lower() in TYPES:
                spoiler_type_ = match.group(1).lower()
            continue
        match = re.match(r"spoiler\s*:\s*(.*)", line, flags=re.I)
        if match and match.group(1).strip():
            spans.append(match.group(1).strip())
    if not spans:  # no "spoiler:" lines: take whatever text there is
        spans = [line.strip() for line in text.splitlines()
                 if line.strip() and not re.match(r"type\s*:", line.strip(), flags=re.I)]
    return spoiler_type_ or guess_type(spans), spans


@torch.no_grad()
def causal_generate(model, tokenizer, prompts, max_new_tokens, batch_size=8, bf16=False):
    """Greedy generation for a list of prompt token-id lists (left-padded batches, longest first).
    Returns only the newly generated text for each prompt."""
    model.eval()
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id
    eos = model.generation_config.eos_token_id
    eos = sorted({*(eos if isinstance(eos, list) else [eos]), tokenizer.eos_token_id} - {None})
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    outputs = [None] * len(prompts)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        width = max(len(prompts[i]) for i in idx)
        input_ids = torch.tensor([[pad] * (width - len(prompts[i])) + prompts[i] for i in idx], device=device)
        attention = torch.tensor([[0] * (width - len(prompts[i])) + [1] * len(prompts[i]) for i in idx], device=device)
        with _autocast(model, bf16):
            generated = model.generate(input_ids=input_ids, attention_mask=attention, max_new_tokens=max_new_tokens,
                                       do_sample=False, temperature=None, top_p=None, top_k=None,
                                       pad_token_id=pad, eos_token_id=eos, use_cache=True)
        for i, text in zip(idx, tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)):
            outputs[i] = text
    return outputs
