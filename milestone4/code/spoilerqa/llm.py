"""Prompted LLMs for clickbait spoiling (Extension 2).

The same prompt goes to every model: a system prompt that explains the task, k worked examples from the training
set (as earlier user/assistant turns), then the post and its article. Every model answers in the same JSON shape,

    {"type": "phrase" | "passage" | "multi", "spoilers": ["...", ...]}

enforced by each provider's structured-output option. Three setups:
- "article": the post and the article (the main setup);
- "closedbook": the post only, to see how much a model already knows about this public dataset.

Backends:
- Ollama, for the local models (Llama 3.1 8B, GPT-OSS 20B), one request at a time;
- Anthropic, for Claude (Sonnet 5.5, Haiku 5.5), through the Message Batches API (half price) or directly;
- OpenAI, for ChatGPT (GPT-6.1 Sol, GPT-6 Luna), through the Responses API, in a Batch (half price) or directly.

Every answer is appended to a raw file (one JSON line per post) as soon as it arrives, so an interrupted run
picks up where it stopped. Raw answers are turned into predictions afterwards, with or without snapping each
spoiler to the closest span of the article.
"""
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rapidfuzz import fuzz

from .data import TYPES, build_context, load_jsonl, post_text, spoiler_type
from .gen import clean_span, guess_type, to_prediction

# ----------------------------------------------------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------------------------------------------------

# Articles longer than this are cut (3 of the 1,000 test posts; about 6,000 tokens), so every model reads the same
# input and the local models fit in their context window.
MAX_ARTICLE_CHARS = 24_000

# Worked examples from the training set: short articles, spoilers found exactly at their annotated positions.
# The order matters: the first three (one per type) are the 3-shot prompt, all six the 6-shot prompt.
FEWSHOT_UUIDS = [
    "81e4fd0f-0a03-4c6d-9223-b6b11513bbf6",  # phrase:  secret code for an early Pumpkin Spice Latte -> "PSL10"
    "6e53b422-0d86-48d0-9e1d-c1ac29ced0c1",  # passage: why some backpacks have a diamond patch
    "2b1e56a9-bb2c-4efd-8c1a-378edabf7f9b",  # multi:   who is in the "Mary Tyler Moore" reunion (3 names)
    "29061666-a776-4205-b269-4ef89a8036af",  # phrase:  the surprise in a box of Walmart cookies -> "$100 bill"
    "f12bc32a-2b74-4826-b4e1-7f7138fc7d72",  # passage: Bill Murray's reason for joining the Ghostbusters remake
    "93f49a6b-6384-41e8-8de1-319a2eb8780a",  # multi:   two medications that are dangerous together
]

TYPE_GUIDE = """Spoiler types:
- "phrase": one short answer of a few words, such as a name, a number or a thing.
- "passage": an answer that needs a sentence or two.
- "multi": several separate answers, for example when the post promises a list ("5 ways to ...") or asks about more than one thing. Give each part as its own spoiler, at most 5."""

SYSTEM_ARTICLE = f"""You spoil clickbait. You get a clickbait post and the article it links to. The post teases something without saying it; the spoiler is the text from the article that gives it away.

{TYPE_GUIDE}

Copy each spoiler word for word from the article, using the shortest text that fully answers the post. Do not paraphrase, explain or add anything.

Reply with JSON only: {{"type": "phrase" | "passage" | "multi", "spoilers": ["..."]}}. For phrase and passage, give exactly one spoiler."""

SYSTEM_CLOSED_BOOK = f"""You spoil clickbait. You get a clickbait post, but not the article it links to. The post teases something without saying it. Give your best guess of the spoiler: the information the post is hiding, written as briefly as the article would put it.

{TYPE_GUIDE}

Do not explain or hedge; if you are not sure, guess.

Reply with JSON only: {{"type": "phrase" | "passage" | "multi", "spoilers": ["..."]}}. For phrase and passage, give exactly one spoiler."""

SETUPS = {"article": SYSTEM_ARTICLE, "closedbook": SYSTEM_CLOSED_BOOK}

SPOILER_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": TYPES},
        "spoilers": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["type", "spoilers"],
    "additionalProperties": False,
}


def cut_article(text, max_chars=MAX_ARTICLE_CHARS):
    """The article, cut at the last line break or space before max_chars if it is longer."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    return cut[:max(cut.rfind("\n"), cut.rfind(" "), max_chars // 2)]


def user_message(record, setup):
    if setup == "closedbook":
        return f"Post: {post_text(record)}"
    return f"Post: {post_text(record)}\n\nArticle:\n{cut_article(build_context(record))}"


def answer_json(record):
    """The gold answer of a worked example, in the reply format."""
    return json.dumps({"type": spoiler_type(record), "spoilers": record["spoiler"]}, ensure_ascii=False)


def fewshot_examples(train_records, k):
    by_uuid = {r["uuid"]: r for r in train_records}
    return [by_uuid[u] for u in FEWSHOT_UUIDS[:k]]


def build_prompt(record, setup, examples):
    """(system prompt, messages): the worked examples as earlier turns, then the post to spoil."""
    messages = []
    for ex in examples:
        messages.append({"role": "user", "content": user_message(ex, setup)})
        messages.append({"role": "assistant", "content": answer_json(ex)})
    messages.append({"role": "user", "content": user_message(record, setup)})
    return SETUPS[setup], messages


# ----------------------------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------------------------

MODELS = {
    "llama3.1-8b": {"backend": "ollama", "model": "llama3.1:8b",
                    "label": "Llama 3.1 8B Instruct (Ollama, 4-bit)"},
    "gpt-oss-20b": {"backend": "ollama", "model": "gpt-oss:20b", "think": "low",
                    "label": "GPT-OSS 20B (Ollama, low reasoning)"},
    "claude-sonnet-5.5": {"backend": "anthropic", "model": "claude-sonnet-5-5", "effort": "low",
                          "label": "Claude Sonnet 5.5 (low effort)"},
    "claude-haiku-5.5": {"backend": "anthropic", "model": "claude-haiku-5-5", "effort": "low",
                         "label": "Claude Haiku 5.5 (low effort)"},
    # ChatGPT: the two OpenAI models priced like Claude Sonnet 5.5 and Claude Haiku 5.5.
    "gpt-6.1-sol": {"backend": "openai", "model": "gpt-6.1-sol", "effort": "low",
                    "label": "GPT-6.1 Sol (low reasoning effort)"},
    "gpt-6-luna": {"backend": "openai", "model": "gpt-6-luna", "effort": "low",
                   "label": "GPT-6 Luna (low reasoning effort)"},
}

# List prices in US$ per million tokens at standard rates (Anthropic, October 2026, prompts up to 100K tokens for
# Haiku; OpenAI's pricing page, November 2026, short context). Both batch APIs charge half of these.
PRICES = {
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
    "gpt-6.1-sol": {"input": 2.00, "output": 10.00, "cache_read": 0.10},
    "gpt-6-luna": {"input": 0.10, "output": 0.50, "cache_read": 0.01},
}
BATCH_BACKENDS = {"anthropic", "openai"}  # these run through a batch API (half price) unless batch=False


def add_openai_model(key, model, effort="low", label=None, prices=None):
    """Register a ChatGPT model, e.g. add_openai_model("gpt-x", "gpt-x", prices={"input": .., "output": ..})."""
    MODELS[key] = {"backend": "openai", "model": model, "effort": effort, "label": label or model}
    if prices:
        PRICES[model] = prices


def backend_available(key):
    """(available, reason) for a model key: is the server up / the API key set?"""
    spec = MODELS[key]
    if spec["backend"] == "ollama":
        try:
            import ollama
            names = {m.model for m in ollama.Client(host=os.environ.get("OLLAMA_HOST")).list().models}
        except Exception as e:  # server not running
            return False, f"Ollama not reachable ({type(e).__name__})"
        return (spec["model"] in names, "ok" if spec["model"] in names else f"run: ollama pull {spec['model']}")
    if spec["backend"] == "anthropic":
        return (bool(os.environ.get("ANTHROPIC_API_KEY")), "ok" if os.environ.get("ANTHROPIC_API_KEY")
                else "ANTHROPIC_API_KEY not set")
    return (bool(os.environ.get("OPENAI_API_KEY")), "ok" if os.environ.get("OPENAI_API_KEY")
            else "OPENAI_API_KEY not set")


class RawWriter:
    """Appends one JSON line per answer to the raw file, safely from several threads."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, row):
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _progress(done, total, start, every=25):
    if done % every == 0 or done == total:
        rate = (time.time() - start) / max(done, 1)
        print(f"  {done}/{total} posts, {rate:.1f} s/post, about {rate * (total - done) / 60:.0f} min left", flush=True)


class OllamaBackend:
    """Local models through the Ollama server, one request at a time (the GPU runs one model anyway)."""

    def __init__(self, spec, schema=SPOILER_SCHEMA, schema_name="spoiler"):
        import ollama
        self.spec, self.schema = spec, schema
        self.client = ollama.Client(host=os.environ.get("OLLAMA_HOST"), timeout=900)

    def ask(self, system, messages):
        start = time.time()
        options = {"temperature": 0, "seed": 0, "num_ctx": 16384, "num_predict": 4096}
        kwargs = {"think": self.spec["think"]} if "think" in self.spec else {}
        resp = self.client.chat(model=self.spec["model"], messages=[{"role": "system", "content": system}, *messages],
                                format=self.schema, options=options, keep_alive="30m", **kwargs)
        thinking = getattr(resp.message, "thinking", None) or ""
        return {"text": resp.message.content, "model": self.spec["model"], "stop_reason": resp.done_reason,
                "usage": {"input_tokens": resp.prompt_eval_count, "output_tokens": resp.eval_count,
                          "thinking_chars": len(thinking)},
                "seconds": round(time.time() - start, 2)}

    def run(self, prompts, writer, **_):
        start = time.time()
        for i, (uuid, (system, messages)) in enumerate(prompts.items(), 1):
            writer.write({"uuid": uuid, **self.ask(system, messages)})
            _progress(i, len(prompts), start)


class AnthropicBackend:
    """Claude through the Anthropic API.

    - Structured outputs (output_config.format) guarantee the JSON reply.
    - Thinking stays adaptive at low effort: Claude Sonnet 5.5 and Haiku 5.5 decide how much to think; low effort
      keeps it short for this extraction task. Sampling parameters (temperature) are not accepted by these models.
    - The worked examples are marked for prompt caching, so repeated requests read them at a tenth of the price.
    - No refusal fallback: a fallback would let another model answer inside a model comparison, and the Batches
      API rejects it anyway. Refusals are recorded in the raw file (stop_reason "refusal") instead.
    """

    def __init__(self, spec, schema=SPOILER_SCHEMA, schema_name="spoiler"):
        import anthropic
        self.spec, self.schema = spec, schema
        self.client = anthropic.Anthropic(max_retries=5)

    def params(self, system, messages):
        msgs = [dict(m) for m in messages]
        if len(msgs) > 1:  # cache everything up to the last worked example
            last_example = msgs[-2]
            last_example["content"] = [{"type": "text", "text": last_example["content"],
                                        "cache_control": {"type": "ephemeral"}}]
        return {
            "model": self.spec["model"],
            "max_tokens": 16000,
            "system": system,
            "messages": msgs,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.spec.get("effort", "low"),
                              "format": {"type": "json_schema", "schema": self.schema}},
        }

    @staticmethod
    def _row(message):
        text = "".join(b.text for b in message.content if b.type == "text")
        u = message.usage
        return {"text": text, "model": message.model, "stop_reason": message.stop_reason,
                "usage": {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                          "cache_read_input_tokens": u.cache_read_input_tokens or 0,
                          "cache_creation_input_tokens": u.cache_creation_input_tokens or 0}}

    def ask(self, system, messages):
        start = time.time()
        message = self.client.messages.create(**self.params(system, messages))
        return {**self._row(message), "seconds": round(time.time() - start, 2)}

    def cancel_batch(self, batch_id):
        """Cancels a batch that is still running. True if it was cancelled, False if it had already ended."""
        if self.client.messages.batches.retrieve(batch_id).processing_status == "in_progress":
            self.client.messages.batches.cancel(batch_id)
            return True
        return False

    def run(self, prompts, writer, batch=True, wait=True, workers=4, poll_seconds=30):
        """With batch=True, submits one batch (or picks up the one already submitted for this raw file) and, if
        wait=True, polls until it ends and writes the answers. wait=False returns right after submitting, so
        several batches (and the local models) can run at the same time; call again later to collect.
        With batch=False, asks directly with `workers` parallel requests (after collecting any earlier batch)."""
        if not batch:
            return _run_threads(self.ask, _drain_pending_batch(self, prompts, writer), writer, workers)
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        state_path = writer.path.parent / f"{writer.path.stem}.batch.json"
        if state_path.exists():  # a batch from an earlier, interrupted call: collect it instead of resubmitting
            state = json.loads(state_path.read_text())
            print(f"  resuming batch {state['batch_id']}")
        else:
            requests = [Request(custom_id=uuid, params=MessageCreateParamsNonStreaming(**self.params(*prompt)))
                        for uuid, prompt in prompts.items()]
            batch_obj = self.client.messages.batches.create(requests=requests)
            state = {"batch_id": batch_obj.id, "submitted": time.time(), "requests": len(requests)}
            state_path.write_text(json.dumps(state))
            print(f"  submitted batch {batch_obj.id} with {len(requests)} requests")
        while True:
            batch_obj = self.client.messages.batches.retrieve(state["batch_id"])
            if batch_obj.processing_status == "ended":
                break
            c = batch_obj.request_counts
            if not wait:
                print(f"  batch {batch_obj.processing_status}: {c.succeeded} done, {c.processing} processing; "
                      "collect it later", flush=True)
                return
            print(f"  batch {batch_obj.processing_status}: {c.succeeded} done, {c.processing} processing", flush=True)
            time.sleep(poll_seconds)
        failed = 0
        for result in self.client.messages.batches.results(state["batch_id"]):
            if result.result.type == "succeeded":
                writer.write({"uuid": result.custom_id, **self._row(result.result.message), "seconds": None,
                              "batch_id": state["batch_id"]})
            else:  # errored / canceled / expired: left out, so the next call asks again
                failed += 1
        minutes = (time.time() - state["submitted"]) / 60
        log_run(writer.path, {"batch_id": state["batch_id"], "requests": state["requests"], "failed": failed,
                              "minutes": round(minutes, 1)})
        state_path.unlink()
        print(f"  batch ended after {minutes:.0f} min; {failed} requests failed" + (" (re-run to retry)" if failed else ""))


class OpenAIBackend:
    """ChatGPT models through the OpenAI Responses API with a strict JSON schema and low reasoning effort.

    Like Claude, they run through OpenAI's Batch API (half price) by default: the requests are uploaded as a
    JSONL file, the batch runs on OpenAI's side, and the answers are downloaded when it completes. OpenAI caches
    repeated prompt beginnings (the worked examples) automatically. store=False keeps the answers off OpenAI's
    stored-responses list.
    """

    def __init__(self, spec, schema=SPOILER_SCHEMA, schema_name="spoiler"):
        import openai
        self.spec, self.schema, self.schema_name = spec, schema, schema_name
        self.client = openai.OpenAI(max_retries=5)

    def body(self, system, messages):
        body = {"model": self.spec["model"], "instructions": system, "input": messages, "store": False,
                "max_output_tokens": 16000,
                "text": {"format": {"type": "json_schema", "name": self.schema_name, "schema": self.schema,
                                    "strict": True}}}
        if self.spec.get("effort"):
            body["reasoning"] = {"effort": self.spec["effort"]}
        return body

    @staticmethod
    def _row(resp):
        u = resp.usage
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        reasoning = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
        # OpenAI counts cached tokens inside input_tokens; store them apart, as Anthropic reports them.
        return {"text": resp.output_text, "model": resp.model, "stop_reason": resp.status,
                "usage": {"input_tokens": u.input_tokens - cached, "output_tokens": u.output_tokens,
                          "cache_read_input_tokens": cached, "reasoning_tokens": reasoning}}

    def ask(self, system, messages):
        start = time.time()
        resp = self.client.responses.create(**self.body(system, messages))
        return {**self._row(resp), "seconds": round(time.time() - start, 2)}

    def cancel_batch(self, batch_id):
        """Cancels a batch that is still running. True if it was cancelled, False if it had already ended."""
        if self.client.batches.retrieve(batch_id).status in ("validating", "in_progress"):
            self.client.batches.cancel(batch_id)
            return True
        return False

    def run(self, prompts, writer, batch=True, wait=True, workers=8, poll_seconds=30):
        """Same contract as AnthropicBackend.run: batch=False asks directly with `workers` parallel requests.

        A batch can fail as a whole: OpenAI limits the tokens queued per model in batches (1.35M for GPT-6.1 Sol on
        our account, less than a 1,000-post run with 6 worked examples). Direct requests have no such limit.
        """
        if not batch:
            return _run_threads(self.ask, _drain_pending_batch(self, prompts, writer), writer, workers)
        from openai.types.responses import Response

        state_path = writer.path.parent / f"{writer.path.stem}.batch.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            print(f"  resuming batch {state['batch_id']}")
        else:
            input_path = writer.path.parent / f"{writer.path.stem}.batch-input.jsonl"
            with open(input_path, "w", encoding="utf-8") as f:
                for uuid, prompt in prompts.items():
                    f.write(json.dumps({"custom_id": uuid, "method": "POST", "url": "/v1/responses",
                                        "body": self.body(*prompt)}, ensure_ascii=False) + "\n")
            with open(input_path, "rb") as f:
                uploaded = self.client.files.create(file=f, purpose="batch")
            batch_obj = self.client.batches.create(input_file_id=uploaded.id, endpoint="/v1/responses",
                                                   completion_window="24h")
            input_path.unlink()
            state = {"batch_id": batch_obj.id, "submitted": time.time(), "requests": len(prompts)}
            state_path.write_text(json.dumps(state))
            print(f"  submitted batch {batch_obj.id} with {len(prompts)} requests")
        while True:
            batch_obj = self.client.batches.retrieve(state["batch_id"])
            if batch_obj.status in ("completed", "failed", "expired", "cancelled"):
                break
            c = batch_obj.request_counts
            progress = f"{c.completed}/{c.total} done" if c else "starting"
            if not wait:
                print(f"  batch {batch_obj.status}: {progress}; collect it later", flush=True)
                return
            print(f"  batch {batch_obj.status}: {progress}", flush=True)
            time.sleep(poll_seconds)
        written, failed, first_error = 0, 0, None
        if batch_obj.output_file_id:
            for line in self.client.files.content(batch_obj.output_file_id).text.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                response = item.get("response") or {}
                if response.get("status_code") == 200:
                    resp = Response.model_validate(response["body"])
                    writer.write({"uuid": item["custom_id"], **self._row(resp), "seconds": None,
                                  "batch_id": state["batch_id"]})
                    written += 1
                else:
                    failed += 1
                    first_error = first_error or item.get("error") or response.get("body")
        if batch_obj.error_file_id:
            for line in self.client.files.content(batch_obj.error_file_id).text.splitlines():
                if line.strip():
                    failed += 1
                    first_error = first_error or json.loads(line).get("error") or json.loads(line).get("response")
        if batch_obj.status != "completed" and written == 0:
            first_error = first_error or getattr(batch_obj, "errors", None)
        minutes = (time.time() - state["submitted"]) / 60
        log_run(writer.path, {"batch_id": state["batch_id"], "requests": state["requests"], "failed": failed,
                              "minutes": round(minutes, 1), "status": batch_obj.status})
        state_path.unlink()
        print(f"  batch {batch_obj.status} after {minutes:.0f} min: {written} answers, {failed} failed"
              + (" (re-run to retry)" if failed else ""))
        if first_error:
            print(f"  first error: {str(first_error)[:300]}")


def _drain_pending_batch(backend, prompts, writer):
    """Before asking directly: keeps the answers of a batch submitted earlier for this raw file.

    A finished batch is downloaded. One still running is cancelled first; the answers it already has are still
    delivered (and billed at batch price). Returns the prompts that still have no answer.
    """
    state_path = writer.path.parent / f"{writer.path.stem}.batch.json"
    if not state_path.exists():
        return prompts
    batch_id = json.loads(state_path.read_text())["batch_id"]
    if backend.cancel_batch(batch_id):
        print(f"  cancelled the unfinished batch {batch_id}; collecting the answers it already has")
    backend.run(prompts, writer, batch=True, wait=True, poll_seconds=10)
    done = {row["uuid"] for row in load_jsonl(writer.path)} if writer.path.exists() else set()
    return {uuid: prompt for uuid, prompt in prompts.items() if uuid not in done}


def _run_threads(ask, prompts, writer, workers, rounds=3, pause_seconds=60):
    """Asks every prompt with `workers` parallel requests. Failed requests (e.g. still rate-limited after the
    SDK's own retries) get up to two more rounds, after a pause and with half the parallel requests."""
    todo, start = dict(prompts), time.time()
    for round_ in range(1, rounds + 1):
        failed, done = {}, 0
        with ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(ask, *prompt): uuid for uuid, prompt in todo.items()}
            for future in as_completed(futures):
                uuid = futures[future]
                try:
                    writer.write({"uuid": uuid, **future.result()})
                except Exception as e:  # left out of the raw file, so it is asked again
                    failed[uuid] = todo[uuid]
                    print(f"  {uuid}: {type(e).__name__}: {e}"[:300])
                done += 1
                _progress(done, len(todo), start)
        if not failed:
            return
        todo = failed
        if round_ < rounds:
            workers = max(1, workers // 2)
            print(f"  {len(failed)} requests failed; asking them again in {pause_seconds} s with {workers} at a time")
            time.sleep(pause_seconds)
    print(f"  {len(todo)} requests still failed (re-run to retry)")


BACKENDS = {"ollama": OllamaBackend, "anthropic": AnthropicBackend, "openai": OpenAIBackend}


def log_run(raw_path, info):
    """Run-level facts (batch ids, wall time) go to run-log.jsonl next to the raw files."""
    with open(Path(raw_path).parent / "run-log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"run": Path(raw_path).stem, "time": time.strftime("%Y-%m-%d %H:%M"), **info}) + "\n")


def run_name(model_key, setup, k, split_tag):
    return f"{model_key}_{setup}_k{k}_{split_tag}"


def ask_all(model_key, prompts, raw_path, schema=SPOILER_SCHEMA, schema_name="spoiler", **run_kwargs):
    """Asks one model every prompt ({id: (system, messages)}) that has no answer in the raw file yet.

    The generic, resumable runner behind run_llm; the summary notebook uses it with its own JSON schema. The raw
    file keys answers by id (stored as "uuid"). Returns {id: raw answer} for the prompts answered so far.
    """
    raw_path = Path(raw_path)
    done = {row["uuid"] for row in load_jsonl(raw_path)} if raw_path.exists() else set()
    todo = {key: prompt for key, prompt in prompts.items() if key not in done}
    print(f"{raw_path.stem}: {len(prompts) - len(todo)} done, {len(todo)} to ask")
    if todo:
        start = time.time()
        backend = BACKENDS[MODELS[model_key]["backend"]](MODELS[model_key], schema=schema, schema_name=schema_name)
        backend.run(todo, RawWriter(raw_path), **run_kwargs)
        if MODELS[model_key]["backend"] not in BATCH_BACKENDS or not run_kwargs.get("batch", True):
            log_run(raw_path, {"requests": len(todo), "minutes": round((time.time() - start) / 60, 1)})
    rows = {row["uuid"]: row for row in load_jsonl(raw_path)} if raw_path.exists() else {}  # empty while a batch runs
    return {key: rows[key] for key in prompts if key in rows}


def run_llm(model_key, records, setup, k, examples, raw_dir, split_tag, **run_kwargs):
    """Ask one model about every record (skipping those already in the raw file). Returns {uuid: raw answer}."""
    raw_path = Path(raw_dir) / f"{run_name(model_key, setup, k, split_tag)}.jsonl"
    prompts = {r["uuid"]: build_prompt(r, setup, examples) for r in records}
    return ask_all(model_key, prompts, raw_path, **run_kwargs)


# ----------------------------------------------------------------------------------------------------------------
# From raw answers to predictions
# ----------------------------------------------------------------------------------------------------------------

def _first_json_object(text):
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else ""


def parse_response(text):
    """(type, spans, ok) from a raw reply. ok is False when no JSON object could be read."""
    data = None
    for candidate in (text or "", _first_json_object(text or "")):
        try:
            data = json.loads(candidate)
            break
        except (json.JSONDecodeError, TypeError):
            continue
    if not isinstance(data, dict):
        return None, [], False
    spans = data.get("spoilers", data.get("spoiler", []))
    if isinstance(spans, str):
        spans = [spans]
    spans = [clean_span(str(s)) for s in spans if str(s).strip()]
    type_ = data.get("type") if data.get("type") in TYPES else guess_type(spans)
    return type_, spans, True


SNAP_MIN_SCORE = 80  # rapidfuzz partial_ratio (0-100) needed to replace a spoiler by its closest article span


def snap_span(span, context, min_score=SNAP_MIN_SCORE):
    """(text, how): the article's own text for a predicted spoiler.

    - "exact": the spoiler occurs in the article (ignoring case and spacing); its article text is used.
    - "fuzzy": no exact occurrence, but an article span is at least `min_score` similar; that span, widened to
      whole words, is used instead (fixes small slips such as changed punctuation or a dropped word).
    - "kept": nothing similar enough (a paraphrase or an invented answer); the spoiler stays as written.
    """
    if not span:
        return span, "kept"
    match = re.search(r"\s+".join(re.escape(w) for w in span.split()), context, flags=re.IGNORECASE)
    if match:
        return match.group(0), "exact"
    if len(span) >= len(context):
        return span, "kept"
    low_span, low_context = span.lower(), context.lower()
    if len(low_span) != len(span) or len(low_context) != len(context):  # rare Unicode case changes shift offsets
        low_span, low_context = span, context
    alignment = fuzz.partial_ratio_alignment(low_span, low_context, score_cutoff=min_score)
    if alignment is None:
        return span, "kept"
    start, end = alignment.dest_start, alignment.dest_end
    while start > 0 and context[start - 1].isalnum():
        start -= 1
    while end < len(context) and context[end].isalnum():
        end += 1
    return context[start:end].strip(), "fuzzy"


def to_llm_prediction(record, raw, snap=True):
    """A prediction in the SemEval run format, plus how its spans were snapped (for the analysis)."""
    type_, spans, ok = parse_response(raw.get("text"))
    how = []
    if snap and spans:
        context = cut_article(build_context(record))
        snapped = [snap_span(s, context) for s in spans]
        spans, how = [s for s, _ in snapped], [h for _, h in snapped]
    pred = to_prediction(record["uuid"], type_ or "phrase", spans)
    return pred, {"parsed": ok, "snap": how}


# ----------------------------------------------------------------------------------------------------------------
# Samples and costs
# ----------------------------------------------------------------------------------------------------------------

def stratified_sample(records, n, seed=0):
    """n records with the same type mix as `records`, drawn with a fixed seed (kept in their original order)."""
    by_type = {t: sorted((r for r in records if spoiler_type(r) == t), key=lambda r: r["uuid"]) for t in TYPES}
    quotas = {t: round(n * len(rs) / len(records)) for t, rs in by_type.items()}
    quotas["phrase"] += n - sum(quotas.values())
    rng = random.Random(seed)
    chosen = set()
    for t, rs in by_type.items():
        rs = rs[:]
        rng.shuffle(rs)
        chosen.update(r["uuid"] for r in rs[:quotas[t]])
    return [r for r in records if r["uuid"] in chosen]


def run_cost(rows, model_id, batch=None):
    """US$ for a set of raw answers at list prices (None if the model has no price entry).

    batch=None prices every answer by how it was made: half price if it came from a batch (the raw row has a
    batch_id), standard price if it was asked directly. True / False force one price for all rows.
    """
    p = PRICES.get(model_id)
    if p is None:
        return None
    total = 0.0
    for row in rows:
        u = row.get("usage") or {}
        cost = (u.get("input_tokens", 0) * p["input"] + u.get("output_tokens", 0) * p["output"]
                + u.get("cache_read_input_tokens", 0) * p.get("cache_read", p["input"])
                + u.get("cache_creation_input_tokens", 0) * p.get("cache_write", p["input"])) / 1e6
        from_batch = row.get("batch_id") is not None if batch is None else batch
        total += cost * (0.5 if from_batch else 1.0)
    return total
