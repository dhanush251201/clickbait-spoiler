"""Score clickbait spoiling predictions against the gold standard.

Usage:
    python score.py --gold ../data/splits/test.jsonl --pred output/simple-baseline-test.jsonl

Predictions are JSON Lines with one object per post:
    {"uuid": "...", "spoiler": "text" or ["span 1", "span 2"], "spoilerType": "phrase"}
`spoilerType` is optional; when present we also score spoiler type classification. This is the run format
of SemEval-2023 Task 5, so the same file can be fed to the official evaluator.

The main number is BLEU (official definition, averaged over posts). See scoring.md for every metric.
"""
import argparse
import json
from pathlib import Path

from spoilerqa.data import TYPES, load_jsonl, spoiler_type
from spoilerqa.metrics import (as_text, bertscore_items, bleu, ensure_nltk_data, exact_match, meteor, token_f1,
                               type_scores)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", required=True, help="gold split, e.g. ../data/splits/test.jsonl")
    parser.add_argument("--pred", required=True, help="predictions (.jsonl) with uuid, spoiler and optionally spoilerType")
    parser.add_argument("--output", help="also write all scores to this JSON file")
    parser.add_argument("--no-bertscore", action="store_true", help="skip BERTScore (it needs roberta-large and is slow on CPU)")
    parser.add_argument("--no-meteor", action="store_true", help="skip METEOR")
    return parser.parse_args()


def spoiler_scores(truths, preds, bert_f1=None, with_meteor=True):
    scores = {
        "bleu": bleu(truths, preds),
        "exact-match": sum(map(exact_match, truths, preds)) / len(truths),
        "token-f1": sum(map(token_f1, truths, preds)) / len(truths),
    }
    if with_meteor:
        scores["meteor"], scores["meteor-implementation"] = meteor(truths, preds)
    if bert_f1 is not None:
        scores["bertscore"] = sum(bert_f1) / len(bert_f1)
    return scores


def main():
    args = parse_args()
    ensure_nltk_data()
    gold = load_jsonl(args.gold)
    predictions = {}
    for p in load_jsonl(args.pred):
        if p["uuid"] in predictions:
            raise SystemExit(f"duplicate prediction for uuid {p['uuid']}")
        predictions[p["uuid"]] = p

    # Missing predictions count as empty spoilers, as in the official evaluator.
    missing = [g["uuid"] for g in gold if g["uuid"] not in predictions]
    truths = [as_text(g["spoiler"]) for g in gold]
    preds = [as_text(predictions.get(g["uuid"], {}).get("spoiler", "")) for g in gold]
    gold_types = [spoiler_type(g) for g in gold]

    bert_f1 = None if args.no_bertscore else bertscore_items(truths, preds)
    results = {"n": len(gold), "missing-predictions": len(missing), "spoiler": {}}
    for subset in ["all"] + TYPES:
        idx = [i for i, t in enumerate(gold_types) if subset == "all" or t == subset]
        results["spoiler"][subset] = spoiler_scores(
            [truths[i] for i in idx], [preds[i] for i in idx],
            None if bert_f1 is None else [bert_f1[i] for i in idx], with_meteor=not args.no_meteor)
        results["spoiler"][subset]["n"] = len(idx)

    if all("spoilerType" in predictions.get(u, {}) for u in [g["uuid"] for g in gold]):
        pred_types = [predictions[g["uuid"]]["spoilerType"] for g in gold]
        pred_types = [t[0] if isinstance(t, list) else t for t in pred_types]
        results["type"] = type_scores(gold_types, pred_types)

    print_summary(results)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


def print_summary(results):
    columns = ["all"] + TYPES
    print(f"Scored {results['n']} posts ({results['missing-predictions']} missing predictions)\n")
    print(f"{'spoiler':<14}" + "".join(f"{c:>10}" for c in columns))
    print(f"{'n':<14}" + "".join(f"{results['spoiler'][c]['n']:>10}" for c in columns))
    for metric in ["bleu", "meteor", "bertscore", "exact-match", "token-f1"]:
        if metric in results["spoiler"]["all"]:
            print(f"{metric:<14}" + "".join(f"{results['spoiler'][c][metric]:>10.4f}" for c in columns))
    if results["spoiler"]["all"].get("meteor-implementation") == "nltk":
        print("(METEOR computed with NLTK as an approximation; run setup_meteor.sh for the official METEOR 1.5)")
    if "type" in results:
        t = results["type"]
        print(f"\nspoiler type: balanced accuracy {t['balanced-accuracy']:.4f}")
        for label in TYPES:
            print(f"  {label:<8} precision {t[f'precision-{label}']:.4f}  recall {t[f'recall-{label}']:.4f}"
                  f"  f1 {t[f'f1-{label}']:.4f}")
    print(f"\nBLEU (all): {results['spoiler']['all']['bleu']:.4f}")


if __name__ == "__main__":
    main()
