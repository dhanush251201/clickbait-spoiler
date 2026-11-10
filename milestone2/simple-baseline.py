"""Simple baseline for clickbait spoiling: no learning beyond counting labels.

For every post it predicts
  - spoiler type: the most common type in the training data (majority class), and
  - spoiler: the first sentence of the article (or, with --strategy title, the article title, which is the
    naive baseline the SemEval-2023 organizers published).

Usage:
    python simple-baseline.py --train ../data/splits/train.jsonl --input ../data/splits/test.jsonl \
        --output output/simple-baseline-test.jsonl
"""
import argparse
from collections import Counter

import nltk

from spoilerqa.data import load_jsonl, spoiler_type, write_jsonl
from spoilerqa.metrics import ensure_nltk_data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, help="training split, used only to find the majority type")
    parser.add_argument("--input", required=True, help="split to predict, e.g. ../data/splits/test.jsonl")
    parser.add_argument("--output", required=True, help="where to write the predictions (.jsonl)")
    parser.add_argument("--strategy", choices=["first-sentence", "title"], default="first-sentence",
                        help="what to use as the spoiler (default: first sentence of the article)")
    return parser.parse_args()


def majority_type(train):
    # The type that occurs most often in train; on our data that is "phrase" (about 43% of posts).
    return Counter(spoiler_type(r) for r in train).most_common(1)[0][0]


def first_sentence(record):
    # The first sentence of the first non-empty paragraph, split with NLTK's Punkt sentence tokenizer.
    # Falls back to the title for the rare article without any text.
    for paragraph in record["targetParagraphs"]:
        if paragraph.strip():
            return nltk.sent_tokenize(paragraph.strip())[0]
    return record["targetTitle"]


def main():
    args = parse_args()
    ensure_nltk_data()
    train = load_jsonl(args.train)
    records = load_jsonl(args.input)

    label = majority_type(train)
    predictions = []
    for r in records:
        spoiler = first_sentence(r) if args.strategy == "first-sentence" else r["targetTitle"]
        predictions.append({"uuid": r["uuid"], "spoilerType": label, "spoiler": spoiler})

    write_jsonl(predictions, args.output)
    print(f"majority type in train: {label}; wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
