# Milestone 2: evaluation script, simple baseline, strong baseline

Everything for Milestone 2 of the SpoilerQA project (CIS 5300, due Nov 18, 2026). The data lives one level up in `../data/splits/` (built in Milestone 1); nothing here modifies it.

| Course deliverable | File(s) |
|---|---|
| Evaluation script | `score.py` |
| Metric definition + how to run it | `scoring.md` |
| Simple baseline | `simple-baseline.py`, `simple-baseline.md` |
| Strong baseline | `strong-baseline.ipynb` (+ `spoilerqa/`), `strong-baseline.md` |
| 1–2 page PDF report | written by the team |

## Folder layout

```
milestone2/
├── score.py                 evaluation script (official SemEval-2023 Task 5 metrics)
├── simple-baseline.py       majority type + first sentence of the article
├── strong-baseline.ipynb    type classifier + extractive QA (3 learning rates), prediction, scoring
├── spoilerqa/               shared helpers: data.py (loading, span offsets), qa.py (windows, decoding),
│                            metrics.py (BLEU / METEOR / BERTScore / balanced accuracy)
├── setup_meteor.sh          downloads the official METEOR 1.5 jar + paraphrase table into tools/
├── requirements.txt
├── output/                  predictions (.jsonl) and scores (output/scores/*.json)
└── runs/                    model checkpoints from the notebook (not committed)
```

## Quick start

```bash
cd milestone2
pip install -r requirements.txt
bash setup_meteor.sh                      # optional; needs Java. Without it METEOR falls back to NLTK.

# simple baseline (seconds, any machine)
python simple-baseline.py --train ../data/splits/train.jsonl --input ../data/splits/test.jsonl \
    --output output/simple-baseline-test.jsonl
python score.py --gold ../data/splits/test.jsonl --pred output/simple-baseline-test.jsonl

# strong baseline (about 40 minutes on the L4 GPU)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 strong-baseline.ipynb
```

## Status

All Milestone 2 code and markdown deliverables are done (the PDF report is written by the team).

- `score.py`: BLEU matches the official evaluator exactly, and gold scored against gold gives 1.0.
- **Simple baseline:** test BLEU 0.0497.
- **Strong baseline** (run on the L4, Nov 6): test BLEU **0.3843** (phrase 0.570 / passage 0.222 / multi 0.308), METEOR 0.418, BERTScore 0.914, type balanced accuracy 0.692. With gold types: 0.4091. The learning-rate sweep and the analysis are in `strong-baseline.md`. `strong-baseline.ipynb` is saved with the outputs of that run.

The code was tested with transformers 5.17 on Python 3.12, with PyTorch 2.14 on macOS and PyTorch 2.9.1 (CUDA) on the VM.
