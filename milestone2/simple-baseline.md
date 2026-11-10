# Simple baseline

The simplest system that produces both outputs of the task, with no learning beyond counting labels:

- **Spoiler type:** always the most common type in the training data (majority class). On our training split that's *phrase* (1,348 of 3,157 posts, 42.7%).
- **Spoiler:** the **first sentence of the article**, i.e. the first sentence of the first non-empty paragraph, found with NLTK's Punkt sentence splitter. News articles often put the key fact first, so this is the natural "lead" baseline.

`simple-baseline.py` can also use the **article title** as the spoiler instead (`--strategy title`). That's the naive baseline published by the SemEval-2023 organizers, and we report it for comparison.

## Usage

```bash
python simple-baseline.py --train ../data/splits/train.jsonl --input ../data/splits/test.jsonl \
    --output output/simple-baseline-test.jsonl
python score.py --gold ../data/splits/test.jsonl --pred output/simple-baseline-test.jsonl
```

| Argument | Meaning |
|---|---|
| `--train` | training split; only used to find the majority type |
| `--input` | the split to predict (`dev.jsonl` or `test.jsonl`) |
| `--output` | where to write predictions (JSON Lines, SemEval run format) |
| `--strategy` | `first-sentence` (default) or `title` |

It runs in a couple of seconds on a laptop; the only dependency is NLTK.

## Sample output

The first three lines of `output/simple-baseline-test.jsonl`, next to the post and the gold spoiler:

```json
{"uuid": "2f3e30d7-972c-4812-b727-6b582de52137", "spoilerType": "phrase", "spoiler": "Erin Zaikis often used to think about how she could help change the world, but it wasn't until she got very sick with dengue fever that she decided to take action."}
{"uuid": "45425f74-4c0e-420c-aea8-6337ea91e107", "spoilerType": "phrase", "spoiler": "We weren't expecting this one!"}
{"uuid": "2fb84020-cfdb-4b7f-9cfc-b891c306c946", "spoilerType": "phrase", "spoiler": "This whole Taylor Swift, Tom Hiddleston thing just doesn't add up to us."}
```

| Post | Gold spoiler (type) | Baseline spoiler |
|---|---|---|
| This simple household item saves lives | soap (phrase) | Erin Zaikis often used to think about how she could help change the world, but it wasn't until she got very sick with dengue fever… |
| You'll Never Believe Who Robert Downey Jr.'s 'Free Pass' Is! | Gwyneth Paltrow (phrase) | We weren't expecting this one! |
| THE NEWEST THEORY ON HIDDLESWIFT'S ROMANCE IS BIZARRE BUT STRANGELY COMPELLING | filming the next Nicholas Sparks film (passage) | This whole Taylor Swift, Tom Hiddleston thing just doesn't add up to us. |

Clickbait articles usually open with a teaser rather than the answer, which is exactly why this baseline does badly.

## Results

Scored with `score.py` (see `scoring.md`). **Test set (1,000 posts):**

| Spoiler | All | Phrase | Passage | Multi |
|---|---|---|---|---|
| **BLEU** | **0.0497** | 0.0323 | 0.0816 | 0.0182 |
| METEOR | 0.1281 | 0.0535 | 0.2013 | 0.1005 |
| BERTScore | 0.8478 | 0.8395 | 0.8622 | 0.8346 |
| Exact match | 0.0200 | 0.0000 | 0.0471 | 0.0057 |
| Token F1 | 0.1174 | 0.0692 | 0.1761 | 0.0988 |

**Spoiler type:** balanced accuracy **0.3333**, as for any constant prediction.

For comparison:

| Variant / split | BLEU (all) | Phrase | Passage | Multi |
|---|---|---|---|---|
| First sentence, **test** | **0.0497** | 0.0323 | 0.0816 | 0.0182 |
| First sentence, dev | 0.0365 | 0.0263 | 0.0552 | 0.0187 |
| Title (organizers' naive baseline), test | 0.0236 | 0.0413 | 0.0143 | 0.0023 |

The first sentence beats the title overall because it helps on passage spoilers. The title is slightly better for phrase spoilers, which are sometimes named in the headline. Either way, the simple baselines stay near zero BLEU: the spoiler is rarely in the opening lines. In our data analysis, only 10–23% of first sentences contain a spoiler span, depending on type. BERTScore stays high (0.85) even for these weak predictions, which is why BLEU is the main metric.

Score files: `output/scores/simple-baseline-test.json`, `output/scores/simple-baseline-dev.json`, `output/scores/title-baseline-test.json`.
