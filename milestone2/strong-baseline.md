# Strong baseline: type classifier + extractive QA

Our strong baseline is the approach of Hagen et al. (2022) and the SemEval-2023 Task 5 organizer baseline (Fröbe et al., 2023): treat clickbait spoiling as **extractive question answering**, with the post as the question and the article as the passage, plus a separate **spoiler type classifier**. It's written as our own code on top of Hugging Face `transformers`, using our data pipeline, metrics and decoding. All the code is in `strong-baseline.ipynb` plus the small `spoilerqa/` package it imports.

## How it works

**1. Spoiler type classifier.** RoBERTa-base (Liu et al., 2019) with a classification head reads `<s> post </s></s> title + article </s>`, cut at 512 tokens, and predicts *phrase*, *passage* or *multi*. Multi spoilers are only 17% of train and the metric is balanced accuracy, so the cross-entropy loss weights each class by inverse frequency, $w_c = \frac{N}{3 \cdot N_c}$.

**2. Extractive QA.** We fine-tune `deepset/roberta-base-squad2`, a RoBERTa-base already trained on SQuAD 2.0 (Rajpurkar et al., 2018).
- **Input:** the post (question) followed by the context (the article title and its paragraphs, one per line). The title comes first so the 2% of spoilers that sit in the title can be extracted.
- **Windows:** articles are longer than one input (median 485 tokens, and 64% exceed 384), so each context is split into **windows of 384 tokens that overlap by 128**.
- **Gold spans:** the dataset gives spoiler positions per paragraph, and we convert them to character offsets in the context. The 1.6% of spans whose position is wrong are located by searching for the gold text instead. The 14 training spans that can't be found at all are skipped.
- **Labels:** a multi spoiler becomes one training example per span. A window containing the answer is labeled with its start and end token; any other window is labeled with the first token (the standard sliding-window convention).
- **Model:** the model predicts a start and an end score for every token. A span is scored as start score + end score, compared across all windows of the article.

**3. Decoding.** For each post we collect candidate spans of up to 128 tokens and decode by the predicted type:
- *phrase* and *passage*: the best-scoring span no longer than a per-type length cap;
- *multi*: the $k$ best non-overlapping spans, output in reading order.

The caps and $k$ are tuned on dev (grid below), separately for each predicted type, using the classifier's predicted types as at test time.

## Hyperparameters

| | Type classifier | Extractive QA |
|---|---|---|
| Starting model | `roberta-base` | `deepset/roberta-base-squad2` |
| Input | post + title + article, ≤ 512 tokens | post + context windows of 384 tokens, overlap 128 |
| Learning rate | 2e-5 | **1e-5 / 3e-5 / 5e-5** (sweep) |
| Batch size, epochs | 16, 5 | 16, 3 |
| Optimizer | AdamW, weight decay 0.01, linear schedule, 10% warmup | same |
| Precision | bf16 on the GPU | same |
| Best epoch chosen by | dev balanced accuracy | dev BLEU (decoded with gold types) |
| Seed | 42 | 42 |

**Decoding grid** (dev): max answer tokens for phrase ∈ {8, 16, 24, 32}, passage ∈ {48, 64, 96, 128}, multi spans ∈ {24, 48, 64}; spans per multi spoiler $k \in \{1, \dots, 6\}$.

## How to run

On the Google Cloud L4 VM (our run took 36 minutes: 4 for the classifier, 3 × 9 for the QA sweep, 4 for prediction and scoring):

```bash
cd milestone2
pip install -r requirements.txt
sudo apt-get install -y default-jre && bash setup_meteor.sh     # official METEOR (optional, but needed for the official numbers)
tmux new -s m2                                                  # so the run survives a dropped SSH session
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 strong-baseline.ipynb
```

You can also open `strong-baseline.ipynb` in JupyterLab and run all cells. The notebook reads the splits from `../data/splits`; set `DATA_DIR` to change that.

**Smoke test** (about 2 minutes on a laptop CPU, everything written under `smoke/`):
```bash
SMOKE=1 jupyter nbconvert --to notebook --execute strong-baseline.ipynb --output smoke-run.ipynb
```

**Outputs:**

| File | Contents |
|---|---|
| `runs/type-classifier/best/`, `runs/qa-roberta-base-lr*/best/` | trained models (not committed; about 0.5 GB each) |
| `output/type-classifier-{dev,test}.jsonl` | predicted types with class probabilities |
| `output/qa-sweep.json` | dev BLEU per epoch for each learning rate (the parameter sweep) |
| `output/strong-baseline-settings.json` | chosen learning rate, answer-length caps and $k$ |
| `output/strong-baseline-{dev,test}.jsonl` | **the strong baseline's predictions** (SemEval run format) |
| `output/strong-baseline-goldtypes-test.jsonl` | test predictions decoded with gold types (for analysis only) |
| `output/scores/*.json` | `score.py` results for each of the above |

To rescore any prediction file: `python score.py --gold ../data/splits/test.jsonl --pred output/strong-baseline-test.jsonl`.

## Results

From one run on the L4 (seed 42). The numbers come from `output/qa-sweep.json`, `output/strong-baseline-settings.json` and `output/scores/`; the full training log is in the executed `strong-baseline.ipynb`.

**Test set (1,000 posts):**

| System | BLEU (all) | Phrase | Passage | Multi | METEOR | BERTScore | Type bal. acc. |
|---|---|---|---|---|---|---|---|
| Simple baseline (first sentence) | 0.0497 | 0.0323 | 0.0816 | 0.0182 | 0.1281 | 0.8478 | 0.3333 |
| **Strong baseline** | **0.3843** | **0.5704** | **0.2219** | **0.3080** | **0.4182** | **0.9140** | **0.6920** |
| Strong baseline with gold types (analysis) | 0.4091 | 0.6188 | 0.2301 | 0.3138 | 0.4586 | 0.9176 | — |
| *Reference:* SemEval organizer baseline (DeBERTa) | 0.40 | 0.65 | 0.24 | 0.12 | | | |
| *Reference:* best SemEval system | 0.48 | 0.66 | 0.32 | 0.44 | | | 0.74 (best type classifier) |

The reference rows are from the SemEval-2023 overview on the same test set. The organizer baseline uses a larger reader (DeBERTa), so our RoBERTa-base baseline was expected to land somewhat below it.

**Strong baseline in detail** (`output/scores/strong-baseline-test.json`; dev in brackets, from `strong-baseline-dev.json`):

| Spoiler | All | Phrase | Passage | Multi |
|---|---|---|---|---|
| **BLEU** | **0.3843** (0.4041) | 0.5704 (0.5669) | 0.2219 (0.2683) | 0.3080 (0.3272) |
| METEOR | 0.4182 (0.4366) | 0.3240 (0.2950) | 0.3820 (0.4066) | 0.5317 (0.5425) |
| BERTScore | 0.9140 (0.9150) | 0.9402 (0.9403) | 0.8915 (0.8942) | 0.9022 (0.9024) |
| Exact match | 0.2700 (0.2884) | 0.5154 (0.5105) | 0.0918 (0.1352) | 0.0862 (0.1119) |
| Token F1 | 0.5127 (0.5133) | 0.6624 (0.6367) | 0.3629 (0.3887) | 0.4957 (0.5028) |

**Learning-rate sweep** (the parameter sweep; dev BLEU decoded with gold types after each epoch, from `output/qa-sweep.json`):

| Learning rate | Epoch 1 | Epoch 2 | Epoch 3 | Best |
|---|---|---|---|---|
| **1e-5** (chosen) | 0.4123 | **0.4434** | 0.4351 | 0.4434 (epoch 2) |
| 3e-5 | 0.4182 | 0.4289 | 0.4432 | 0.4432 (epoch 3) |
| 5e-5 | 0.4186 | 0.4272 | 0.4214 | 0.4272 (epoch 2) |

1e-5 and 3e-5 are tied (a 0.0002 difference is far smaller than the run-to-run noise one would expect); the notebook takes the higher one. 5e-5 is about 0.016 worse. Each run takes about 9 minutes.

**Decoding settings chosen on dev** (`output/strong-baseline-settings.json`): max answer tokens phrase 24, passage 96, multi 24 per span; $k = 5$ spans for multi. Neighbouring grid values score almost the same on dev (phrase 24 vs 32: 0.5811 vs 0.5811; passage 96 vs 48: 0.3130 vs 0.3130), so the decoder isn't sensitive to these caps. The best $k$ matches the corpus's apparent 5-span limit for multi spoilers.

**Type classifier.** Dev balanced accuracy by epoch: 0.515, 0.656, **0.737**, 0.714, 0.732, so epoch 3 is kept. On test it reaches **0.692** balanced accuracy:

| Gold ↓ / predicted → | phrase | passage | multi | Recall |
|---|---|---|---|---|
| phrase (423) | **296** | 81 | 46 | 0.700 |
| passage (403) | 74 | **286** | 43 | 0.710 |
| multi (174) | 30 | 28 | **116** | 0.667 |
| Precision | 0.740 | 0.724 | 0.566 | |

### What the numbers say

- **The strong baseline beats the simple baseline by a wide margin:** 0.384 vs 0.050 test BLEU. It lands slightly below the organizer DeBERTa baseline overall (0.384 vs 0.40), as expected for a smaller reader, and is below it on phrase (0.57 vs 0.65) and passage (0.22 vs 0.24). On **multi it is much higher (0.31 vs 0.12)**, probably because our decoder returns 5 separate non-overlapping spans for multi posts.
- **Wrong types cost about 0.025 BLEU.** With gold types the same QA model scores 0.409 instead of 0.384. 302 of the 1,000 test posts get the wrong type, and on those posts BLEU is 0.195 with the predicted type vs 0.277 with the gold type. The most expensive mistake is a phrase spoiler predicted as multi (46 posts, mean BLEU 0.07): the decoder outputs five spans where one word was wanted.
- **Passage spoilers are the hard part:** 0.22–0.23 BLEU even with gold types, and 22% of passage posts score 0 BLEU (vs 16% for phrase, 8% for multi). That's where a stronger reader has the most room to help in Milestone 3.
- **Read per-type METEOR with care.** The official METEOR is a corpus-level score: it pools matches and lengths over all posts before computing precision and recall. So a few long wrong outputs drag the whole group down. Phrase METEOR is 0.32 with predicted types, but 0.67 on the 296 phrase posts whose type was right, because the 127 mis-typed phrase posts produce 73% of all predicted words in that group. BLEU, which is averaged per post, doesn't have this problem. That's one more reason to compare systems on BLEU.
- **Dev is a little easier than test** (0.404 vs 0.384 BLEU; 0.737 vs 0.692 balanced accuracy), so the dev-tuned settings carry over reasonably but not perfectly.

## References

- Hagen, M., Fröbe, M., Jurk, A., Potthast, M. (2022). *Clickbait Spoiling via Question Answering and Passage Retrieval.* ACL. https://aclanthology.org/2022.acl-long.484/
- Fröbe, M., et al. (2023). *SemEval-2023 Task 5: Clickbait Spoiling.* SemEval. https://aclanthology.org/2023.semeval-1.312/
- Liu, Y., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* https://arxiv.org/abs/1907.11692
- Rajpurkar, P., Jia, R., Liang, P. (2018). *Know What You Don't Know: Unanswerable Questions for SQuAD.* ACL. https://aclanthology.org/P18-2124/
- deepset (2020). `roberta-base-squad2`. https://huggingface.co/deepset/roberta-base-squad2
