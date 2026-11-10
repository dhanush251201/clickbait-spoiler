# Evaluation

Our project is based on **SemEval-2023 Task 5: Clickbait Spoiling** (Fröbe et al., 2023), so we reuse its official metrics. A system gets a clickbait post and the linked article, and must produce

1. the **spoiler type** (*phrase*, *passage* or *multi*), scored with **balanced accuracy**, and
2. the **spoiler** text, scored with **BLEU** (the main metric), **METEOR** and **BERTScore**, overall and separately for each gold spoiler type.

`score.py` reimplements the official evaluator (`semeval23/clickbait-spoiling-eval.py` in [pan-webis-de/pan-code](https://github.com/pan-webis-de/pan-code/tree/master/semeval23)), so our numbers are comparable to the shared-task leaderboard. We checked this for BLEU: on 1,000 test items (identical, partially overlapping, unrelated and empty predictions) our per-item scores match the official function exactly. We also report two SQuAD-style extras, **exact match** and **token F1**, which are easier to interpret.

## Running it

```bash
python score.py --gold ../data/splits/test.jsonl --pred output/simple-baseline-test.jsonl \
    --output output/scores/simple-baseline-test.json
```

| Argument | Meaning |
|---|---|
| `--gold` | a gold split in the corpus format (`../data/splits/{train,dev,test}.jsonl`) |
| `--pred` | system output, JSON Lines: `{"uuid": ..., "spoiler": "text" or ["span", ...], "spoilerType": "phrase"}`. `spoilerType` is optional: without it, type scores are skipped. This is the SemEval run format. |
| `--output` | optional: also write every score to a JSON file |
| `--no-bertscore` | skip BERTScore (it loads `roberta-large`, which takes a minute on a CPU) |
| `--no-meteor` | skip METEOR |

METEOR uses the official METEOR 1.5 jar and paraphrase table, which `bash setup_meteor.sh` downloads and which need Java. Without them, `score.py` falls back to NLTK's METEOR and labels it as an approximation.

Example output (the simple baseline on the test set):

```
Scored 1000 posts (0 missing predictions)

spoiler              all    phrase   passage     multi
n                   1000       423       403       174
bleu              0.0497    0.0323    0.0816    0.0182
meteor            0.1281    0.0535    0.2013    0.1005
bertscore         0.8478    0.8395    0.8622    0.8346
exact-match       0.0200    0.0000    0.0471    0.0057
token-f1          0.1174    0.0692    0.1761    0.0988

spoiler type: balanced accuracy 0.3333
  phrase   precision 0.4230  recall 1.0000  f1 0.5945
  passage  precision 0.0000  recall 0.0000  f1 0.0000
  multi    precision 0.0000  recall 0.0000  f1 0.0000

BLEU (all): 0.0497
```

Columns are subsets of the gold data by spoiler type, so `phrase` means "posts whose gold spoiler is a phrase". The last line is the single headline number.

## Preparing the text

As in the official evaluator, a multi spoiler (a list of spans) is joined into one string with spaces, newlines become spaces, and the result is stripped. This is done for both gold and predicted spoilers. A post with no prediction is scored as an empty string.

## BLEU (main metric)

BLEU (Papineni et al., 2002) measures n-gram overlap between the predicted spoiler $\hat{y}$ and the gold spoiler $y$. The shared task computes it **per post and then averages**, with three changes from textbook BLEU that come from Hagen et al. (2022):

1. **Tokenization and filtering.** Both texts are tokenized with NLTK's `word_tokenize`. English stopwords (NLTK's list) are removed, then tokens are lowercased and punctuation tokens are removed. Call the resulting token lists $t$ (gold) and $p$ (prediction).
2. **Shorter n-grams for short spoilers.** Many spoilers are one or two words, so the maximum n-gram order $N$ depends on the lengths:
$$
N = \begin{cases} 4 & |t| > 3 \text{ and } |p| > 3 \\ 3 & |t| > 2 \text{ and } |p| > 2 \\ 2 & |t| > 1 \text{ and } |p| > 1 \\ 1 & \text{otherwise} \end{cases}
$$
3. **No smoothing.** If any n-gram order up to $N$ has no match, the post scores 0.

For one post, with $\text{count}_p(g)$ the number of times n-gram $g$ occurs in $p$:
$$
p_n = \frac{\sum_{g \in \text{ngrams}_n(p)} \min\big(\text{count}_p(g), \text{count}_t(g)\big)}{\sum_{g \in \text{ngrams}_n(p)} \text{count}_p(g)}, \qquad
\text{BP} = \begin{cases} 1 & |p| > |t| \\ e^{\,1 - |t|/|p|} & |p| \le |t| \end{cases}
$$
$$
\text{BLEU}(t, p) = \text{BP} \cdot \exp\Big(\sum_{n=1}^{N} \tfrac{1}{N} \log p_n\Big), \qquad
\text{BLEU} = \frac{1}{M} \sum_{i=1}^{M} \text{BLEU}(t_i, p_i)
$$
over the $M$ posts being scored.

## METEOR

METEOR (Banerjee and Lavie, 2005), version 1.5 (Denkowski and Lavie, 2014), is run exactly as in the official evaluator: `java -jar meteor-1.5.jar truths.txt preds.txt -l en -norm -t adq`. It aligns prediction and gold words in four stages: exact match, same stem, WordNet synonym, and paraphrase table, with weights $w = (1.0, 1.0, 0.6, 0.8)$. From the aligned words it computes a weighted precision $P$ and recall $R$, in which content words count with weight $\delta$ and function words with $1-\delta$. These combine into
$$
F_{\text{mean}} = \frac{P \cdot R}{\alpha P + (1-\alpha) R}, \qquad
\text{Pen} = \gamma \left(\frac{ch}{m}\right)^{\beta}, \qquad
\text{METEOR} = (1 - \text{Pen}) \cdot F_{\text{mean}}
$$
where $m$ is the number of matched words and $ch$ the number of contiguous matched chunks. The fragmentation penalty rewards predictions that match the gold text in long contiguous runs. The `adq` parameters are $\alpha = 0.75$, $\beta = 1.4$, $\gamma = 0.45$, $\delta = 0.7$. The reported number is METEOR's final system-level score over all posts in the subset.

We pass the paraphrase table with `-a`. It's the same file the jar loads by default, but the jar can't find its default location when the project path contains a space, as ours does.

## BERTScore

BERTScore (Zhang et al., 2020) compares contextual embeddings instead of surface words, so paraphrases still get credit. With the default English model (`roberta-large`, layer 17, no idf weighting, no baseline rescaling, as in the official evaluator), each gold token $y_i$ and predicted token $\hat{y}_j$ get embeddings $\mathbf{y}_i$ and $\hat{\mathbf{y}}_j$, and
$$
R = \frac{1}{|y|} \sum_{y_i \in y} \max_{\hat{y}_j \in \hat{y}} \cos(\mathbf{y}_i, \hat{\mathbf{y}}_j), \qquad
P = \frac{1}{|\hat{y}|} \sum_{\hat{y}_j \in \hat{y}} \max_{y_i \in y} \cos(\mathbf{y}_i, \hat{\mathbf{y}}_j), \qquad
F = \frac{2PR}{P + R}
$$
We report the mean $F$ over posts; an empty prediction scores 0. BERTScore barely separates systems on this task: the top shared-task systems all score 0.92–0.93. So we use it as a secondary number.

## Spoiler type: balanced accuracy

Spoiler types are imbalanced (42% phrase, 40% passage, 18% multi), so the shared task uses balanced accuracy (Brodersen et al., 2010), the mean recall over the three classes:
$$
\text{BalancedAccuracy} = \frac{1}{3} \sum_{c \in \{\text{phrase}, \text{passage}, \text{multi}\}} \frac{|\{i : y_i = c \wedge \hat{y}_i = c\}|}{|\{i : y_i = c\}|}
$$
A constant prediction always scores $1/3$, whatever the class. `score.py` also prints per-type precision, recall and F1, computed as in the official evaluator: one type against the rest, over the posts where the gold or predicted type is that type.

## Extras: exact match and token F1

Following SQuAD (Rajpurkar et al., 2016), both texts are lowercased, and punctuation, the articles *a/an/the* and extra whitespace are removed.
- **Exact match** is 1 if the two normalized strings are identical.
- **Token F1** is the harmonic mean of precision and recall of the shared tokens (counted with multiplicity).

Both are averaged over posts. They aren't part of the shared task, but they're easier to read than BLEU and help with error analysis.

## How the metrics were used before

Hagen et al. (2022) introduced the corpus and evaluated spoiler generation with BLEU-4, METEOR and BERTScore. The SemEval-2023 Task 5 overview (Fröbe et al., 2023) used the same three for spoiler generation and balanced accuracy for spoiler type classification. The best system reached 0.48 BLEU on the test set we use, and the organizers' DeBERTa baseline 0.40.

## References

- Papineni, K., Roukos, S., Ward, T., Zhu, W.-J. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation.* ACL. https://aclanthology.org/P02-1040/
- Banerjee, S., Lavie, A. (2005). *METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments.* ACL Workshop on Intrinsic and Extrinsic Evaluation Measures. https://aclanthology.org/W05-0909/
- Denkowski, M., Lavie, A. (2014). *Meteor Universal: Language Specific Translation Evaluation for Any Target Language.* WMT. https://aclanthology.org/W14-3348/
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., Artzi, Y. (2020). *BERTScore: Evaluating Text Generation with BERT.* ICLR. https://arxiv.org/abs/1904.09675
- Brodersen, K. H., Ong, C. S., Stephan, K. E., Buhmann, J. M. (2010). *The Balanced Accuracy and Its Posterior Distribution.* ICPR.
- Rajpurkar, P., Zhang, J., Lopyrev, K., Liang, P. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text.* EMNLP. https://aclanthology.org/D16-1264/
- Hagen, M., Fröbe, M., Jurk, A., Potthast, M. (2022). *Clickbait Spoiling via Question Answering and Passage Retrieval.* ACL. https://aclanthology.org/2022.acl-long.484/
- Fröbe, M., et al. (2023). *SemEval-2023 Task 5: Clickbait Spoiling.* SemEval. https://aclanthology.org/2023.semeval-1.312/
