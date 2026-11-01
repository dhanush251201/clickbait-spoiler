# Data: Webis Clickbait Spoiling Corpus 2022

Our project, SpoilerQA, takes a clickbait post and the article it links to. It predicts (1) the spoiler type (phrase, passage or multi) and (2) the spoiler itself, as one or more spans of text copied from the article. We use the **Webis Clickbait Spoiling Corpus 2022** (Webis-Clickbait-22), the dataset of SemEval-2023 Task 5. We did not collect any new data.

| | |
|---|---|
| **Source** | Hagen, Fröbe, Jurk and Potthast (2022), released on Zenodo: [doi.org/10.5281/zenodo.8136637](https://doi.org/10.5281/zenodo.8136637) |
| **License** | CC BY 4.0 (redistribution allowed with attribution) |
| **Size** | 4,951 posts after de-duplication (5,000 in the original release) |
| **Splits** | train 3,157 · dev 794 · test 1,000 |
| **Format** | JSON Lines (`.jsonl`), UTF-8, one post per line |
| **Our archive** | `spoilerqa_data.tar.gz` (≈11 MB), with the **full** dataset, not a sample |

---

## 1. Where the data comes from

The corpus was built by the Webis research group (Hagen et al., ACL 2022). Clickbait posts were collected from **Twitter, Reddit and Facebook** after they had already been "spoiled" by accounts and communities that do this publicly. The `spoilerPublisher` field names them: mostly *savedyouaclick* and *HuffPoSpoilers*, plus several *Stop Clickbait* Facebook pages. The linked web pages were archived, the main article text was **manually cleaned** and split into paragraphs, and annotators **extracted the spoiler from the article** and labeled its type. The same corpus was used for the SemEval-2023 Task 5 shared task (Fröbe et al., 2023), whose test set is our test set.

We downloaded the newest Zenodo version of the dataset (record **8136637**, uploaded July 2023, file `webis-clickbait-22.zip`, MD5 `51ae4e5ec98c2719a202901282fe8886`). The older and more commonly linked record (6362726) only contains train and validation. The newer one adds the labeled test set.

**Please cite:**
- Matthias Hagen, Maik Fröbe, Artur Jurk, Martin Potthast. 2022. *Clickbait Spoiling via Question Answering and Passage Retrieval.* ACL 2022. https://aclanthology.org/2022.acl-long.484/
- Maik Fröbe et al. 2023. *SemEval-2023 Task 5: Clickbait Spoiling.* SemEval 2023. https://aclanthology.org/2023.semeval-1.312/

---

## 2. Splits

We start from the official three-way split, since this is a shared task, and rename the files:

| Our split | Official file | Posts (original) | Removed | **Posts (ours)** | Share |
|---|---|---|---|---|---|
| train | `train.jsonl` | 3,200 | 43 | **3,157** | 63.8% |
| dev | `validation.jsonl` | 800 | 6 | **794** | 16.0% |
| test | `test.jsonl` | 1,000 | 0 | **1,000** | 20.2% |
| **total** | | 5,000 | 49 | **4,951** | |

- **test** is the SemEval-2023 test set, **unchanged** (byte-identical to the release). Keeping it intact means our scores can be compared with the shared-task leaderboard. We treat it as blind: no tuning or model selection on it.
- **dev** is the official validation set. We use it for hyperparameters and for comparing approaches.
- The split is 64/16/20 rather than the 80/10/10 suggested for self-made splits, because we keep the official test set.

### Removing items that appear in more than one split

The official splits contain a few **near-duplicate items across splits**: the same story crawled twice, often with a slightly reworded post. For example, *"The Jet Black iPhone 7 Has A Major Flaw"* is in train and *"The Jet Black iPhone 7 has a major flaw"* is in test, with the same article and the same spoiler. To make sure no item appears in more than one split:

1. **Candidates** are pairs of posts from different splits that share the same post text, title or article text (after lowercasing and removing punctuation).
2. A candidate counts as a **duplicate** if the two articles' vocabularies overlap by at least 50% (Jaccard similarity on word sets). This keeps recurring generic headlines apart. *"10 things you need to know before the opening bell"* appears in train and dev, but each copy links to a different day's article (about 20% overlap), so both stay.
3. The copy in the **earlier** split is dropped. Dev loses items that are in test, and train loses items that are in dev or test. Test is never touched.

This removed **43 train items** (22 duplicated in test, 21 in dev) and **6 dev items** (duplicated in test). Every removed item is listed in `removed_duplicates.tsv` with the item it duplicates and the overlap score. The procedure is in `scripts/make_splits.py` and reproduces these files exactly.

### Label and platform distribution per split

| Split | Posts | Phrase | Passage | Multi | Twitter | Reddit | Facebook | Spoiler spans |
|---|---|---|---|---|---|---|---|---|
| train | 3,157 | 1,348 (42.7%) | 1,259 (39.9%) | 550 (17.4%) | 1,521 | 1,125 | 511 | 4,547 |
| dev | 794 | 333 (41.9%) | 318 (40.1%) | 143 (18.0%) | 365 | 302 | 127 | 1,186 |
| test | 1,000 | 423 (42.3%) | 403 (40.3%) | 174 (17.4%) | 482 | 344 | 174 | 1,463 |

All three splits have the same type mix. Platforms differ in type mix: about 22% of Twitter posts are multi, against 11% on Facebook.

---

## 3. Files

```
spoilerqa_data/              <- contents of spoilerqa_data.tar.gz
├── train.jsonl              3,157 posts   18.9 MB
├── dev.jsonl                  794 posts    5.0 MB
├── test.jsonl               1,000 posts    5.8 MB
├── removed_duplicates.tsv      49 rows     items dropped from train/dev (section 2)
└── data.md                  this file
```

In the project repository the same files are in `data/splits/`. The untouched original download is in `data/raw/`. `scripts/make_splits.py` rebuilds `data/splits/` and the archive from `data/raw/`. `notebooks/01_eda.ipynb` contains the full exploratory analysis.

---

## 4. File format

Each line of a `.jsonl` file is one JSON object, which is one clickbait post with its article and gold spoiler. All three splits have the same fields, and test includes the gold labels.

| Field | Type | Description | Missing (of 4,951) |
|---|---|---|---|
| `uuid` | string | Unique id of the post. We key predictions on it. | 0 |
| `postId` | string | Id of the post on its platform. | 0 |
| `postText` | list of 1 string | The clickbait post. Always exactly one element. | 0 |
| `postPlatform` | string | `Twitter`, `reddit` or `Facebook` (note the inconsistent casing). | 0 |
| `targetTitle` | string | Title of the linked article. | 3 empty |
| `targetParagraphs` | list of strings | The manually cleaned article body, one string per paragraph (1–265 paragraphs, median 10). | 0 |
| `targetDescription` | string | The page's meta description. | 510 |
| `targetKeywords` | string | The page's meta keywords, comma-separated. | 2,032 |
| `targetMedia` | list of strings | URLs of images on the page. | 793 empty |
| `targetUrl` | string | URL of the linked page. About a third are shortened links (mostly `huff.to`) and a fifth are web-archive links (`archive.is`, `web.archive.org`). | 740 |
| `provenance` | object | `spoilerPublisher` (the account that published the original spoiler), `humanSpoiler` (the spoiler text that account wrote, free text) and `source` (always `"anonymized"`). | `humanSpoiler` missing for 789 |
| `spoiler` | list of strings | **Gold spoiler**: the span(s) copied from the article. One span for phrase and passage, usually 2–5 for multi. | 0 |
| `spoilerPositions` | list | Where each spoiler span is in the article. Same length and order as `spoiler` (see below). | 0 |
| `tags` | list of 1 string | **Gold spoiler type**: `phrase`, `passage` or `multi`. | 0 |

### How `spoilerPositions` works

Each span has a position `[[start_paragraph, start_char], [end_paragraph, end_char]]`:
- character offsets are **relative to the paragraph**, not to the whole article;
- `end_char` is **exclusive**, so the span is `paragraph[start_char:end_char]`;
- paragraph index **`-1` means the title** (`targetTitle`). This happens for 108 spans in 107 posts.

Example: the position `[[0, 0], [0, 4]]` means characters 0–4 of paragraph 0, which in the record below is `"2070"`. A multi spoiler simply has one position per span:

```json
"postText": ["Passion is overrated — 7 work habits you need instead"],
"spoiler": ["Purpose connects us to something bigger and in doing so makes us right sized",
            "be ruthless with your \"No’s.\"",
            "Practice means greatness is doable ... one tiny step after another",
            "planning of the SMART goal and number-crunching variety",
            "Objectivity — the ability to see the world as it truly is"],
"spoilerPositions": [[[11, 25], [11, 101]], [[17, 56], [17, 85]], [[23, 240], [23, 306]],
                     [[28, 65], [28, 120]], [[37, 106], [37, 163]]],
"tags": ["multi"]
```

---

## 5. Example record

One complete training record (a phrase spoiler). The article is shortened to 3 of its 10 paragraphs and the image list to 2 of 10 URLs; nothing else is changed.

```json
{
  "uuid": "b1a1f63d-8853-4a11-89e8-6b2952a393ec",
  "postId": "411701128456593408",
  "postText": ["NASA sets date for full recovery of ozone hole"],
  "postPlatform": "Twitter",
  "targetParagraphs": [
    "2070 is shaping up to be a great year for Mother Earth.",
    "That's when NASA scientists are predicting the hole in the ozone layer might finally make a full recovery. Researchers announced their conclusion, in addition to other findings, in a presentation Wednesday during the annual American Geophysical Union meeting in San Francisco.",
    "...",
    "By 2070, however, the ozone hole is expected to have made a full recovery."
  ],
  "targetTitle": "Hole In Ozone Layer Expected To Make Full Recovery By 2070: NASA",
  "targetDescription": "2070 is shaping up to be a great year for Mother Earth.\r\n\r\nThat's when NASA scientists are predicting the",
  "targetKeywords": "ozone layer,ozone hole determined by weather,Montreal Protocol,Climate change,greenhouse gas emissions,ozone hole,hole in ozone layer",
  "targetMedia": ["http://i.huffpost.com/gen/1515029/images/n-OZONE-HOLE-628x314.jpg", "..."],
  "targetUrl": "http://huff.to/1cH672Z",
  "provenance": {"source": "anonymized", "humanSpoiler": "2070", "spoilerPublisher": "HuffPoSpoilers"},
  "spoiler": ["2070"],
  "spoilerPositions": [[[0, 0], [0, 4]]],
  "tags": ["phrase"]
}
```


---

## 6. Summary statistics

Computed on our final splits (4,951 posts). Word counts use whitespace splitting. Token counts use the `roberta-base` tokenizer on the title plus the article (the context our QA model reads).

| | Phrase | Passage | Multi | All |
|---|---|---|---|---|
| Posts | 2,104 | 1,980 | 867 | 4,951 |
| Post length, words (median) | 10.5 | 11 | 10 | 11 |
| Article paragraphs (median) | 9 | 10 | 16 | 10 |
| Article length, words (median / mean) | 309 / 430 | 354 / 511 | 544 / 752 | 354 / 519 |
| Context length, RoBERTa tokens (median) | 421 | 478 | 740 | 485 |
| Contexts longer than 384 / 512 tokens | 56.6% / 38.3% | 63.1% / 46.0% | 81.0% / 69.0% | 63.5% / 46.7% |
| Spoiler span length, words (median / 90th pct / max) | 2 / 4 / 11 | 17 / 39 / 174 | 5 / 18 / 117 | 5 / 25 / 174 |
| Spans per post | always 1 | always 1 | 1–12 | |

Multi spoilers by number of spans: 1 span: 30 · 2: 311 · 3: 79 · 4: 43 · 5: 389 · 6: 11 · 8: 1 · 10: 2 · 12: 1.

Two points from the exploratory analysis (`notebooks/01_eda.ipynb`) shape how the data is used:
- **Type is largely a matter of spoiler length.** On train, "5 words or fewer → phrase" separates phrase from passage spoilers 98.5% of the time when the gold spoiler is known.
- **Articles are long relative to model input.** Most contexts exceed 384 tokens, so QA models need a sliding window over the article. Simply truncating would drop the spoiler for 7% of phrase posts, 16% of passage posts and at least one span for 48% of multi posts.

---

## 7. Known issues and quirks

These are properties of the original release, left as they are in our files. Where it matters, we note how we handle them in our code.

1. **Most positions are exact, but not all.** 7,080 of 7,196 spans (98.4%) are reproduced exactly by `paragraph[start:end]`. The rest:
   - 89 spans **cross paragraph boundaries**, and their offsets don't follow the per-paragraph format (most start past the end of their paragraph);
   - 26 spans have offsets that **point at different text**;
   - 1 position is **malformed** (it has a start but no end).

   *Handling:* use the position when it reproduces the gold text, otherwise search for the gold text in the whitespace-normalized article. On the full release this recovers about 99.8% of all spans. Only 16 posts have a span that doesn't occur verbatim in the article, usually because of accents or punctuation edited by the annotator.
2. **Spoilers in the title.** Paragraph `-1` refers to `targetTitle` (108 spans). Any model input should include the title, which ours puts before the paragraphs.
3. **Multi spoilers with one span.** 30 posts are labeled `multi` but have a single span. They read like passage spoilers.
4. **Multi spoilers seem capped at five spans.** This isn't documented, but on the full release, 208 of the 240 multi posts whose text promises more than five items (e.g., *"23 facts that will totally mess with your perception of time"*) have exactly five spans.
5. **Repeated spoiler strings.** 44% of phrase spoilers occur two or more times in their article. The position marks the occurrence the annotator chose.
6. **Inconsistent casing.** `postPlatform` mixes `Twitter`, `reddit` and `Facebook`. `spoilerPublisher` has both `savedyouaclick` and `SavedYouAClick`. We lowercase both.
7. **Text noise.** Curly and straight quotes are both used, 29 posts contain HTML entities such as `&amp;`, and 820 `humanSpoiler` texts include hashtags like `#StopClickBait`.
8. **Sparse metadata.** `targetDescription`, `targetKeywords`, `targetMedia` and `targetUrl` are often missing (counts in section 4). We don't rely on them.
9. **Generic repeated headlines.** A few post texts recur with different articles (e.g., daily newsletter headlines). These are different items and are kept, including across splits.

---

## 8. Evaluation

We follow the SemEval-2023 Task 5 setup so that our numbers are comparable with published results:
- **Spoiler type classification:** balanced accuracy over the three types.
- **Spoiler generation:** BLEU-4 (main metric), METEOR and BERTScore between the predicted and gold spoiler, reported overall and per type.

The official evaluator is in the shared-task repository: https://github.com/pan-webis-de/SEMEVAL-2023
