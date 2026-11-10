# Rating the summaries

Each row of `human-rating-sheet.csv` is a clickbait post, its spoiler (the information the post hides; several parts are separated by |) and a two-sentence summary of the article written by a language model. Fill in three columns, then save your copy as `human-ratings-<your initials>.csv` in this folder.

- **reveals**: `yes` if a reader of the summary learns the spoiler (in any wording), `partly` if they learn only part of it (some parts of a several-part spoiler, or a vaguer version), otherwise `no`.
- **faithful** (1-5): 5 = everything in the summary is supported by the article (column `article`); 1 = it states something wrong or invented.
- **fluent** (1-5): 5 = clear, natural English; 1 = hard to read.
- **notes** (optional): anything odd.

Rate each row on its own; the rows are shuffled and don't say which model wrote them. Don't open `human-rating-key.csv`.
