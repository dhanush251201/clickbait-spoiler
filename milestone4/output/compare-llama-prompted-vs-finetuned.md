| Llama 3.1 8B | BLEU | phrase | passage | multi | type bal. acc. | predicted phrase | predicted passage | predicted multi | items per post | several items, phrase/passage | words per post (gold phrase) | words per post (gold passage) | copied from the article | zero BLEU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prompted (0 worked examples) | 0.198 | 0.270 | 0.160 | 0.110 | 0.545 | 0.138 | 0.707 | 0.155 | 2.519 | 0.058 | 9.000 | 17.000 | 0.974 | 0.310 |
| fine-tuned (QLoRA) | 0.507 | 0.648 | 0.366 | 0.491 | 0.758 | 0.433 | 0.413 | 0.154 | 1.437 |  | 2.000 | 20.000 | 0.991 | 0.111 |
| gold (for reference) |  |  |  |  |  | 0.423 | 0.403 | 0.174 | 1.463 | 0.000 | 2.000 | 18.000 |  |  |
