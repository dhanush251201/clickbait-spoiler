| system | size | types | BLEU | phrase | passage | multi | METEOR | BERTScore | type bal. acc. | BLEU (gold types) |
|---|---|---|---|---|---|---|---|---|---|---|
| RoBERTa-base QA (M2 baseline) | 125M | M2 classifier | 0.3843 | 0.5704 | 0.2219 | 0.3080 | 0.4182 | 0.9140 | 0.6920 | 0.4091 |
| DeBERTa-v3-large QA | 435M | M2 classifier | 0.4377 | 0.6026 | 0.2925 | 0.3731 | 0.4885 | 0.9234 | 0.6920 | 0.4818 |
| Flan-T5-large | 783M | generated | 0.4621 | 0.5955 | 0.3346 | 0.4332 | 0.4252 | 0.9253 | 0.7473 | 0.5141 |
| Llama-3.1-8B QLoRA | 8B (42M trained) | generated | 0.5071 | 0.6481 | 0.3659 | 0.4911 | 0.5079 | 0.9337 | 0.7585 | 0.5607 |
| Reference: SemEval organizer baseline (DeBERTa-large QA) |  |  | 0.4000 | 0.6500 | 0.2400 | 0.1200 |  |  |  |  |
| Reference: best SemEval-2023 system |  |  | 0.4800 | 0.6600 | 0.3200 | 0.4400 |  |  | 0.7400 |  |
