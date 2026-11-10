| system | kind | size | worked examples | BLEU | phrase | passage | multi | METEOR | BERTScore | type bal. acc. | BLEU (gold types) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Simple baseline (first sentence) | baseline | - |  | 0.0497 | 0.0323 | 0.0816 | 0.0182 | 0.1281 | 0.8478 | 0.3333 |  |
| RoBERTa-base QA (M2 strong baseline) | fine-tuned | 125M |  | 0.3843 | 0.5704 | 0.2219 | 0.3080 | 0.4182 | 0.9140 | 0.6920 | 0.4091 |
| DeBERTa-v3-large QA | fine-tuned | 435M |  | 0.4377 | 0.6026 | 0.2925 | 0.3731 | 0.4885 | 0.9234 | 0.6920 | 0.4818 |
| Flan-T5-large | fine-tuned | 783M |  | 0.4621 | 0.5955 | 0.3346 | 0.4332 | 0.4252 | 0.9253 | 0.7473 | 0.5141 |
| Llama 3.1 8B, fine-tuned (QLoRA) | fine-tuned | 8B (42M trained) |  | 0.5071 | 0.6481 | 0.3659 | 0.4911 | 0.5079 | 0.9337 | 0.7585 | 0.5607 |
| Llama 3.1 8B, prompted | prompted | 8B (4-bit) | 0 | 0.1978 | 0.2701 | 0.1597 | 0.1103 | 0.2286 | 0.8646 | 0.5451 |  |
| GPT-OSS 20B | prompted | 21B MoE (4-bit) | 0 | 0.3541 | 0.5851 | 0.1893 | 0.1743 | 0.4046 | 0.9052 | 0.4879 |  |
| Claude Sonnet 5.5 | prompted | not disclosed | 6 | 0.4871 | 0.5911 | 0.4225 | 0.3837 | 0.5056 | 0.9291 | 0.7348 |  |
| Claude Haiku 5.5 | prompted | not disclosed | 3 | 0.3655 | 0.4407 | 0.3280 | 0.2694 | 0.4020 | 0.9097 | 0.6525 |  |
| GPT-6.1 Sol | prompted | not disclosed | 6 | 0.5147 | 0.6436 | 0.4428 | 0.3678 | 0.5079 | 0.9355 | 0.7507 |  |
| GPT-6 Luna | prompted | not disclosed | 3 | 0.4620 | 0.6241 | 0.3517 | 0.3235 | 0.4826 | 0.9264 | 0.6649 |  |
| Reference: SemEval organizer baseline (DeBERTa-large QA) |  |  |  | 0.4000 | 0.6500 | 0.2400 | 0.1200 |  |  |  |  |
| Reference: best SemEval-2023 system |  |  |  | 0.4800 | 0.6600 | 0.3200 | 0.4400 |  |  | 0.7400 |  |
