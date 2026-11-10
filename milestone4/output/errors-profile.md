| system | kind | predicted phrase | predicted passage | predicted multi | items per multi answer | words, phrase | words, passage | zero BLEU |
|---|---|---|---|---|---|---|---|---|
| Simple baseline (first sentence) | baseline | 1.000 | 0.000 | 0.000 |  | 18.000 | 18.000 | 0.544 |
| RoBERTa-base QA (M2 strong baseline) | fine-tuned | 0.400 | 0.395 | 0.205 | 5.000 | 2.000 | 11.000 | 0.169 |
| DeBERTa-v3-large QA | fine-tuned | 0.400 | 0.395 | 0.205 | 5.000 | 2.000 | 13.000 | 0.137 |
| Flan-T5-large | fine-tuned | 0.389 | 0.474 | 0.137 | 4.234 | 2.000 | 22.000 | 0.136 |
| Llama 3.1 8B, fine-tuned (QLoRA) | fine-tuned | 0.433 | 0.413 | 0.154 | 3.838 | 2.000 | 20.000 | 0.111 |
| Llama 3.1 8B, prompted | prompted | 0.138 | 0.707 | 0.155 | 9.613 | 9.000 | 17.000 | 0.310 |
| GPT-OSS 20B | prompted | 0.800 | 0.132 | 0.068 | 6.912 | 2.000 | 6.000 | 0.256 |
| Claude Sonnet 5.5 | prompted | 0.419 | 0.450 | 0.131 | 4.206 | 3.000 | 24.000 | 0.093 |
| Claude Haiku 5.5 | prompted | 0.289 | 0.583 | 0.128 | 4.508 | 5.000 | 21.000 | 0.160 |
| GPT-6.1 Sol | prompted | 0.421 | 0.459 | 0.120 | 4.300 | 3.000 | 20.000 | 0.104 |
| GPT-6 Luna | prompted | 0.561 | 0.334 | 0.105 | 5.095 | 3.000 | 16.000 | 0.127 |
| gold |  | 0.423 | 0.403 | 0.174 | 3.661 | 2.000 | 18.000 |  |
