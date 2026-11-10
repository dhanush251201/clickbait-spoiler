| system | learning rate | best dev BLEU | best epoch | dev BLEU per epoch | dev decoded with | train minutes |
|---|---|---|---|---|---|---|
| RoBERTa-base QA (M2 baseline) | 1e-05 | 0.4434 | 2 | 0.412 / 0.443 / 0.435 | gold types | 8.6000 |
| RoBERTa-base QA (M2 baseline) | 3e-05 | 0.4432 | 3 | 0.418 / 0.429 / 0.443 | gold types | 8.6000 |
| RoBERTa-base QA (M2 baseline) | 5e-05 | 0.4272 | 2 | 0.419 / 0.427 / 0.421 | gold types | 8.6000 |
| DeBERTa-v3-large QA | 5e-06 | 0.4857 | 3 | 0.468 / 0.481 / 0.486 | gold types | 66.0000 |
| DeBERTa-v3-large QA | 1e-05 | 0.4964 | 3 | 0.473 / 0.483 / 0.496 | gold types | 66.0000 |
| DeBERTa-v3-large QA | 2e-05 | 0.4887 | 3 | 0.476 / 0.488 / 0.489 | gold types | 66.0000 |
| Flan-T5-large | 5e-05 | 0.4157 | 4 | 0.262 / 0.329 / 0.345 / 0.416 / 0.411 | own types | 170.6000 |
| Flan-T5-large | 0.0001 | 0.4262 | 5 | 0.274 / 0.358 / 0.408 / 0.424 / 0.426 | own types | 165.9000 |
| Flan-T5-large | 0.0003 | 0.4505 | 5 | 0.333 / 0.362 / 0.425 / 0.443 / 0.450 | own types | 167.6000 |
| Llama-3.1-8B QLoRA | 5e-05 | 0.4905 | 2 | 0.475 / 0.490 | own types | 241.5000 |
| Llama-3.1-8B QLoRA | 0.0001 | 0.5018 | 2 | 0.466 / 0.502 | own types | 239.2000 |
| Llama-3.1-8B QLoRA | 0.0002 | 0.5038 | 2 | 0.482 / 0.504 | own types | 242.7000 |
