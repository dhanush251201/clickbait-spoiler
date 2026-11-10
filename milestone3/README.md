# Milestone 3: Extension 1, trained-model comparison

Everything for Milestone 3 of the SpoilerQA project (CIS 5300, due Dec 7, 2026). The extension asks **whether a bigger or different kind of model beats the Milestone 2 strong baseline, and on which spoiler types.** Three models are each trained with three learning rates and compared with the strong baseline on the same test set. `model-comparison.md` explains the extension and holds the results.

| Course deliverable | File(s) |
|---|---|
| Code for the extension | `1-deberta-qa.ipynb`, `2-flan-t5.ipynb`, `3-llama-qlora.ipynb`, `4-compare.ipynb` (+ `spoilerqa/`, `score.py`, `run-all.sh`) |
| Markdown explaining how to use the code | this README (and the top cell of each notebook) |
| 2–3 paragraph explanation + empirical evaluation | `model-comparison.md` |
| ≤ 1-page report, draft slides | written by the team |

## The models

| Notebook | Model | Size | How it spoils | Learning rates | Time on the L4 | Test BLEU |
|---|---|---|---|---|---|---|
| (Milestone 2) | RoBERTa-base QA + RoBERTa-base type classifier | 125M | extracts spans | 1e-5 / 3e-5 / 5e-5 | 36 min | 0.384 |
| `1-deberta-qa.ipynb` | `deepset/deberta-v3-large-squad2` + the same classifier | 435M | extracts spans | 5e-6 / 1e-5 / 2e-5 | ≈ 3.5 h (66 min per learning rate) | 0.438 |
| `2-flan-t5.ipynb` | `google/flan-t5-large` | 783M | writes `type: spoiler` | 5e-5 / 1e-4 / 3e-4 | ≈ 9 h (2.8 h per learning rate) | 0.462 |
| `3-llama-qlora.ipynb` | `meta-llama/Llama-3.1-8B-Instruct`, QLoRA (4-bit, rank 16) | 8B (42M trained) | writes `type:` / `spoiler:` lines | 5e-5 / 1e-4 / 2e-4 | ≈ 12.5 h (4 h per learning rate) | **0.507** |
| `4-compare.ipynb` | reads all results: tables, significance tests, chart | | | | seconds, no GPU | |

The times are from our 2026-11-07/08 runs: training per learning rate (including the dev evaluation after each epoch), plus prediction.

## Folder layout

```
milestone3/
├── 1-deberta-qa.ipynb      DeBERTa-v3-large extractive QA (Milestone 2 pipeline, bigger reader)
├── 2-flan-t5.ipynb         Flan-T5-large, generates the type and the spoiler
├── 3-llama-qlora.ipynb     Llama-3.1-8B-Instruct fine-tuned with QLoRA, generates the type and the spoiler
├── 4-compare.ipynb         comparison tables, paired bootstrap tests, BLEU-per-type chart
├── run-all.sh              runs the four notebooks in order, with logs in logs/
├── model-comparison.md     the extension write-up and results
├── score.py                the Milestone 2 evaluation script (copy)
├── spoilerqa/              shared helpers: Milestone 2's data.py, qa.py, metrics.py, plus gen.py for the
│                           generative models (input/target formats, generation, parsing outputs)
├── inputs/milestone2/      frozen copies of Milestone 2 outputs: type-classifier predictions (used by the
│                           DeBERTa notebook) and the strong baseline's predictions and scores (for comparison)
├── setup_meteor.sh, tools/ official METEOR 1.5 (tools/ is not committed)
├── requirements.txt
├── output/                 predictions (.jsonl), sweeps, scores (output/scores/), comparison tables, figures
└── runs/                   checkpoints and adapters (not committed)
```

## Running it on the L4 VM

**Once, before the first run:**
```bash
cd ~/project/milestone3
source ~/project/.venv/bin/activate
pip install -r requirements.txt     # adds peft, bitsandbytes, papermill, matplotlib to the M2 environment
cp -r ../milestone2/tools .         # official METEOR, already downloaded in Milestone 2
```
For Llama, also:
1. Accept the license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct.
2. Run `hf auth login` on the VM with a read token (or `export HF_TOKEN=...` in the shell that runs the notebooks).

**Running everything takes about 25 hours.** That is more than the VM's 24-hour limit, so use two sessions, each inside tmux:
```bash
tmux new -s m3                      # detach: Ctrl-b d · reattach: tmux attach -t m3
bash run-all.sh 1-deberta-qa.ipynb 2-flan-t5.ipynb       # session 1, ≈ 12.5 h
bash run-all.sh 3-llama-qlora.ipynb 4-compare.ipynb      # session 2, ≈ 12.5 h (after restarting the VM)
```
`bash run-all.sh` with no arguments runs all four notebooks in order.
- `run-all.sh` runs the notebooks with papermill: it saves each notebook after every cell and streams its output to `logs/<notebook>.log`. Follow along with `tail -f logs/3-llama-qlora.log`.
- If a notebook fails, the next one still runs, and the script prints which ones failed at the end.
- **Re-running is safe:** a learning rate whose `runs/<model>-lr<lr>/run_info.json` exists is skipped. An interrupted Llama run resumes from its last saved epoch.

**One notebook at a time** works too: `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 1-deberta-qa.ipynb`.

**Disk and memory:**
- Checkpoints need about 15 GB for DeBERTa, about 28 GB for Flan-T5 and about 2 GB for Llama, plus the 16 GB Llama download. The VM has about 175 GB free.
- Everything fits in the L4's 24 GB: Llama is loaded in 4 bits, and Flan-T5 uses gradient checkpointing.

**Afterwards:**
- Copy the results to the Mac: `rsync -az nlp-l4:project/milestone3/output/ milestone3/output/`. Also copy the executed notebooks if you want their logs.
- Then fill in the results in `model-comparison.md`. `4-compare.ipynb` writes the tables as markdown to `output/comparison-*.md`.
- Stop the VM.

## Smoke tests

Each notebook has a `SMOKE=1` switch. It uses a tiny stand-in model (DeBERTa-v3-xsmall, Flan-T5-small, SmolLM2-135M-Instruct), 16 training posts, 8 dev and 8 test posts and 2 training steps, and writes everything under `smoke/`:
```bash
SMOKE=1 jupyter nbconvert --to notebook --execute 3-llama-qlora.ipynb --output smoke-run.ipynb
```
On a laptop this runs on the CPU, without 4-bit loading. On the VM it uses the GPU with 4-bit loading and the paged 8-bit optimizer, so it tests the real Llama setup in a couple of minutes. It doesn't need the Llama license.

## Status

- Code written and smoke-tested on the Mac (CPU) and on the L4 (GPU, including 4-bit QLoRA).
- **Full run done on the L4 (2026-11-07/08).**
  - All 9 learning-rate runs finished, and `output/` was copied to the Mac.
  - Re-scoring the test predictions on the Mac gives the same BLEU.
  - The executed notebooks are saved with their outputs. `4-compare.ipynb` was re-run on the Mac to add the error table.
- **Results:**
  - Llama-3.1-8B QLoRA reaches 0.507 test BLEU, above the best SemEval-2023 system (0.48).
  - Flan-T5-large reaches 0.462 and DeBERTa-v3-large 0.438, against 0.384 for the strong baseline. All three gains are significant.
  - Details and analysis are in `model-comparison.md`.
- The checkpoints (43 GB) are only on the VM, under `~/project/milestone3/runs/`.
