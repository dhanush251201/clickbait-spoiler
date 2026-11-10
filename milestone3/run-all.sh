#!/usr/bin/env bash
# Runs every Milestone 3 notebook in order. Start it inside tmux:
#
#   tmux new -s m3
#   bash run-all.sh
#
# All four take about 25 hours on the L4 (DeBERTa 3.5 h, Flan-T5 9 h, Llama 12.5 h), more than the VM's 24-hour
# limit, so run them in two sessions:
#   bash run-all.sh 1-deberta-qa.ipynb 2-flan-t5.ipynb
#   bash run-all.sh 3-llama-qlora.ipynb 4-compare.ipynb
#
# Each notebook's output streams to logs/<notebook>.log (follow it with: tail -f logs/2-flan-t5.log), and the
# notebook is saved after every cell. If a notebook fails, the next one still runs. Running the script again
# skips learning rates that already finished (and resumes an interrupted Llama run from its last epoch).
# To run only some notebooks: bash run-all.sh 2-flan-t5.ipynb 4-compare.ipynb
set -u
cd "$(dirname "$0")"
mkdir -p logs

notebooks=("$@")
if [ ${#notebooks[@]} -eq 0 ]; then
  notebooks=(1-deberta-qa.ipynb 2-flan-t5.ipynb 3-llama-qlora.ipynb 4-compare.ipynb)
fi

failed=()
for nb in "${notebooks[@]}"; do
  log="logs/${nb%.ipynb}.log"
  echo "$(date '+%F %T')  start  $nb  (log: $log)"
  if papermill "$nb" "$nb" --log-output --request-save-on-cell-execute > "$log" 2>&1; then
    echo "$(date '+%F %T')  done   $nb"
  else
    echo "$(date '+%F %T')  FAILED $nb  (see the end of $log)"
    failed+=("$nb")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "failed: ${failed[*]}"
  exit 1
fi
echo "all done; results in output/ (summary: output/comparison-test.md)"
