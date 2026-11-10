#!/usr/bin/env bash
# Downloads METEOR 1.5 and its English paraphrase table into tools/meteor-1.5/, the same files the
# official SemEval-2023 Task 5 evaluator uses (webis/pan-clickbait-spoiling-evaluator Docker image).
# score.py finds them there; without them it falls back to NLTK's METEOR (an approximation).
# Needs Java on the PATH (on the VM: sudo apt-get install -y default-jre).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p tools/meteor-1.5/data

JAR_URL="https://github.com/pan-webis-de/pan-code/raw/master/semeval23/evaluation/meteor-1.5/meteor-1.5.jar"
PARAPHRASE_URL="https://files.webis.de/data-in-production/data-research/acl22-clickbait-spoiling/paraphrase-en.gz"

[ -f tools/meteor-1.5/meteor-1.5.jar ] || curl -L --fail -o tools/meteor-1.5/meteor-1.5.jar "$JAR_URL"
[ -f tools/meteor-1.5/data/paraphrase-en.gz ] || curl -L --fail -o tools/meteor-1.5/data/paraphrase-en.gz "$PARAPHRASE_URL"
ls -l tools/meteor-1.5 tools/meteor-1.5/data
