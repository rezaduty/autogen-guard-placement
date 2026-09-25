#!/bin/zsh
# Test-scale run on the ten test carriers of config.json. Sequential against
# LM Studio, in batches of 40 runs per process (exit 3 means more remain).
# Bus blocking runs only where the bus receives the text before an agent
# reads it (user and agent entry), and the tool boundary exists only for
# tool entry.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export HF_HUB_OFFLINE=1
# set -e would abort on exit code 3, which means "more runs remain", so the
# exit code is read with || rather than left to set -e.
batch() { while true; do rc=0; $PY src/run_experiment.py "$@" --subset --max-runs 40 || rc=$?; [ $rc -eq 3 ] || return $rc; done }
set -e
batch 2 --block-guard llm
batch 3 --block-guard llm
batch 4 --block-guard llm --entries user,agent
batch 5 --block-guard llm --entries tool
for b in none model bus tool; do $PY src/run_experiment.py 6 --subset --block-at $b --block-guard llm; done
batch 10
$PY src/run_experiment.py 8 --subset
$PY src/run_experiment.py 9 --subset
echo SWEEP_DONE
