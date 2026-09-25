#!/bin/zsh
# Prompt-level defenses (phase 12) and SelectorGroupChat (phase 13), after the
# detector panel finishes, one process at a time against LM Studio.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export HF_HUB_OFFLINE=1
while pgrep -f "run_experiment.py 11" >/dev/null; do sleep 20; done
batch() { while true; do rc=0; $PY src/run_experiment.py "$@" --max-runs 40 || rc=$?; [ $rc -eq 3 ] || return $rc; done }
batch 12
batch 13
echo SWEEP2_DONE
