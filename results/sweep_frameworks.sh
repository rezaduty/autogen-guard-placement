#!/bin/zsh
# Runs after the main sweep so that only one experiment uses LM Studio at a
# time. Batched like the main sweep: exit 3 means more runs remain.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export HF_HUB_OFFLINE=1
until grep -q SWEEP_DONE results/sweep.log; do sleep 60; done
batch() { while true; do $PY src/run_experiment.py "$@" --max-runs 40; rc=$?; [ $rc -eq 3 ] || return $rc; done }
batch 10
echo PHASE10_DONE
$PY src/run_experiment.py 8
$PY src/run_experiment.py 9
echo FRAMEWORKS_DONE
