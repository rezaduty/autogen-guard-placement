# Guard Placement and Interception Timing for Runtime Risk Control in AutoGen Multi-Agent Teams

Measurement artifact for the paper. One guard is held fixed and moved between
the three interception points AutoGen exposes: the model client, the runtime
message bus, and the tool boundary. The same team is rebuilt in LangGraph and
CrewAI, and AutoGen's text-mention termination condition is tested against
data that contains its keyword.

Every number in the paper is generated from the raw logs in `results/` by
`src/analyze.py` and re-derived by `src/verify_numbers.py`, which fails the
build on any disagreement.

## Layout

    config.json            models, endpoints, seeds, test carriers
    build.sh               analysis, figures, checks, verifier, PDF
    make_arxiv.sh          source tarball with a clean-room compile
    data/README.md         datasheet: probes, keyword probes, InjecAgent, panel corpora
    data/probes/           probe set, keyword probes, composition
    data/sources/          frozen NWS and arXiv snapshot
    data/thirdparty/       frozen InjecAgent selection and tool schemas
    src/fetch_sources.py   writes the source snapshot (run once)
    src/build_probes.py    builds the probe set from the snapshot only
    src/fetch_thirdparty.py freezes the InjecAgent selection
    src/harness.py         AutoGen team, guards, the three placements
    src/run_experiment.py  numbered phases 0 to 13
    src/frameworks/        the same team in AutoGen, LangGraph and CrewAI
    src/studio_workflows.py the scenarios as bench workflows
    src/analyze.py         every macro and generated table
    src/verify_numbers.py  independent, fail-closed verifier
    src/check_figures.py   figure geometry, arrows and text rules
    src/test_checks.py     breaks every check on purpose and confirms it fires
    results/               raw per-phase logs, captures, hand-read labels
    paper/                 source, generated tables, figures

## Phases

| phase | what runs |
| --- | --- |
| 0 | capture versions, served models, model echo, bench state |
| 1 | guards alone on every probe: full probe, passage, memory-wrapped |
| 2 | AutoGen team, all placements observing |
| 3, 4, 5 | the same runs blocking at the model client, bus, tool boundary |
| 6 | InjecAgent cases through the tool entry point |
| 7 | the bench's NeMo Guardrails input rail on every probe |
| 8, 9 | the team in LangGraph and CrewAI, native hooks observing |
| 10 | text-mention termination against data containing its keyword |
| 11 | detector panel: four published classifiers and two language-model guards |
| 12 | prompt-level defenses: spotlighting and the sandwich defense |
| 13 | SelectorGroupChat, a model-chosen speaker order |

`python src/run_experiment.py N` runs phase N. Runs resume by id. Team phases
take `--subset` for the ten test carriers used in the paper and
`--max-runs` to run in batches.

## Requirements

Python 3.11, the packages in `requirements.txt`, LM Studio serving
`qwen/qwen3.5-9b`, and `qwen/qwen3.8-27b` for the phase-1 replication. For
phases 7 to 9 it also needs the NeMo Guardrails test bench described in the paper.
The DeBERTa classifier is pinned to one revision in `config.json`.

## Licences

Code: MIT (`LICENSE`). Probe set, snapshot and results: CC-BY-4.0
(`LICENSE-DATA`). NWS text is public domain, arXiv metadata CC0, and the
AgentDojo templates and InjecAgent cases keep their MIT licences
(`LICENSE-THIRD-PARTY`).
