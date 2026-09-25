"""
Prove that every verifier and figure check fires.

Each case copies the project into a scratch directory, breaks one thing, runs
the checker there, and requires a non-zero exit with the expected message.
A check that never fires is decoration. The real files are never modified.

  python src/test_checks.py
"""
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable


def copy_project(dst):
    for d in ("src", "data", "paper", "config.json"):
        s = ROOT / d
        (shutil.copytree if s.is_dir() else shutil.copy)(s, dst / d)
    # The gateway rules file lives outside the project. Point the copy at it.
    cfg = json.loads((dst / "config.json").read_text())
    cfg["gateway_rules_file"] = str((ROOT / cfg["gateway_rules_file"]).resolve())
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    # Raw results are large. Phases a case mutates are copied, the rest linked.
    (dst / "results").mkdir()
    for p in (ROOT / "results").iterdir():
        if p.name in ("phase0", "phase1", "phase7"):
            shutil.copytree(p, dst / "results" / p.name)
        else:
            (dst / "results" / p.name).symlink_to(p)


def edit(path, fn):
    """Apply a mutation. A mutation that changes nothing proves nothing, so
    it is an error rather than a pass."""
    before = path.read_text()
    after = fn(before)
    if after == before:
        raise SystemExit(f"mutation did not change {path.name}: the test is stale")
    path.write_text(after)


def jl_edit(path, fn):
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    fn(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def insert_after_keywords(text, s):
    return text.replace("\\section{Introduction}", s + "\n\n\\section{Introduction}", 1)


def first_macro(d, prefix):
    m = re.search(r"\\newcommand\{\\(" + prefix + r"[A-Za-z]*)\}\{([^}]*)\}", (d / "paper/tables/macros.tex").read_text())
    return m.group(1), m.group(2)


def mut_macro(d):
    name, val = first_macro(d, "OneLlmTextAttackK")
    edit(d / "paper/tables/macros.tex", lambda t: t.replace(f"\\newcommand{{\\{name}}}{{{val}}}", f"\\newcommand{{\\{name}}}{{{int(val) - 1}}}"))


def mut_rawflag(d):
    def f(rows):
        r = next(r for r in rows if r["guard"] == "deberta" and r["unit"] == "passage")
        r["flag"] = not r["flag"]
    jl_edit(d / "results/phase1/verdicts.jsonl", f)


def mut_hash(d):
    def f(rows):
        rows[0]["text_sha"] = "0" * 16
    jl_edit(d / "results/phase1/verdicts.jsonl", f)


def mut_echo(d):
    p = d / "results/phase0/capture.json"
    c = json.loads(p.read_text())
    c["agent_model_echoed"] = "some-other-model"
    p.write_text(json.dumps(c))


def mut_drift(d):
    p = d / "results/phase1/workbench_state.json"
    s = json.loads(p.read_text())
    s["after"] = "f" * 64
    p.write_text(json.dumps(s))


def mut_claim(d):
    # DeBERTa stops favouring the passage: every passage verdict copies the
    # full-probe verdict, raw log included, so only the claim check can object.
    rows = [json.loads(l) for l in (d / "results/phase1/verdicts.jsonl").read_text().splitlines()]
    text_flag = {r["probe_id"]: r["flag"] for r in rows if r["guard"] == "deberta" and r["unit"] == "text"}
    raw_p = d / "results/phase1/guard_raw.jsonl"
    raw = [json.loads(l) for l in raw_p.read_text().splitlines()]
    by_sha = {(r["guard"], r["text_sha"]): r for r in raw}
    for r in rows:
        if r["guard"] == "deberta" and r["unit"] == "passage":
            r["flag"] = text_flag[r["probe_id"]]
            rr = by_sha[("deberta", r["text_sha"])]
            rr["raw"] = json.dumps({"label": "INJECTION" if r["flag"] else "SAFE", "score": 0.9})
    (d / "results/phase1/verdicts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    raw_p.write_text("".join(json.dumps(r) + "\n" for r in raw))


def mut_gateway_outcome(d):
    def f(rows):
        r = next(r for r in rows if r.get("outcome") == "blocked")
        r["outcome"] = "allowed"
    jl_edit(d / "results/phase7/runs.jsonl", f)


def tex(d, s):
    edit(d / "paper/Paper.tex", lambda t: insert_after_keywords(t, s))


CASES = [
    ("numeric disagreement", "verify", mut_macro, "data gives"),
    ("verdict against raw output", "verify", mut_rawflag, "disagrees with raw output"),
    ("text hash alignment", "verify", mut_hash, "text hash does not match"),
    ("echoed model provenance", "verify", mut_echo, "echoed model differs"),
    ("workbench drift", "verify", mut_drift, "workbench state drifted"),
    ("headline claim", "verify", mut_claim, "claim broken"),
    ("datasheet count", "verify", lambda d: edit(d / "data/README.md", lambda t: t.replace("`probes/probes.jsonl`, 150 items", "`probes/probes.jsonl`, 151 items")), "datasheet item count"),
    ("undefined macro", "verify", lambda d: tex(d, "We report \\NoSuchMacro{} here."), "undefined macro"),
    ("typed percentage", "verify", lambda d: tex(d, "About 42\\% of runs."), "typed percentage"),
    ("uncited bib entry", "verify", lambda d: edit(d / "paper/bibliography.bib", lambda t: t + "\n@misc{orphan2026,\n  title = {Orphan},\n  year = {2026}\n}\n"), "never cited"),
    ("missing bib entry", "verify", lambda d: tex(d, "As shown before~\\cite{nosuchkey2026}."), "has no bibliography entry"),
    ("ref without label", "verify", lambda d: tex(d, "See Section~\\ref{sec:nowhere}."), "has no label"),
    ("missing input", "verify", lambda d: tex(d, "\\input{tables/nosuchtable}"), "missing input"),
    ("label before caption", "verify", lambda d: tex(d, "\\begin{table}\\label{tab:bad}\\caption{Bad.}\\end{table}"), "label precedes its caption"),
    ("em dash", "verify", lambda d: tex(d, "A claim --- with a dash."), "em dash"),
    ("en dash", "verify", lambda d: tex(d, "Pages 3--4 of it."), "en dash"),
    ("arrow", "verify", lambda d: tex(d, "Input -> output."), "arrow"),
    ("semicolon", "verify", lambda d: tex(d, "One clause; another clause."), "semicolon"),
    ("vague deixis", "verify", lambda d: tex(d, "The former holds."), "banned or vague"),
    ("self-hedging", "verify", lambda d: tex(d, "We honestly report it."), "banned or vague"),
    ("unreferenced code", "verify", lambda d: tex(d, "The model answers S5 on most probes."), "without a reference"),
    ("filler word", "verify", lambda d: tex(d, "The effect is clearly large."), "banned or vague"),
    ("uncited method", "verify", lambda d: tex(d, "We also discuss AgentDojo briefly."), "first used without a citation"),
    ("datasheet em dash", "verify", lambda d: edit(d / "data/README.md", lambda t: t + "\nA line \u2014 with a dash.\n"), "README.md: em dash"),
    ("figure box overlap", "figures", lambda d: edit(d / "paper/figures/fig_placements.html", lambda t: t.replace('style="left:392px;top:212px', 'style="left:300px;top:212px')), "overlap"),
    ("figure digit", "figures", lambda d: edit(d / "paper/figures/fig_placements.html", lambda t: t.replace('<div class="t">Memory</div>', '<div class="t">Memory 3</div>')), "digit in visible text"),
    ("figure small type", "figures", lambda d: edit(d / "paper/figures/fig_placements.html", lambda t: t.replace(".s{font-size:15px", ".s{font-size:11px")), "below the"),
    ("gateway outcome against its events", "verify", mut_gateway_outcome, "outcome disagrees with its events"),
    ("figure subtitled box overflow", "figures", lambda d: edit(d / "paper/figures/fig_frameworks.html", lambda t: t.replace('<div class="t">Graph node</div>', '<div class="t">Graph node inserted between the researcher and the writer in the state graph</div>')), "needs about"),
    ("figure turned arrowhead", "figures", lambda d: edit(d / "paper/figures/fig_placements.html", lambda t: t.replace('C574,440 580,380 598,380 L620,380', 'C580,440 590,380 620,380')), "arrowhead turned"),
    ("figure arrow short of its box", "figures", lambda d: edit(d / "paper/figures/fig_placements.html", lambda t: t.replace('<line x1="300" y1="186" x2="300" y2="210"', '<line x1="300" y1="186" x2="300" y2="199"')), "from the nearest box edge"),
    ("figure wire through a box", "figures", lambda d: edit(d / "paper/figures/fig_placements.html", lambda t: t.replace('<!-- outputs -->', '<line x1="220" y1="440" x2="614" y2="440" stroke="#8a93a2" stroke-width="2" marker-end="url(#ag)"/>\n  <!-- outputs -->')), "passes through box"),
    ("figure label overflow", "figures", lambda d: edit(d / "paper/figures/fig_placements.html", lambda t: t.replace('<div class="t">Task</div>', '<div class="t">Task message typed by the user into the team chat window</div>')), "needs about"),
]


def run(kind, d):
    script = "verify_numbers.py" if kind == "verify" else "check_figures.py"
    env = {**os.environ, "TITLE36_IN_BREAK_TEST": "1"}
    return subprocess.run([PY, f"src/{script}"], cwd=d, capture_output=True, text=True, env=env)


def main():
    # The unmodified copy must pass, or every case below proves nothing.
    with tempfile.TemporaryDirectory() as t:
        d = pathlib.Path(t)
        copy_project(d)
        for kind in ("verify", "figures"):
            r = run(kind, d)
            if r.returncode != 0:
                print(f"baseline {kind} fails on the unmodified copy:\n{r.stdout[-800:]}")
                sys.exit(1)
    bad = 0
    for name, kind, mutate, expect in CASES:
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            copy_project(d)
            mutate(d)
            r = run(kind, d)
            ok = r.returncode != 0 and expect in r.stdout
            bad += not ok
            print(f"  {'fires' if ok else 'SILENT':6s}  {name}")
            if not ok:
                print("        exit", r.returncode, "|", r.stdout.strip().splitlines()[-3:])
    print(f"{len(CASES) - bad} of {len(CASES)} checks fire when broken")
    (ROOT / "results" / "check_tests.json").write_text(json.dumps(
        {"cases": len(CASES), "fire": len(CASES) - bad, "names": [c[0] for c in CASES]}, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
