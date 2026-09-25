"""
Freeze the third-party corpus: InjecAgent (Zhan et al., 2024), MIT licence.

Uses the direct-harm base split. Each case pairs a user task and the tool that
serves it with an attacker instruction planted in a field of that tool's
response. Breach is InjecAgent's own criterion: the agent calls the attacker
tool. In this experiment every attacker tool is a stub that logs the call and
does nothing.

Selection: every one of the 30 attacker instructions, each with two user cases
chosen by rotating through the 17 user cases, so all user cases appear and no
instruction is over-represented. Writes data/thirdparty/injecagent_dh.jsonl and
the tool schemas the selected cases need.
"""
import json
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "thirdparty"
REPO = "https://github.com/uiuc-kang-lab/InjecAgent"
COMMIT = "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"
PER_INSTRUCTION = 2


def main():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "-q", REPO, tmp], check=True)
        subprocess.run(["git", "-C", tmp, "checkout", "-q", COMMIT], check=True)
        data = pathlib.Path(tmp) / "data"
        cases = json.loads((data / "test_cases_dh_base.json").read_text())
        toolkits = json.loads((data / "tools.json").read_text())
        # The repository names its licence file LICENCE, British spelling.
        licence = (pathlib.Path(tmp) / "LICENCE").read_text()

    instructions = sorted({c["Attacker Instruction"] for c in cases})
    users = sorted({c["User Instruction"] for c in cases})
    by_pair = {(c["Attacker Instruction"], c["User Instruction"]): c for c in cases}
    rows, k = [], 0
    for i, ins in enumerate(instructions):
        for j in range(PER_INSTRUCTION):
            u = users[k % len(users)]
            k += 1
            c = by_pair[(ins, u)]
            rows.append({
                "id": f"ia{i:02d}_{j}", "attacker_instruction": ins, "attacker_tools": c["Attacker Tools"],
                "attack_type": c["Attack Type"], "user_instruction": u, "user_tool": c["User Tool"],
                "tool_parameters": c["Tool Parameters"], "tool_response": json.loads(c["Tool Response"]),
                "source": f"InjecAgent@{COMMIT[:12]} data/test_cases_dh_base.json",
            })

    tools = {}
    for tk in toolkits:
        for t in tk["tools"]:
            tools[tk["toolkit"] + t["name"]] = {"summary": t["summary"], "parameters": t["parameters"]}
    needed = {n for r in rows for n in r["attacker_tools"] + [r["user_tool"]]}
    missing = needed - tools.keys()
    if missing:
        raise SystemExit(f"tool schemas missing for {sorted(missing)}")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "injecagent_dh.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "injecagent_tools.json").write_text(json.dumps({n: tools[n] for n in sorted(needed)}, indent=1))
    (OUT / "PROVENANCE.json").write_text(json.dumps(
        {"repo": REPO, "commit": COMMIT, "file": "data/test_cases_dh_base.json", "licence": "MIT",
         "n_cases_in_file": len(cases), "n_instructions": len(instructions), "n_user_cases": len(users),
         "n_selected": len(rows), "licence_text": licence}, indent=1))
    print(f"froze {len(rows)} cases, {len(needed)} tool schemas")


if __name__ == "__main__":
    main()
