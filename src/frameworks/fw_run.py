"""Send one framework-team run to the workbench's Python SDK runner."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import harness as H  # noqa: E402

CFG = H.CFG
FW = pathlib.Path(__file__).resolve().parent


def fw_cfg():
    return {"lmstudio_base": CFG["lmstudio_base"], "agent_model": CFG["agent_model"],
            "guard_model": CFG["guards"]["llm"]["model"], "seed": CFG["seed"],
            "reasoning_effort": CFG["reasoning_effort"], "guard_prompt": H.LLMGuard.SYSTEM,
            "researcher_sys": H.RESEARCHER_SYS, "writer_sys": H.WRITER_SYS, "task": H.ENTRY_TASK["tool"]}


def run(framework, probe, timeout=900):
    code = (FW / f"{framework}_team.py").read_text()
    params = {"text": probe["text"], "passage": probe["passage"].strip(),
              "canary": probe["canary"], "cfg": fw_cfg()}
    resp = H.post_json(CFG["workbench_base"] + "/api/sdk/run",
                       {"code": code, "params": params, "timeout": timeout}, timeout=timeout + 60)
    out = None
    for line in reversed((resp.get("stdout") or "").splitlines()):
        if line.startswith("{"):
            try:
                out = json.loads(line)
                break
            except json.JSONDecodeError:
                pass
    return resp, out
