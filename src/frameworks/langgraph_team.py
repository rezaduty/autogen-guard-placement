"""
The AutoGen team rebuilt in LangGraph, with the guard at LangGraph's native
interception points. Runs inside the workbench's Python SDK runner.

Input: see load_input().
Prints one JSON line: the event log and the outputs.

Placements, each a documented LangChain or LangGraph mechanism:
  model    AgentMiddleware.before_model: every input message, before the
           model reads it
  tool     AgentMiddleware.wrap_tool_call: the tool's result, before the
           agent's model reads it
  handoff  a StateGraph node between the two agents: the researcher's final
           message, before the writer receives it, and the researcher's
           tool messages, which the node sees only after the researcher read them
"""
import json
import re
import sys
import urllib.request
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Two ways in. The experiment passes a params object, which the workbench SDK
# runner delivers as sys.argv[2]. A Studio Python SDK node passes the upstream
# text raw as sys.argv[1], and the workflow builder writes STUDIO_CFG into the
# node's copy of this script.
STUDIO_CFG = None


def load_input():
    if len(sys.argv) > 2 and sys.argv[2] not in ("", "{}", "null"):
        return json.loads(sys.argv[2])
    text = sys.argv[1] if len(sys.argv) > 1 else ""
    m = re.search(r"RX-[0-9A-F]{6}", text)
    return {"text": text, "passage": "", "canary": m.group(0) if m else "", "cfg": STUDIO_CFG}


INP = load_input()
CFG = INP["cfg"]
EVENTS, SEQ, CACHE, NOTES = [], [0], {}, []


def log(kind, **f):
    SEQ[0] += 1
    EVENTS.append({"seq": SEQ[0], "kind": kind, **f})


def guard(text):
    if text in CACHE:
        return CACHE[text]
    body = {"model": CFG["guard_model"], "temperature": 0, "seed": CFG["seed"], "max_tokens": 8,
            "reasoning_effort": CFG["reasoning_effort"],
            "messages": [{"role": "system", "content": CFG["guard_prompt"]}, {"role": "user", "content": text}]}
    req = urllib.request.Request(CFG["lmstudio_base"] + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    out = (json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"].get("content") or "").strip()
    w = out.upper().split()[0].strip(".,:*") if out else ""
    CACHE[text] = w == "INJECTION"
    return CACHE[text]


def scan(placement, kind, text, agent, in_time=True):
    flag = guard(text)
    log("guard", placement=placement, msg_kind=kind, agent=agent, in_time=in_time, flag=flag,
        chars=len(text), contains_passage=bool(INP["passage"]) and INP["passage"][:60] in text,
        contains_canary=bool(INP["canary"]) and INP["canary"] in text)


def text_of(m):
    c = m.content
    return c if isinstance(c, str) else " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)


class Guard(AgentMiddleware):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def before_model(self, state, runtime):
        for m in state["messages"]:
            if isinstance(m, (HumanMessage, SystemMessage, ToolMessage)):
                scan("model", type(m).__name__, text_of(m), self.agent)
        return None

    def wrap_tool_call(self, request, handler):
        result = handler(request)
        if request.tool_call["name"] == "fetch_document":
            scan("tool", "fetch_document", text_of(result), self.agent)
        return result


@tool
def fetch_document(doc_id: str) -> str:
    """Fetch a document by its identifier and return its text."""
    log("tool_call", tool="fetch_document", args={"doc_id": doc_id})
    return INP["text"]


def make_save_note(agent):
    @tool
    def save_note(text: str) -> str:
        """Save a short note for later."""
        log("tool_call", tool="save_note", agent=agent, args={"text": text})
        NOTES.append({"agent": agent, "text": text})
        return "Note saved."
    return save_note


def model():
    return ChatOpenAI(model=CFG["agent_model"], base_url=CFG["lmstudio_base"], api_key="lm-studio",
                      temperature=0, seed=CFG["seed"], extra_body={"reasoning_effort": CFG["reasoning_effort"]})


researcher = create_agent(model(), tools=[fetch_document, make_save_note("researcher")],
                          system_prompt=CFG["researcher_sys"], middleware=[Guard("researcher")])
writer = create_agent(model(), tools=[make_save_note("writer")], system_prompt=CFG["writer_sys"],
                      middleware=[Guard("writer")])


class S(TypedDict):
    task: str
    researcher_msgs: list
    handoff: str
    final: str


def run_researcher(s):
    out = researcher.invoke({"messages": [HumanMessage(s["task"])]})
    msgs = out["messages"]
    log("agent_done", agent="researcher", output=text_of(msgs[-1]))
    return {"researcher_msgs": msgs}


def handoff(s):
    # Tool messages are in the state only after the researcher consumed them.
    for m in s["researcher_msgs"]:
        if isinstance(m, ToolMessage):
            scan("handoff", "ToolMessage", text_of(m), "researcher", in_time=False)
    reply = text_of(s["researcher_msgs"][-1])
    scan("handoff", "AgentReply", reply, "researcher", in_time=True)
    return {"handoff": reply}


def run_writer(s):
    out = writer.invoke({"messages": [HumanMessage(s["task"]), HumanMessage("researcher: " + s["handoff"])]})
    final = text_of(out["messages"][-1])
    log("agent_done", agent="writer", output=final)
    return {"final": final}


g = StateGraph(S)
g.add_node("researcher", run_researcher)
g.add_node("handoff", handoff)
g.add_node("writer", run_writer)
g.set_entry_point("researcher")
g.add_edge("researcher", "handoff")
g.add_edge("handoff", "writer")
g.add_edge("writer", END)

error, final = None, ""
try:
    final = g.compile().invoke({"task": CFG["task"]})["final"]
except Exception as e:
    error = f"{type(e).__name__}: {e}"
print(json.dumps({"framework": "langgraph", "final": final, "notes": NOTES, "events": EVENTS, "error": error}))
