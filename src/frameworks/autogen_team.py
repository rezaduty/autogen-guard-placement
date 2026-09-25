"""
The AutoGen team of the main experiment, tool entry point, as one standalone
script in the same output format as langgraph_team.py and crewai_team.py, so a
Studio workflow can run the three frameworks side by side. The paper's AutoGen
numbers come from src/harness.py, which this script mirrors.

Placements, each an AutoGen interface:
  model    a ChatCompletionClient wrapper: every input message before the model
  tool     a wrapper around fetch_document: the result before the agent reads it
  handoff  a runtime InterventionHandler: the agent reply delivered to the
           writer, and the output-stream copy of the tool result, which the
           bus carries only after the researcher read it
"""
import asyncio
import json
import re
import sys
import urllib.request

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.teams._group_chat._events import GroupChatAgentResponse, GroupChatMessage
from autogen_core import DefaultInterventionHandler, SingleThreadedAgentRuntime
from autogen_core.models import ChatCompletionClient, FunctionExecutionResultMessage, SystemMessage, UserMessage
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

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


class Client(ChatCompletionClient):
    def __init__(self, agent):
        self.agent = agent
        self.inner = OpenAIChatCompletionClient(
            model=CFG["agent_model"], base_url=CFG["lmstudio_base"], api_key="lm-studio", temperature=0,
            seed=CFG["seed"], extra_body={"reasoning_effort": CFG["reasoning_effort"]},
            model_info={"vision": False, "function_calling": True, "json_output": False, "family": "unknown",
                        "structured_output": False, "multiple_system_messages": True})

    async def create(self, messages, **kw):
        for m in messages:
            if isinstance(m, (SystemMessage, UserMessage)) and isinstance(m.content, str):
                scan("model", type(m).__name__, m.content, self.agent)
            elif isinstance(m, FunctionExecutionResultMessage):
                scan("model", "FunctionExecutionResultMessage", "\n".join(r.content for r in m.content), self.agent)
        return await self.inner.create(messages, **kw)

    def create_stream(self, *a, **k):
        raise NotImplementedError

    async def close(self):
        await self.inner.close()

    def actual_usage(self):
        return self.inner.actual_usage()

    def total_usage(self):
        return self.inner.total_usage()

    def count_tokens(self, m, **k):
        return self.inner.count_tokens(m, **k)

    def remaining_tokens(self, m, **k):
        return self.inner.remaining_tokens(m, **k)

    @property
    def capabilities(self):
        return self.inner.capabilities

    @property
    def model_info(self):
        return self.inner.model_info


class Bus(DefaultInterventionHandler):
    async def on_publish(self, message, *, message_context):
        if isinstance(message, GroupChatAgentResponse) and message.name == "researcher":
            c = message.response.chat_message.content
            if isinstance(c, str):
                scan("handoff", "AgentReply", c, "researcher", in_time=True)
                log("agent_done", agent="researcher", output=c)
        elif isinstance(message, GroupChatMessage) and type(message.message).__name__ == "ToolCallExecutionEvent":
            for r in message.message.content:
                if r.name == "fetch_document":
                    scan("handoff", "ToolCallExecutionEvent", r.content, "researcher", in_time=False)
        return message


async def fetch_document(doc_id: str) -> str:
    """Fetch a document by its identifier and return its text."""
    log("tool_call", tool="fetch_document", args={"doc_id": doc_id})
    scan("tool", "fetch_document", INP["text"], "researcher")
    return INP["text"]


def save_note_for(agent):
    async def save_note(text: str) -> str:
        """Save a short note for later."""
        log("tool_call", tool="save_note", agent=agent, args={"text": text})
        NOTES.append({"agent": agent, "text": text})
        return "Note saved."
    return FunctionTool(save_note, description="Save a short note for later.")


async def main():
    researcher = AssistantAgent("researcher", model_client=Client("researcher"), system_message=CFG["researcher_sys"],
                                tools=[FunctionTool(fetch_document, description="Fetch a document by its identifier."),
                                       save_note_for("researcher")], reflect_on_tool_use=True, max_tool_iterations=3)
    writer = AssistantAgent("writer", model_client=Client("writer"), system_message=CFG["writer_sys"],
                            tools=[save_note_for("writer")], reflect_on_tool_use=True, max_tool_iterations=3)
    rt = SingleThreadedAgentRuntime(intervention_handlers=[Bus()])
    team = RoundRobinGroupChat([researcher, writer], termination_condition=MaxMessageTermination(3), runtime=rt)
    rt.start()
    try:
        res = await team.run(task=CFG["task"])
    finally:
        await rt.stop_when_idle()
    final = next((m.content for m in reversed(res.messages) if m.source == "writer" and isinstance(m.content, str)), "")
    log("agent_done", agent="writer", output=final)
    return final


error, final = None, ""
try:
    final = asyncio.run(main())
except Exception as e:
    error = f"{type(e).__name__}: {e}"
print(json.dumps({"framework": "autogen", "final": final, "notes": NOTES, "events": EVENTS, "error": error}))
