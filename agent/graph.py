"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
import struct
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

from pydantic import BaseModel, Field
from urllib3 import response

# Total generate + revise calls before the loop is forced to stop.
# 3-5 is a reasonable range; tune it as part of Phase 3.
MAX_ITERATIONS = 3

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


class VerifyDecision(BaseModel):
    """Structured output for the verify node.

    Bound to the LLM via `.with_structured_output(method="json_schema")`,
    which uses the OpenAI-compatible response_format spec. Both Token Factory
    and vLLM enforce this server-side.
    """
    ok: bool = Field(
        description="True if the execution result plausibly answers the question."
    )
    issue: str = Field(
        default="",
        description="One-sentence explanation when ok=false. Empty string when ok=true."
    )


class SqlWithAssessment(BaseModel):
    """Merged generate+verify output: the SQL plus the model's self-assessment.

    Used to collapse the (generate -> execute -> verify) round-trip from 2 LLM
    calls down to 1 on the happy path. The model produces the SQL AND predicts
    whether the result would plausibly answer the question, BEFORE executing.
    Post-execution checks (was there an actual error?) are still done locally
    in verify_node without an LLM call.
    """
    sql: str = Field(
        description=(
            "A single valid SQLite query, raw (no markdown fences, no prose, "
            "no trailing semicolon-only lines)."
        )
    )
    ok: bool = Field(
        description=(
            "Pre-execution self-assessment: true if the model believes this "
            "SQL will plausibly answer the question given the schema."
        )
    )
    issue: str = Field(
        default="",
        description=(
            "One-sentence explanation when ok=false (which rubric check the "
            "SQL likely fails). Empty string when ok=true."
        )
    )


def llm() -> ChatOpenAI:
    """Chat client pointed at VLLM_BASE_URL (your local vLLM by default)."""
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
        max_tokens=512,
    )


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


def generate_sql_node(state: AgentState) -> dict:
    """Generate SQL AND self-assess in a single LLM call.

    Phase 6 / iter3: previously this was two LLM calls (generate, then verify).
    We now use structured output to get {sql, ok, issue} back in one shot,
    halving vLLM round-trips per /answer. The post-execution sanity check
    (did SQLite actually return an error?) still runs in verify_node, but
    without an LLM call.
    """
    structured = llm().with_structured_output(SqlWithAssessment, method="json_schema")
    decision: SqlWithAssessment = structured.invoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            question=state.question,
        )),
    ])
    return {
        "sql": decision.sql,
        "verify_ok": decision.ok,
        "verify_issue": decision.issue,
        "iteration": state.iteration + 1,
        "history": state.history + [{
            "node": "generate_sql",
            "sql": decision.sql,
            "self_ok": decision.ok,
            "self_issue": decision.issue,
        }],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


# Phase 6 / iter4 — cheap deterministic post-execution heuristics.
# Keyword-based, no LLM. The point is to catch the most common quality
# failures (empty-when-expected, suspicious-cardinality) that iter3's pre-
# execution self-assessment misses, AND to feed revise concrete evidence
# ("0 rows but expected results") that the model can actually act on.
_NONEMPTY_QUESTION_HINTS = (
    "top ", "highest", "lowest", "most ", "least ", "best ", "worst ",
    "oldest", "newest", "largest", "smallest", "biggest",
    "list ", "name the", "name all", "find the", "find all",
    "which ", "who ", "what is the", "what are the",
    "all the", "every ",
)
_SINGLE_ROW_QUESTION_HINTS = (
    "the highest", "the lowest", "the most", "the least",
    "the best", "the worst", "the oldest", "the newest",
    "the largest", "the smallest", "the biggest",
    "who is the", "which is the", "what is the",
)


def _question_expects_nonempty(question: str) -> bool:
    q = question.lower()
    return any(h in q for h in _NONEMPTY_QUESTION_HINTS)


def _question_expects_single_row(question: str) -> bool:
    q = question.lower()
    return any(h in q for h in _SINGLE_ROW_QUESTION_HINTS)


def verify_node(state: AgentState) -> dict:
    """LLM-free post-execution gate.

    Phase 6 evolution:
    - iter3: collapsed generate+verify into one LLM call. verify_node became
      a pass-through of the model's pre-execution self-assessment, with an
      override only when SQLite returned an error.
    - iter4 (this version): adds two deterministic post-execution heuristics
      to restore the loop's value without re-introducing an LLM call. These
      run AFTER the exec-error gate but BEFORE the self-assessment pass-
      through, so they can override an overly-optimistic self-assessment
      with concrete, evidence-grounded feedback for revise.

    Override priority:
      1. Execution error (SQLite raised)            -> revise
      2. EMPTY-WHEN-EXPECTED heuristic              -> revise
      3. SUSPICIOUS-CARDINALITY heuristic           -> revise
      4. Pass-through model self-assessment         -> end or revise
    """
    exec_result = state.execution

    if exec_result is None or not exec_result.ok:
        issue = exec_result.error if exec_result else "no execution result"
        return {
            "verify_ok": False,
            "verify_issue": issue,
            "history": state.history
            + [{"node": "verify", "ok": False, "issue": issue, "source": "execution_error"}]
        }

    if exec_result.row_count == 0 and _question_expects_nonempty(state.question):
        issue = (
            "EMPTY-WHEN-EXPECTED: the query returned 0 rows but the question "
            "implies non-empty results."
        )
        return {
            "verify_ok": False,
            "verify_issue": issue,
            "history": state.history
            + [{"node": "verify", "ok": False, "issue": issue, "source": "post_exec_heuristic"}]
        }

    if exec_result.row_count > 1 and _question_expects_single_row(state.question):
        issue = (
            f"SUSPICIOUS-CARDINALITY: the query returned {exec_result.row_count} rows "
            "but the question implies a single answer; consider adding LIMIT 1 "
            "or an appropriate aggregation."
        )
        return {
            "verify_ok": False,
            "verify_issue": issue,
            "history": state.history
            + [{"node": "verify", "ok": False, "issue": issue, "source": "post_exec_heuristic"}]
        }

    return {
        "history": state.history
        + [{
            "node": "verify",
            "ok": state.verify_ok,
            "issue": state.verify_issue,
            "source": "self_assessment",
        }]
    }


def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL + self-assessment in a single LLM call.

    Phase 6 / iter3: same structured-output pattern as generate_sql_node so the
    next verify_node call is LLM-free. previous_sql/previous_result/issue are
    fed in so the model can fix the prior mistake.
    """
    exec_result = state.execution

    structured = llm().with_structured_output(SqlWithAssessment, method="json_schema")
    decision: SqlWithAssessment = structured.invoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            previous_sql=state.sql,
            previous_result=exec_result.render() if exec_result else "no result",
            issue=state.verify_issue,
        )),
    ])
    return {
        "sql": decision.sql,
        "verify_ok": decision.ok,
        "verify_issue": decision.issue,
        "iteration": state.iteration + 1,
        "history": state.history
        + [{
            "node": "revise",
            "sql": decision.sql,
            "addressed_issue": state.verify_issue,
            "self_ok": decision.ok,
            "self_issue": decision.issue,
        }]
    }


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    Two reasons to end: the verifier was happy (state.verify_ok), or you've hit
    the iteration cap (state.iteration >= MAX_ITERATIONS). Otherwise, revise.
    """
    if state.verify_ok:
        return "end"
    if state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
