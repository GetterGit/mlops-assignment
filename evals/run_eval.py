"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from agent.graph import MAX_ITERATIONS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------


def _error_record(item: dict, error: str, latency_s: float) -> dict:
    """Skeleton record for a question whose agent call never completed.
    Record-and-continue semantics: a single failure must not poison the run,
    and the per-question fields stay shape-compatible with successful records
    so summarize() doesn't need special-casing.
    """
    return {
        "db_id": item["db_id"],
        "question": item["question"],
        "gold_sql": item["gold_sql"],
        "completed": False,
        "error": error,
        "iterations_run": 0,
        "hit_max_iterations": False,
        "final_sql": "",
        "per_iter_sql": [],
        "per_iter_exec_match": [],
        "final_exec_match": False,
        "latency_s": round(latency_s, 3),
    }


async def eval_one(
    client: httpx.AsyncClient,
    item: dict,
    agent_url: str,
    run_id: str,
    idx: int
) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = item["db_id"]
    question = item["question"]
    gold_sql = item["gold_sql"]

    tags = {
        "phase": "baseline",
        "eval_run_id": run_id,
        "db_id": db_id,
        "question_idx": str(idx),
    }

    t0 = time.monotonic()
    try:
        resp = await client.post(
            agent_url,
            json={"db": db_id, "question": question, "tags": tags},
            timeout=180.0
        )
    except Exception as e:
        return _error_record(item, f"{type(e).__name__}: {e}", time.monotonic() - t0)

    latency_s = time.monotonic() - t0

    if resp.status_code != 200:
        return _error_record(item, f"HTTP {resp.status_code}: {resp.text[:200]}", latency_s)
    
    body = resp.json()
    final_sql: str = body.get("sql", "")
    history: list[dict] = body.get("history", [])
    iterations_run: int = int(body.get("iterations", 0))

    per_iter_sql: list[str] = [
        entry["sql"]
        for entry in history
        if entry.get("node") in ("generate_sql", "revise")
        and isinstance(entry.get("sql"), str)
    ]

    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)
    if not gold_ok:
        # Unscoreable - gold SQL itself is broken for this DB. Flag loudly,
        # mark every attempt as no-match so this question can't inflate accuracy.
        print(f"  WARN: gold SQL failed for {db_id}: {gold_err}", flush=True)
        per_iter_exec_match: list[bool] = [False] * len(per_iter_sql)
    else:
        per_iter_exec_match = []
        for sql in per_iter_sql:
            pred_ok, pred_rows, _ = run_sql(db_id, sql)
            per_iter_exec_match.append(pred_ok and matches(gold_rows, pred_rows))

    # hit_max_iterations: ran the full budget AND the verifier never said ok.
    # (If verify said ok at iter k < MAX, the agent exited gracefully; that's
    # not "hit the cap", so we exclude it here even though iterations_run
    # might equal MAX in some edge cases.)
    last_verify = next((e for e in reversed(history) if e.get("node") == "verify"), None)
    verify_ok_at_end = bool(last_verify and last_verify.get("ok"))
    hit_max_iterations = (iterations_run >= MAX_ITERATIONS) and (not verify_ok_at_end)

    return {
        "db_id": db_id,
        "question": question,
        "gold_sql": gold_sql,
        "completed": True,
        "iterations_run": iterations_run,
        "hit_max_iterations": hit_max_iterations,
        "final_sql": final_sql,
        "per_iter_sql": per_iter_sql,
        "per_iter_exec_match": per_iter_exec_match,
        "final_exec_match": per_iter_exec_match[-1] if per_iter_exec_match else False,
        "latency_s": round(latency_s, 3),
    }


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolated percentile via stdlib statistics.quantiles.

    Falls back to nearest-rank when n<2 (quantiles raises StatisticsError).
    p is a float in [0, 1].
    """
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 3)
    cuts = statistics.quantiles(sorted_vals, n=100, method="inclusive")
    return round(cuts[int(p * 100) - 1], 3)


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    if n == 0:
        return {"n": 0}

    completed = [r for r in results if r["completed"]]

    # Carry-forward: pad each per_iter_exec_match list to MAX_ITERATIONS by
    # repeating its last value (or [False] for records that never emitted).
    extended: list[list[bool]] = []
    for r in results:
        per_iter = list(r.get("per_iter_exec_match") or [False])
        if len(per_iter) < MAX_ITERATIONS:
            per_iter = per_iter + [per_iter[-1]] * (MAX_ITERATIONS - len(per_iter))
        extended.append(per_iter[:MAX_ITERATIONS])

    exec_match_at_k = {
        k + 1: round(sum(em[k] for em in extended) / n, 4)
        for k in range(MAX_ITERATIONS)
    }

    final_exec_match_rate = round(sum(r["final_exec_match"] for r in results) / n, 4)
    completion_rate = round(len(completed) / n, 4)
    hit_max_rate = round(sum(r.get("hit_max_iterations", False) for r in results) / n, 4)

    iter_counts = [r["iterations_run"] for r in completed]
    latencies = sorted(r["latency_s"] for r in completed)

    mean_iterations = round(statistics.fmean(iter_counts), 2) if iter_counts else 0.0
    mean_latency_s = round(statistics.fmean(latencies), 3) if latencies else 0.0
    p50_latency_s = _percentile(latencies, 0.50)
    p95_latency_s = _percentile(latencies, 0.95)

    by_db: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_db[r["db_id"]].append(r)

    by_db_id: dict[str, dict] = {}
    for db_id, rows in sorted(by_db.items()):
        n_db = len(rows)
        completed_db = [r for r in rows if r["completed"]]
        by_db_id[db_id] = {
            "n": n_db,
            "final_exec_match_rate": round(sum(r["final_exec_match"] for r in rows) / n_db, 4),
            "mean_iterations": round(statistics.fmean([r["iterations_run"] for r in completed_db]), 2)
            if completed_db else 0.0,
            "mean_latency_s": round(statistics.fmean([r["latency_s"] for r in completed_db]), 3)
            if completed_db else 0.0,
            "hit_max_rate": round(sum(r.get("hit_max_iterations", False) for r in rows) / n_db, 4),
        }

    return {
        "n": n,
        "completion_rate": completion_rate,
        "final_exec_match_rate": final_exec_match_rate,
        "exec_match_at_k": exec_match_at_k,
        "hit_max_rate": hit_max_rate,
        "mean_iterations": mean_iterations,
        "mean_latency_s": mean_latency_s,
        "p50_latency_s": p50_latency_s,
        "p95_latency_s": p95_latency_s,
        "by_db_id": by_db_id,
    }


def _print_summary(summary: dict, elapsed: float, run_id: str) -> None:
    """Compact human-readable rollup printed after the JSON dump."""
    print(f"\n=== Eval summary (run_id={run_id}) ===")
    print(f"  n={summary['n']}   completion_rate={summary['completion_rate']:.1%}   "
          f"wall_clock={elapsed:.1f}s")
    print(f"  final_exec_match_rate={summary['final_exec_match_rate']:.1%}   "
          f"hit_max_rate={summary['hit_max_rate']:.1%}   "
          f"mean_iterations={summary['mean_iterations']}")
    print("  exec_match_at_k=" + "  ".join(
        f"k{k}={v:.1%}" for k, v in summary["exec_match_at_k"].items()
    ))
    print(f"  latency: mean={summary['mean_latency_s']}s  "
          f"p50={summary['p50_latency_s']}s  p95={summary['p95_latency_s']}s")
    print("  by_db_id:")
    for db_id, stats in summary["by_db_id"].items():
        print(f"    {db_id:<22} n={stats['n']:<2} "
              f"match={stats['final_exec_match_rate']:.1%}  "
              f"iters={stats['mean_iterations']}  "
              f"lat={stats['mean_latency_s']}s  "
              f"hitmax={stats['hit_max_rate']:.1%}")


# ---------- Main --------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of in-flight /answer requests.")
    parser.add_argument("--run-id",
                        default=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S"),
                        help="Used as the 'eval_run_id' Langfuse tag.")
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")
    print(f"Run ID: {args.run_id}   Concurrency: {args.concurrency}   Agent: {args.agent_url}")

    async def run_all() -> list[dict]:
        sem = asyncio.Semaphore(args.concurrency)
        results: list[dict | None] = [None] * len(questions)

        async def run_one(i: int, q: dict, client: httpx.AsyncClient) -> None:
            async with sem:
                r = await eval_one(client, q, args.agent_url, args.run_id, i)
            status = "OK " if r.get("completed") else "ERR"
            match = r.get("final_exec_match")
            iters = r.get("iterations_run")
            lat = r.get("latency_s", 0.0)
            print(
                f"[{i + 1:>2}/{len(questions)}] {status} "
                f"match={str(match):<5} iters={iters} {lat:5.1f}s "
                f"{q['db_id']:<22} {q['question'][:50]}",
                flush=True,
            )
            results[i] = r

        async with httpx.AsyncClient() as client:
            await asyncio.gather(*(run_one(i, q, client) for i, q in enumerate(questions)))

        return [r for r in results if r is not None]

    t0 = time.monotonic()
    results = asyncio.run(run_all())
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "run_id": args.run_id,
        "summary": summary,
        "wall_clock_seconds": round(elapsed, 2),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")
    _print_summary(summary, elapsed, args.run_id)


if __name__ == "__main__":
    main()
