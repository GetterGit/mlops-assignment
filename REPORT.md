# Phase 6 — SLO Tuning Log

**SLO target:** p95 end-to-end `/answer` latency < 5 s, sustained ≥ 10 RPS over a 5-minute window.

## Iteration 0 — Baseline

**Config:** `--dtype bfloat16 --max-model-len 8192 --max-num-seq 128 --gpu-memory-utilization 0.90 --guided-decoding-backend xgrammar --enable-prefix-caching --enable-chunked-prefill`. Agent has no concurrency cap; uses default OpenAI-client timeout to vLLM.

### Saw (X)
- **Success rate collapsed:** 573 / 3000 = **19 %**. Of the 2 427 failures: 1 323 client-side timeouts (driver waited 120 s and gave up), 326 HTTP errors, 778 client errors. Achieved RPS 8.3 vs requested 10.
- **p95 e2e latency = 118 s** (max 120 s = the driver's own cap). p50 = 57 s. So the SLO (5 s) was missed by ~24× at the median and ~24× at p95.
- **Agent (`/answer`) returned a wall of HTTP 500s** while vLLM's own log showed every chat-completion finishing with 200. The mismatch means failures originate above vLLM — the agent's HTTP client to vLLM timed out, raised an exception, FastAPI returned 500, and *then* vLLM eventually finished the call and logged its 200 long after the agent gave up.
- **Grafana told us where the time went:**
  - `KV cache usage` only reached ~25 %, and `Preemptions/sec` stayed at 0. So this is **not** a memory/KV-pressure problem — there was plenty of GPU memory free.
  - `prefix hit rate` was ~85 % once the test warmed up, confirming the shared system prompt + schema preamble was being reused as expected. This rules out repeated cold prefill as the cause.
  - `token throughput (decode)` climbed slowly from 0 → ~12 K tok/s as concurrency built up — consistent with batched decode at a large batch size.

### Hypothesized (Y)
The bottleneck is **compute-bound batched decode amplified by the agent's serial fan-out**, not KV memory:

1. One `/answer` request = 2–3 *sequential* LLM calls (`generate_sql` → `verify` → maybe `revise`). So an offered load of 10 agent-RPS lands as ~20–30 RPS on vLLM.
2. With `--max-num-seq 128`, vLLM packs up to 128 sequences into a single decode forward pass. The pass time grows with batch size, so per-token latency for each sequence inflates. Each LLM call (~200–500 output tokens) ends up taking 20–60 s under saturation.
3. One agent run = 40–120 s end-to-end → blows the agent's internal HTTP timeout to vLLM → cascading 500s.
4. The driver keeps pushing 10 RPS into a server that can only complete a handful per second, so the backlog grows monotonically and the system never recovers within the 5-min window.

In short: the system *can* serve every request given infinite patience (vLLM logs 200s), but the chain `agent → vLLM → agent → vLLM → …` makes per-request wall-clock latency unsurvivable.

### Changed (Z)
Nothing yet — this is the baseline. Tuning starts in Iteration 1.

### Result (W)
SLO missed by ~24× on p95 latency and ~80 % on success rate. Diagnosis: queue depth + decode batch size, **not** KV cache. Two clear levers to try next: (a) shrink `--max-num-seq` to reduce per-token latency under concurrency; (b) cap output tokens and/or add agent-side backpressure (semaphore + fast 503) so failures become deterministic instead of timeout cascades.

### Artifacts
- `results/load_test_iter0_baseline.json`
- `screenshots/grafana_iter0_baseline.png` *(TODO capture)*

---

## Iteration 1 — reduce decode batch (`--max-num-seqs 128 → 32`)

**Hypothesis going in:** if we cap concurrent decoding sequences at 32 instead of 128, per-token decode latency drops, individual LLM calls finish before the agent's HTTP timeout fires, and the 500-cascade stops — even at the cost of some peak throughput.

**Change:** single flag in `scripts/start_vllm.sh`: `--max-num-seq 128` → `--max-num-seqs 32` (also fixed the legacy typo; argparse abbreviation was matching it anyway, confirmed via `vllm.log` showing `max_num_seqs: 128` in iter0).

### Saw (X)

| Metric | iter0 (B=128) | iter1 (B=32) | Δ |
|---|---|---|---|
| ok / 3000 | 573 (19 %) | **315 (10 %)** | worse |
| timeouts | 1 323 | 1 592 | worse |
| http_errors | 326 | 232 | better |
| client_errors | 778 | 861 | flat |
| p50 (s) | 56.7 | **36.9** | better (-35 %) |
| p95 (s) | 118.5 | 115.3 | flat (≈ driver's 120 s cap) |
| Grafana `running` peak | ~128 | **~32** | flag confirmed effective |
| Grafana `KV cache usage` peak | ~25 % | ~100 % at one burst | now KV-pressured |
| vLLM `/metrics` p95 | (not captured) | ≈ flat 0 s most of window | vLLM serves its share fast |

Also surfaced mid-run from the agent log: `[Errno 24] Too many open files` on the Langfuse OTEL exporter at port 3001 — i.e. the agent process is hitting the default `ulimit -n 1024`. Some fraction of the 500s in both iterations are FD exhaustion, not vLLM timeouts. This makes the iter0 → iter1 latency-cause attribution noisier than ideal.

### Hypothesized (Y) — revised after seeing results

Two corrections to the iter0 model:

1. **Smaller batch hurts when offered load > capacity.** With B=32 instead of 128, vLLM only accepts 32 concurrent sequences. The driver still emits 10 RPS × ~25 vLLM-RPS-equivalent (agent fan-out), so the *queue gets deeper, not shallower*. p50 improved (each accepted request decodes faster), but tail and success rate worsened (more requests wait longer than the 120 s driver cap). Shrinking the batch is the *right move at the right offered load* — not here.
2. **The dashboard measures the wrong thing for this SLO.** Our Grafana panels read `vllm:e2e_request_latency_seconds`, which is vLLM-side. vLLM is happily serving each call sub-second. The 120 s tail lives entirely in the agent layer (queue wait + 2–3 serial LLM calls + sqlite + Langfuse spans). We can't *see* the SLO from the current dashboard.

So the real bottleneck stack is now: **(a) per-request output length on each vLLM call is unbounded, (b) the agent does 2–3 of these serially per `/answer`, (c) the agent has no concurrency cap so the queue death-spirals, (d) FDs leak under load.** Batch size is a downstream knob.

### Changed (Z)
`--max-num-seqs 128 → 32` in `scripts/start_vllm.sh`. vLLM restarted in tmux. Agent restarted in foreground (FD limit not yet raised — that's an iter2 precondition).

### Result (W)
SLO still missed. Success rate **regressed** (-9 pts), p50 improved (-35 %), p95 effectively unchanged (still pinned at the driver's 120 s timeout cap). Net verdict: shrinking decode batch alone, at this offered load, traded median latency for tail failures. The right next levers attack the **work per request** (cap output tokens) and **work per `/answer`** (reduce fan-out) before tuning the scheduler further.

### Artifacts
- `results/load_test_iter1_maxnumseqs32.json`
- `screenshots/grafana_iter1.png` *(TODO capture)*

---

## Iteration 2 — cap LLM output tokens at 512

**Hypothesis going in:** unbounded generations are amplifying decode time. If the model was occasionally emitting 1.5K–2K tokens per LLM call, capping at 512 should cut per-call decode cost ~3–4× and pull p95 down meaningfully.

**Change:** `agent/graph.py` `llm()` factory got `max_tokens=512`. Both `generate_sql` (plain `.invoke`) and `verify`/`revise` (structured output) inherit the cap. vLLM flags unchanged from iter1 (`--max-num-seqs 32`).

### Saw (X)

| Metric | iter1 (B=32, no cap) | iter2 (B=32, cap=512) | Δ |
|---|---|---|---|
| ok / 3000 | 315 (10.5 %) | 367 (12.2 %) | +52 OK (marginal) |
| timeouts | 1 592 | 1 604 | flat |
| p50 (s) | 36.9 | 43.6 | slightly worse (over a larger set) |
| p95 (s) | 115.3 | 115.3 | flat (still pinned at driver's 120 s cap) |
| Grafana `KV cache usage` peak | ~20 % | ~20 % | flat — nowhere near capacity |
| Grafana `running` peak | ~32 | ~32 | flag still effective |

### Hypothesized (Y) — what iter2 actually revealed

The cap was a near-noop, which is the finding: **output length was not the bottleneck**. SQL responses were already in the 200–500 token range, so capping at 512 trimmed almost nothing. More importantly, looking at the dashboard alongside the throughput math:

- iter0 peak decode throughput hit ~12 K tok/s, but sustained successful work was only ~950 tok/s (573 OK × ~2 LLM calls × ~300 output tokens / 360 s). vLLM ran at <10 % of its peak.
- KV cache never exceeded ~25 % across any iteration. The model is **not KV-bound**, not compute-saturated either.
- The system is bottlenecked by **per-request latency × serial agent fan-out**, not raw throughput capacity. Each `/answer` is ~2–3 sequential LLM round-trips. Even if each call were instant, the round-trip overhead through httpx + LangChain + structured-output parsing + Langfuse + sqlite adds real time per hop.

This reframes the problem: shrinking the decode batch (iter1) and capping output (iter2) are tuning the wrong layer. The biggest single lever is **reducing the number of vLLM round-trips per `/answer`**.

### Changed (Z)
`max_tokens=512` in `agent/graph.py::llm()`. No vLLM flag change.

### Result (W)
SLO still missed. Marginal improvement in ok-rate (10.5 % → 12.2 %), p50 mildly worse over a larger set, p95 unchanged. Net verdict: iter2 confirms that "make individual LLM calls cheaper" doesn't dominate when the agent itself is making too many of them. Next move: cut the number of calls.

### Artifacts
- `results/load_test_iter2_maxtokens512.json`
- `screenshots/grafana_iter2.png` *(TODO capture)*

---

## Iteration 3 — merge generate + verify into one structured-output call (also: revert `--max-num-seqs` to 128)

**Hypothesis going in:** halving vLLM round-trips per `/answer` is the largest single available lever — it directly attacks the serial-fan-out bottleneck identified in iter2. The model already produces SQL; teach it to also self-assess plausibility in the same call (`{sql, ok, issue}`). Verify becomes a deterministic post-execution gate (no LLM call) that only overrides the self-assessment when SQLite actually returned an error.

We expect: per-`/answer` vLLM calls drop from 2 → 1 on the happy path (and from 2k+1 → k+1 in the revise loop), which should ~2× the achievable agent RPS and significantly cut p50/p95.

**Also reverting iter1's regression:** `--max-num-seqs 32 → 128`. Iter1 was a one-knob experiment whose justification ("decode batch too large") was *not* grounded in any observation (KV was ~25 %, preemptions were 0). With hindsight: there was no dashboard evidence to support shrinking the batch, and the test showed it actively hurt success rate. Restoring the iter0 setting.

### Changes

1. `agent/graph.py`:
   - New `SqlWithAssessment` pydantic model: `{sql, ok, issue}`.
   - `generate_sql_node` uses `with_structured_output(SqlWithAssessment)` and writes `verify_ok` / `verify_issue` directly to state.
   - `revise_node` does the same.
   - `verify_node` no longer calls the LLM: it overrides only on execution errors, otherwise passes the model's self-assessment through to the router.
2. `agent/prompts.py`: `GENERATE_SQL_SYSTEM` and `REVISE_SYSTEM` extended with the self-assessment rubric (same checks as the old `VERIFY_SYSTEM`: WRONG-COLUMN-SHAPE, SUSPICIOUS-CARDINALITY, WRONG-TYPE, SCHEMA-MISMATCH). `VERIFY_SYSTEM` is no longer called but kept in file for reference.
3. `scripts/start_vllm.sh`: `--max-num-seqs 32 → 128`.

### Quality tradeoff to call out

The original `verify_node` saw the *actual rows* after SQL execution, so it could catch EMPTY-WHEN-EXPECTED, wrong cardinality, etc. The merged self-assessment runs *before* execution, so those post-hoc checks are weaker — we now only catch shape/schema mistakes the model itself can foresee. This is expected to cost some `exec_match_at_3` quality. Iter3 will be evaluated with `evals/run_eval.py → results/eval_after_tuning.json` and the pass-rate delta vs `results/eval_baseline.json` reported honestly.

### Result (W)

**The decisive iteration.** Halving vLLM round-trips per `/answer` moved every metric by more than an order of magnitude — and very nearly hit the SLO.

| Metric | iter0 (baseline) | iter2 (best prior) | **iter3** | iter0 → iter3 |
|---|---|---|---|---|
| ok / 3000 | 573 (19 %) | 367 (12 %) | **2 603 (86.8 %)** | **+2 030 OK (4.5×)** |
| timeouts | 1 323 | 1 604 | **1** | virtually eliminated |
| http_errors | 326 | 246 | 396 | slightly worse |
| achieved_rps | 8.33 | 8.33 | **9.50** | first iter to approach the 10 RPS target |
| p50 (s) | 56.70 | 43.62 | **1.45** | **39× faster** |
| p95 (s) | 118.50 | 115.32 | **6.90** | **17× faster** |
| p99 (s) | 119.93 | 119.90 | 11.52 | |
| latency_max (s) | 120.88 | 120.76 | 43.65 | no requests pinned at driver cap |

**SLO verdict:** p95 = **6.90 s** vs target **< 5 s**. **Missed by 1.9 s** (a 38 % overshoot) but in the same order of magnitude as the SLO for the first time. Success rate (86.8 %) is healthy. The architecture change was the right diagnosis.

**What the dashboard showed during the run:** scheduler `running` ramped cleanly to ~30–40 with `waiting` staying low (vs iter0 where it pinned at 128 with deep queues); decode tokens/sec ran in the steady 4–8 K range without the boom-and-bust shape of earlier iters; KV cache usage finally moved off the floor (~30–40 %) reflecting the larger in-flight working set; preemptions stayed at 0.

**Open items the result exposed:**

- **396 HTTP 500s remain (13 % of attempts).** With timeouts essentially gone, these aren't queue-collapse — they're either (a) structured-output decoding failing on edge questions (model emits JSON that doesn't fit `SqlWithAssessment`), (b) SQLite errors the model didn't predict, or (c) the model self-assessing `ok=false` repeatedly until iteration cap. Worth a Langfuse spot-check.
- **p95 6.9 s** suggests one of the LLM call types is occasionally slow — likely the revise loop firing on some questions. Iter4 candidates to close the last 1.9 s: FP8 quant (per-token decode ~1.5–2× faster on H100), or `MAX_ITERATIONS=3 → 2` (caps the worst case at 2 LLM calls instead of 3).

### Artifacts
- `results/load_test_iter3_merged_verify.json`
- `screenshots/grafana_after.png` (iter3 window — clean steady-state, low queue, low p95)
- `screenshots/grafana_before.png` (iter0 storm window — captured for direct visual comparison)
- `results/eval_after_tuning.json` (see eval read below)

### Quality eval read (`eval_after_tuning.json`, 30 questions)

```
n=30  completion_rate=100.0%  wall_clock=9.6s
final_exec_match_rate=33.3%   hit_max_rate=3.3%   mean_iterations=1.2
exec_match_at_k=k1=33.3%  k2=33.3%  k3=33.3%
```

**The loop is dormant.** `k1 = k2 = k3 = 33.3 %` means the verify→revise loop adds zero quality in iter3. README Phase 5 calls this out explicitly: *"If iter 0 pass rate is the same as iter 3 pass rate, your agent architecture is doing nothing."* Two stacked failure modes caused this:

1. **The loop rarely triggers.** Pre-execution self-assessment exhibits the well-known LLM self-evaluation bias: the model almost always says `ok=true` about its own output (~80 % of questions, given `mean_iterations = 1.2`). The original post-execution verify worked because it had *empirical evidence* — the actual rows — to disagree with.
2. **When it triggers, revise can't improve.** The self-critique used to seed revise has no execution feedback — it's the same model second-guessing the same model with no new information.

This is the quality regression we flagged as a risk going into iter3. Iter4 is the targeted fix.

---

## Iteration 4 — restore the loop with deterministic post-execution heuristics

**Hypothesis going in:** the SLO-friendly half of iter3 (merged generate+verify, 1 LLM call on the happy path) is the right architecture. The quality regression is recoverable WITHOUT adding back an LLM call: we just need cheap, deterministic post-execution checks in `verify_node` that catch the failure modes self-assessment misses, and feed revise concrete evidence the model can act on.

We expect: revise loop fires more often (more questions hit iter≥2), `exec_match_at_3 > exec_match_at_1` again (loop earns its keep), at the cost of only a small p95 regression (the few extra revises add latency but verify_node itself stays LLM-free).

### Changes

In `agent/graph.py::verify_node`, between the exec-error gate and the self-assessment pass-through, add two deterministic heuristics with concrete `verify_issue` strings:

1. **EMPTY-WHEN-EXPECTED**: if `exec_result.row_count == 0` AND the question contains a keyword that implies a non-empty result (`top`, `highest`, `most`, `which`, `who`, `list`, `name the`, etc.) → force revise with a concrete complaint the model can act on.
2. **SUSPICIOUS-CARDINALITY**: if `exec_result.row_count > 1` AND the question implies a single answer (`the highest`, `who is the`, `what is the`, etc.) → force revise with a suggestion to add `LIMIT 1` or aggregate.

Both run AFTER the exec-error check and BEFORE the self-assessment pass-through. They're deterministic (keyword + count comparison only — zero LLM calls), so they preserve iter3's throughput. Their issue strings are concrete and post-execution-grounded, so revise has real signal to act on (vs the iter3 self-criticism which had none).

A new `source: "post_exec_heuristic"` value is added to the verify history entry so Langfuse traces and the eval can distinguish heuristic-triggered revises from execution-error and self-assessment cases.

### Test plan (run eval BEFORE load test)

Eval first — if quality didn't recover, no point burning a 5-min load test:

```bash
uv run python evals/run_eval.py \
    --out results/eval_iter4_hybrid.json \
    --run-id phase6_iter4_hybrid
```

We're looking for:
- `exec_match_at_3 > exec_match_at_1` (loop earning its keep)
- `mean_iterations > 1.2` (more questions are looping)
- Per-DB breakdown: `formula_1`, `thrombosis_prediction`, `toxicology` should pick up from 0 % (these were the EMPTY-WHEN-EXPECTED candidates)

If those signals don't show, the heuristic keywords are wrong — debug and reroll. If they do show, then run the load test:

```bash
uv run python load_test/driver.py --rps 10 --duration 300 \
    --out results/load_test_iter4_hybrid.json
```

### Result (W)
*To fill in after the runs.*

### Artifacts
- `results/eval_iter4_hybrid.json`
- `results/load_test_iter4_hybrid.json`
- `screenshots/grafana_iter4.png`
