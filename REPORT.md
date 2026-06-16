# Phase 6 — SLO Tuning Log

**SLO target:** p95 end-to-end `/answer` latency < 5 s, sustained ≥ 10 RPS over a 5-minute window.

## Iteration 0 — Baseline

**Config:** `--dtype bfloat16 --max-model-len 8192 --max-num-seq 128 --gpu-memory-utilization 0.90 --guided-decoding-backend xgrammar --enable-prefix-caching --enable-chunked-prefill`. Agent has no concurrency cap; uses default OpenAI-client timeout to vLLM.

**Load test:** `uv run python load_test/driver.py --rps 10 --duration 300 --out results/load_test_iter0_baseline.json`

### Saw (X)
- **Success rate collapsed:** 573 / 3000 = **19 %**. Of the 2 427 failures: 1 323 client-side timeouts (driver waited 120 s and gave up), 326 HTTP errors, 778 client errors. Achieved RPS 8.3 vs requested 10.
- **p95 e2e latency = 118 s** (max 120 s = the driver's own cap). p50 = 57 s. So the SLO (5 s) was missed by ~24× at the median and ~24× at p95.
- **Agent (`/answer`) returned a wall of HTTP 500s** while vLLM's own log showed every chat-completion finishing with 200. The mismatch means failures originate above vLLM — the agent's HTTP client to vLLM timed out, raised an exception, FastAPI returned 500, and *then* vLLM eventually finished the call and logged its 200 long after the agent gave up.
- **Grafana told us where the time went:**
  - `scheduler: running` saturated at the cap of 128 concurrent sequences with a non-empty `waiting` queue → vLLM was fully loaded but accepting more than it could drain.
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

## Iteration 1 — *(planned)* reduce decode batch
*To fill in after the run.*
