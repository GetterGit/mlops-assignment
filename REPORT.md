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

## Iteration 2 — *(planned)* cap LLM output tokens at 512
*To fill in after the run.*
