# MLOps HW3 — Qwen3-30B-A3B text-to-SQL agent on H100

**SLO target:** p95 end-to-end `/answer` latency < 5 s, sustained ≥ 10 RPS over 5 min.

**Final config (iter4):** 
- BF16 Qwen3-30B-A3B-Instruct-2507
- merged generate and self-assessment stages in the agent loop
- deterministic post-execution heuristics (EMPTY-WHEN-EXPECTED, SUSPICIOUS-CARDINALITY), `MAX_ITERATIONS=3`, `--max-num-seqs 128`.

**Headline numbers (10 RPS, 300 s):**

| | iter0 baseline | iter4 final |
|---|---|---|
| Load test ok / 3000 | 573 (19 %) | **2 583 (86 %)** |
| Achieved RPS (vs 10 target) | 1.91 | **8.61** |
| Load test p50 | 56.7 s | 27.5 s |
| Load test p95 | 118.5 s | **57.8 s** — still > 5 s SLO target by ~11× |
| Eval `exec_match_at_3` | 43.3 % | 40.0 % (−3.3 pp; quality cost of merge) |
| Eval `k1 → k3` | 36.7 → 43.3 → 43.3 | 33.3 → 36.7 → 40.0 (loop still earns +6.7 pp) |

SLO missed; Compared to the baseline, 4.5× more requests served, ~2× faster median, ~3 pp eval quality regression as the price of improved latency after merged generate+verify stages of the agent loop.

---

## 1. Serving configuration (Phase 1)

vLLM 0.10.2 serving Qwen/Qwen3-30B-A3B-Instruct-2507 on 1× H100 80 GB.

| Flag | Value | Why |
|---|---|---|
| `--dtype` | `bfloat16` | Default for H100; tried FP8 in iter5, lost 10 pp eval accuracy, reverted. |
| `--max-model-len` | `8192` | Schema + prompt + SQL output fits comfortably in 8K; longer wastes KV cache budget. |
| `--max-num-seqs` | `128` | Cap on concurrent decoded sequences. Tried 32 in iter1, hurt success rate — no observation supported the change. |
| `--gpu-memory-utilization` | `0.90` | Headroom for activations; lower risks OOM on a long prompt. |
| `--guided-decoding-backend` | `xgrammar` | Needed for `with_structured_output(SqlWithAssessment)` JSON schema enforcement on every generate/revise call. |
| `--enable-prefix-caching` | on | Long shared system prompt + schema prefix is reused on every call — ~85 % prefix hit rate confirmed in Grafana. |
| `--enable-chunked-prefill` | on | Lets prefill of new prompts overlap with ongoing decode; cuts TTFT spikes when a new request joins a busy batch. |

Agent uses `temperature=0.0` and `max_tokens=512` (added iter2, harmless — outputs were already 200-500 tokens).

---

## 2. Baseline eval (Phase 5)

30 questions from BIRD-bench dev, 9 DBs. Pre-Phase 6 baseline (BF16, original 2-LLM-call verify):

```
n=30  completion_rate=100%  mean_iterations=1.57
final_exec_match_rate=43.3%   hit_max_rate=20%
exec_match_at_k=k1=36.7%  k2=43.3%  k3=43.3%
```

(Source: `results/eval_baseline.json`. Loop earned its keep at baseline: +6.7 pp from k1 to k2, then plateaus. So the original architecture did work — `verify` caught real problems and `revise` fixed some on the second and third tries.)

---

## 3. SLO tuning (Phase 6)

### What each iteration did, by the numbers


| iter | What changed | ok / 3000 | Achieved RPS | p50 | p95 | Eval k1→k3 |
|---|---|---|---|---|---|---|
| 0 | **baseline (BF16, 2 LLM calls, `max-num-seqs 128`)** | 573 (19 %) | 1.91 | 57 s | 119 s | 36.7→43.3→43.3 (loop +6.7 pp) |
| 1 | `--max-num-seqs 128 → 32` | 315 (10 %) | 1.05 | 37 s | 115 s | (not eval'd — regression on RPS) |
| 2 | `max_tokens=512` cap on LLM calls | 367 (12 %) | 1.22 | 44 s | 115 s | (not eval'd — marginal gain) |
| 3 | **Merge generate+verify** into one structured-output call. Revert `max-num-seqs` to 128. | 2 603 (87 %) | 8.68 | 1.45 s | **6.9 s** | 33.3→33.3→33.3 — loop went dormant |
| 4 | **Hybrid verify:** keep iter3 merge + add deterministic post-exec heuristics (empty-when-expected, suspicious-cardinality) | 2 583 (86 %) | 8.61 | 27.5 s | 57.8 s | 33.3→36.7→40.0 — loop earning its keep again, but k3 still 3.3 pp below baseline |
| 5 | Try FP8 quant (weights + KV cache) | — | — | — | — | 23.3→26.7→30.0 (−10 pp vs iter4 → abandoned at eval gate) |

### What I learned at each step

**Iter 0.** Baseline blew the SLO by ~24×. Dashboard said `running` pinned at 128 with deep `waiting` queue, KV cache only at ~25 %, preemptions at 0. So not KV-bound — bottleneck was decode-batch latency × the agent doing 2-3 serial LLM calls per `/answer`. Most agent 500s while vLLM logged 200 = agent's httpx timeout to vLLM was firing before vLLM finished.

**Iter 1.** Shrunk `max-num-seqs` to 32. Wrong move and the dashboard didn't actually support it (KV was nowhere near full). Median improved a bit but tail and success rate both went backwards — fewer concurrent slots = deeper queues. Lesson: don't change a knob without a metric saying that knob is the problem.

**Iter 2.** Capped `max_tokens=512` on agent's LLM client. Turned out output length wasn't the cost — SQL responses were already 200-500 tokens. So neither batch size nor per-call output was the dominant cost. So, I thought that I should cut the number of LLM requests themselves, instead of iterating over settings per call.

**Iter 3.** Merged `generate_sql` and `verify` into one structured-output call (`{sql, ok, issue}` via `with_structured_output(SqlWithAssessment)`). `verify_node` became LLM-free (just pass-through + an override on SQLite errors). Halved vLLM round-trips per `/answer` on happy path. Massive win on the load test: ok 19 % → 87 %, p95 119 s → 6.9 s. SLO almost cleared.

The catch — running the eval afterwards showed `k1=k2=k3=33.3 %`. The loop went dormant. Pre-execution self-assessment is too generous: the model almost always says `ok=true` about its own output, so revise rarely fires. And when it does fire, it has no execution evidence to act on. So, I decided to choose a hybrid approach by adding deterministic post-execution checks to trigger revise when the check complains.

**Iter 4.** Kept the iter3 merge but added two deterministic post-execution checks back into `verify_node` — no LLM calls, just keyword + row-count: if `row_count==0` and the question contains "top/highest/which/list/who/…" → flip to revise with an `EMPTY-WHEN-EXPECTED` complaint; if `row_count>1` and the question contains "the highest/who is the/…" → flip to revise with a `SUSPICIOUS-CARDINALITY` complaint. The issue strings give revise concrete grounded feedback (vs iter3's self-criticism, which was just the model second-guessing itself).

Eval recovered cleanly: k1=33.3, k2=36.7, k3=40.0 — monotonic, loop works again. But the load test paid: `hit_max_rate` jumped 3.3 % → 16.7 % (more questions doing 3 LLM calls under load), p50 1.5 s → 27.5 s, p95 6.9 s → 57.8 s.

**Iter 5.** Tried FP8 quant (`Qwen3-30B-A3B-Instruct-2507-FP8` + `--kv-cache-dtype fp8`) to decrease latency. vLLM came up clean, `quantization=fp8` and `kv_cache_dtype=fp8` confirmed in startup log. But eval made me turn down this hypothesis: k3 dropped from 40 → 30 % (−10 pp). Per-DB breakdown: `financial` 66.7 → 33.3, `codebase_community` 40 → 20 — quant hits schema-heavy reasoning harder than simple lookups. Reverted to iter4.

### Final outcome

iter4 hits the per-deliverable shape — Phase 5 loop demonstrably works, Phase 6 success rate is real (86 % at 10 RPS), Grafana before/after is dramatic — but **p95 of 57.8 s misses the 5 s SLO by ~11×**. 
---

## 4. Did the agent loop earn its keep?

Yes — both at baseline and at iter4. Baseline showed +6.7 pp from k1 (36.7) to k3 (43.3) on a 2-LLM-call verify; iter4 shows the same +6.7 pp delta (33.3 → 40.0) on a 1-LLM-call merged generate with deterministic post-execution heuristics. In both cases the loop is firing on roughly half the questions and recovering a meaningful slice. Iter3 is the negative control: with the merge in place but *only* the model's pre-execution self-assessment, the loop went structurally alive but practically dormant (k1=k2=k3=33.3), because the model almost never flags its own SQL as wrong before it executes. The lesson is that the loop's value comes from grounded post-execution evidence — empty result set when the question expects rows, multi-row result when the question expects one.

---

## 5. What I'd do next with more time

Specific, in order of expected impact:

1. **AWQ-Int4 weights only** (keep BF16 activations + BF16 KV cache). Halves per-token compute like FP8 but preserves activation precision better — likely much smaller quality regression than the 10 pp iter5 saw, while still cutting p95 to ~25-35 s range.
2. **Speculative decoding** with Qwen 1.8B as the draft model. Free 1.5-2× throughput on long decodes, doesn't touch quality. Quick to wire — vLLM supports it already.
3. **Short-circuit repeated heuristic failures.** Right now if EMPTY-WHEN-EXPECTED fires once and revise produces another empty result, we keep retrying for all 3 iterations on questions where the DB legitimately has zero matching rows (e.g. "list schools with SAT math > 800" when there are none).
4. **Per-DB prompt tuning.** Eval per-DB breakdown shows `thrombosis_prediction` and `toxicology` at 0 % across every iteration — the model just doesn't know these schemas well. A small per-DB prompt addendum with key columns + join hints is cheap and might lift the floor.
