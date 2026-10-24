# Three-engine comparative analysis: what qml demonstrated on each engine

Written 2026-07-31, evening. A summary table that maps the measurements in the three engine documents ([llama.cpp](engine-llama-cpp.md) ·
[vLLM](engine-vllm.md) · [SGLang](engine-sglang.md)) to the advantage claims
P1–P5 (restart survival, session-level operation, more users than slots, cross-engine migration, complete history preservation).
Every row is a **product test** (with qml vs. without qml, same conditions, correctness verified).

---

## 1. How qml is connected to each engine

| | llama.cpp | vLLM 0.26 | SGLang 0.5.16 |
|---|---|---|---|
| Connection method | The engine has no plugin entry point, so a **source modification (patch) is needed** (874 lines) | Plugs into a plugin entry point provided by the engine (0 lines of source modification) | Plugs into a plugin entry point provided by the engine (0 lines of source modification) |
| Session concept | Added by qml (session_id, auto-continue) | None — only prefix block queries | None — only page hash keys |
| Passing the session identifier | Request field `session_id` | `cache_salt` (direct hooking was needed) | `extra_key` is reflected in the hash automatically, **the plaintext is not passed to the backend (#37)** |
| Restore unit | 1 blob for the whole session (1 `set_data` call) | Block (16 tokens) × layer | Page (default **1 token**) |

## 2. Resume after restart — P1 (survives restart) and speed

Same protocol: save completed → kill → relaunch → (warmed) → request → **needle correctness check**.
The comparison is stated in each row.

| Engine | Condition | Without qml | With qml | Verdict |
|---|---|---|---|---|
| llama.cpp | Llama-3.1-8B Q4_K_M, 7,950 tokens, only the new turn sent | 791 ms (full retransmission and recomputation) | **43~51 ms** (`--qml-restore-on-start auto`) | **Wins by 15x** |
| vLLM | gemma-3-4b bf16, 6,617 tokens, full retransmission (forced by the engine) | 621 ms | **544 ms** | Wins by 12% (only above the ~4.6K crossover point) |
| vLLM | same, 1,667 / 3,317 tokens | **267 / 338 ms** | 366 / 389 ms | Recomputation wins |
| SGLang | gemma-3-4b, 1,567 tokens | **1,187 ms** | 3,045 ms (reused 1,556/1,567) | Recomputation wins — the bottleneck was confirmed by instrumentation to be inside SGLang (our reads are ~50 ms = 1.5%) |
| SGLang | same, 6,629 tokens | **1,279 ms** | 4,052 ms (reused 2,965/6,629) | same |

**P1 itself is demonstrated on all three engines** — correct answer after restart + supply from the external store were confirmed with the engines' own
counters (`prompt_n` / `external_kv_transfer` / `cached_tokens`). The speed advantage is
decisive only on llama.cpp; vLLM is conditional, and SGLang still loses.

**Why they diverge — the restore unit decides everything**: llama.cpp puts 280 MiB in with one call (6.5 GB/s),
vLLM reaches 3.0 GB/s after batching per layer, and SGLang is forced by the engine to 1 page per token (larger pages
segfault the engine), leaving internal overhead that our batching cannot overcome.

## 3. Demonstration status per claim

| Claim | llama.cpp | vLLM | SGLang |
|---|---|---|---|
| P1 survives restart | Yes (43 ms, RTO class) | Yes (correct, speed advantage above 4.6K) | Yes (correct, not yet on speed) |
| P2 session-level operation (send only the new turn) | Yes **only one** — auto-continue | No impossible by engine design (full retransmission forced) | No same |
| P3 serving more registered users than concurrent processing positions (slots) | Yes 12 users served with 4 concurrent compute positions by disk swap, 2.04x faster than the full-retransmission method | Comparison not applicable — vLLM has no slot concept (block pool method) | Comparison not applicable — same |
| P4 cross-engine migration | Yes a→b, b→a, c=a+b | Yes (with llama.cpp, both ways) | No #37 (session_id) must be solved first |
| P5 complete history preservation | Yes `qml_token_archive` | — (no such scenario) | — |
| No regression (TPOT, throughput) | Not measured (~0 expected because of the asynchronous persister) | Not measured (save 616 ms per 6.6K request — needs measurement) | Not measured |

## 4. What was found in this cycle (2026-07-31 evening)

1. **The real cause of SGLang restart reuse** — not the radix tree hypothesis (earlier record) but
   the default `prefetch_threshold=256`. The earlier 34-token test was cut off at the gate.
   It works normally at 256 tokens or more. **The conclusion of the earlier document is corrected.**
2. **scatter/gather batch C API** (`qml_lmdb_get_scatter/put_gather`) — for an engine (SGLang) whose destinations are separate
   per-page tensors. Round trip verified by unit test. Reuse completeness
   1,154→1,556/1,567.
3. **Defense against segfaults on non-contiguous tensors** — guarantee CPU + contiguous before passing `data_ptr()` to C
   (`_host_contig`). Confirmed that the scheduler dies with exit -11 on a host pool view/CUDA tensor.
4. **2 SGLang engine bugs documented** — `backend_name` became required, `--page-size` 16/64
   segfault (reproduced without qml).

## 4-1. Three-way mixed load comparison with the same model (2026-08-02) — qml+llama.cpp vs. stock vLLM vs. stock SGLang

Model unification: all three servers use Llama-3.1-8B-Instruct 4-bit (llama.cpp uses Q4_K_M GGUF,
vLLM and SGLang use hugging-quants AWQ-INT4). Identical load: 6 long sessions of 32K tokens returning
without pause + 3 short-request workers, 150 s, needle correctness checked on every response
(`tests/mixed-load/bench_mixed_load.py`, stateless-openai mode).

| | qml+llama.cpp (asynchronous restore) | stock vLLM 0.26 | stock SGLang 0.5.16 |
|---|---|---|---|
| Short requests completed | **882** (p50 184 ms) | 99 (p50 4,784 ms) | 99 (p50 4,541 ms) |
| Long-session returns completed | **46** (p50 3,697 ms) | 32 (p50 4,790 ms) | 33 (p50 4,546 ms) |
| Long-session GPU prefill tokens | **506** | 1,026,116 | 1,058,244 |
| Accuracy | 882/882, 46/46 | 99/99, 32/32 | 99/99, 33/33 |

**Verdict: on this workload, qml+llama.cpp is 8.9x on short requests and 1.4x on long-session returns.**
The reason is the combination of two axes: (1) the prefix/radix cache of the stock engines works only inside
the GPU KV pool, and 6×32K (FP16 KV ≈ 24.6 GB) exceeds the pool (~12-15 GB) and is evicted by LRU, so every
return is effectively a full recomputation (vLLM prefill per return 32,066 tokens = hits converge to 0,
a measured confirmation of the claim "the caching benefit is conditional on prefix sharing and pool capacity"). (2) qml keeps the state
on disk, so a return is data movement rather than GPU computation, and with asynchronous restore the
GPU handles short requests in the meantime.

Fairness footnotes (what could not be made completely identical):
- The quantization implementations differ (Q4_K_M vs AWQ INT4) — both are 4-bit class, but kernel and accuracy characteristics differ.
- KV precision differs (llama.cpp q4_0 36 KiB/token vs vLLM and SGLang default FP16 128 KiB/token).
  KV quantization is part of the real advantage of the llama.cpp+qml stack, but the KV compression options of the other side,
  such as vLLM's FP8 KV, were not turned on this time — even with them on, the pool cannot hold 6×32K
  (12.3 GB needed at 64 KiB/token vs pool ~12-15 GB, borderline).
- The two stock servers were measured through the OpenAI-compatible HTTP path (overhead negligible compared with second-scale latency).

## 4-2. Three-way comparison with a fully identical model (2026-08-02) — bit-identical weights, same KV precision

To remove the fairness footnotes of section 4-1 (differences in quantization implementation and KV precision), the model was fully unified:
vLLM and SGLang load the `unsloth/Llama-3.2-3B-Instruct` FP16 checkpoint directly, and
llama.cpp loads the same checkpoint through a **lossless F16 GGUF conversion** (convert_hf_to_gguf; F16→F16 is
numerically identical). KV is FP16 on all three servers as well. 8B FP16 does not fit this scenario on a 24GB GPU,
so 3B was used, and FP16 KV (112 KiB per token) is 4x q4_0, so the session setup was also adjusted to 6×24K
(same for all three servers). Note: llama.cpp cannot load formats other than GGUF (no safetensors,
no AWQ conversion), and vLLM's GGUF loading is broken for local files in this version (the plugin is also
version-incompatible), so "the same HF FP16 original + lossless conversion" was the only fully identical path.

| 6×24K returns + short requests, 150 s, every response checked | qml+llama.cpp (asynchronous restore) | stock vLLM | stock SGLang |
|---|---|---|---|
| Short requests completed | **1,140** (p50 195 / p90 229 ms) | 852 (p50 505 / p90 726 ms) | 282 (p50 1,268 ms) |
| Long-session returns completed | 20 (p50 4,799 / p90 18,301 ms) | **284 (p50 502 ms)** | 94 (p50 1,268 ms) |
| Long-session GPU prefill | **220 tokens** | 6,964,736 tokens | 2,272,496 tokens |

### Verdict — the dependence on conditions shows up in the numbers

- **Short requests (protecting interactive users): qml is 1st even under fully identical conditions** (1,140 vs 852 vs
  282; p50 195 ms is 2.6x faster than vLLM and 6.5x faster than SGLang). The gain of the asynchronous restore design
  holds regardless of model and precision.
- **On long-session return throughput, stock vLLM wins clearly** (284 vs 94 vs 20). The reason is exactly the physics of the crossover
  point: 3B FP16 prefills very fast, so 24K recomputation is ~0.5 s, while the FP16 state blob is
  2.7 GB, so disk movement (0.5 GB/s) takes ~5.4 s. **Recomputation 0.5 s < restore 5.4 s.** In section 4-1 (8B
  4-bit, 32K), recomputation ~4.5 s > restore ~2-3 s, so the inequality was the other way.
- Therefore **KV quantization (q4_0, state 4x smaller) is not a footnote but a core component of the qml stack**:
  the smaller the state, the faster the restore and the lower the crossover point. The "fully identical conditions" requirement is a comparison that removes this
  component, and as a result the long-session metric moved to the stock side — this is the honest overall
  picture.
- qml long-session p90 18.3 s: a queueing effect of 6 fetches of 2.7 GB serialized on a 0.5 GB/s disk.
  On a fast NVMe (7 GB/s class), restore becomes ~0.4 s, and the inequality returns to the restore side
  even for 3B FP16 (computed value, not measured).

### Combining the two comparisons

| Comparison condition | Short requests | Long-session returns |
|---|---|---|
| Section 4-1: each engine's native 4-bit (8B, 32K) | qml ahead by 8.9x | qml ahead by 1.4x |
| Section 4-2: fully identical FP16 (3B, 24K) | qml ahead by 1.3~4x | stock vLLM ahead by 14x |

The crossover formula (restore = state size ÷ storage medium speed vs. recomputation = number of tokens ÷ prefill speed) explains both
tables. Conditions that favor qml: larger model, longer context, use of KV quantization,
faster disk. Unfavorable conditions: small model + FP16 KV + slow disk.

## 4-3. Same weights + each engine's maximum KV compression (2026-08-02) — not a same-condition comparison

**Caution: this section is not a same-condition comparison.** KV precision differs (4-bit vs FP8 vs
FP8 — 4-bit KV exists only in llama.cpp). The only fully identical comparison is section 4-2; this section
answers a separate question: "on the same weights and the same load, what happens when each engine runs with the maximum KV compression
it can turn on in practice?"

Final configuration following the user's remark ("should have compared each at its maximum KV compression from the start"). The weights are identical on all three
servers, unsloth/Llama-3.2-3B-Instruct FP16, and KV uses the maximum compression each supports:
llama.cpp **4-bit q4_0** (31 KiB/token), vLLM **FP8** (56 KiB/token, nothing beyond this),
SGLang **FP8 e5m2** (56 KiB/token, nothing beyond this). 4-bit KV exists only in llama.cpp.
Identical load: 6 sessions × 24K + 3 ordinary-request users, 150 s.

| | qml+llama.cpp (KV 4-bit) | stock vLLM (KV FP8) | stock SGLang (KV FP8) |
|---|---|---|---|
| Ordinary requests handled (p50) | 813 (224 ms) | **1,692 (251 ms)** | 1,485 (285 ms) |
| Reconnects handled (p50) | 66 (2,469 ms) | 603 (252 ms) | 495 (285 ms) |
| **Reconnect accuracy** | **66/66 (100%)** | **540/603 — 63 wrong (10.4%)** | 495/495 (100%) |
| Reconnect GPU compute tokens | **726** | 13,496,220 | 12,330,714 |

### What this experiment established (honest verdict)

1. **vLLM's FP8 KV produced 10.4% wrong answers on this long-context memory task** (the numbers show that accuracy, not
   throughput, broke — without the needle check, only the throughput of 1,692/603
   would have been reported). SGLang's FP8 (e5m2) kept the answers correct. llama.cpp's 4-bit q4_0 has
   never produced a wrong answer in thousands of responses across this whole project.
2. **For a small 3B-class model, stock + FP8 KV has the throughput advantage** (SGLang 1,485/495 is the best among those that keep
   accuracy). The reason is the same physics as section 4-2: 24K recomputation of 3B is ~0.5 s, cheaper than restore
   (0.74 GB ÷ 0.5 GB/s ≈ 1.5 s+). On a small model, recomputation wins.
3. **qml's compute saving is maintained** (reconnect GPU compute 726 vs 12 million+ tokens) — but on this
   machine and at this model size, that saving does not convert into a throughput advantage. With 8B (experiment section 4-1),
   recomputation becomes expensive at 4.5 s, and qml won on both metrics.
4. A trade-off inside qml is also recorded: as 4-bit KV made reconnects faster (20→66), restore and eviction
   save events increased and ordinary requests dropped 1,140→813 (p90 229→1,218 ms). This is because restore
   apply (set_data) and eviction capture (get_data) still run in the loop.

### Final synthesis of the three experiments — the deciding variable is model size

| Comparison | Ordinary requests | Reconnects (including accuracy) |
|---|---|---|
| Section 4-1: 8B 4-bit (each at its best) | qml 1st by 8.9x | qml 1st (46 vs 32) |
| Section 4-2: 3B fully identical FP16 | qml 1st (1,140) | vLLM 1st (284) |
| Section 4-3: 3B same weights + each at maximum KV | stock 1st (vLLM 1,692) | SGLang 1st among those keeping accuracy (495) |

**Conclusion: qml's domain is where recomputation is expensive — the larger the model (8B+), the longer the context,
and the more concurrent sessions exceed the GPU pool, the more qml wins; facing the fast recomputation of a small model (3B class),
the stock engines win. This boundary can be predicted with the formula "restore cost (state size ÷ disk speed) vs. recomputation cost (number of tokens
÷ prefill speed)", and all three experiments agreed with this formula.**
On-premises proposal logic: 8B+ models + long conversations + surviving restarts + history preservation are qml's field, and
lightweight 3B-class serving is not qml's field.

## 4-5. Measured application boundary of the stock vLLM prefix cache (2026-08-02, approved backlog item completed)

Method (`tests/bench_prefix_hit.py` (internal script, not published)): sessions of about 2K tokens with completely different content (starting with a unique nonce so the shared prefix is
0) are visited in stages. This vLLM build has an instrumentation defect in which the response's cache-hit token metric
(`cached_tokens`) is always reported as 0, so **hits are judged by response latency**
(on a hit prefill is skipped, so latency is clearly shorter). Every response passed the correctness check.

| Stage | Condition | Revisit p50 | Verdict |
|---|---|---|---|
| Cross-session | First visit to 8 sessions that share nothing | 287 ms (full recomputation level) | **0% hits across sessions — with nothing shared, the cache does nothing** |
| Revisit within the pool | Revisit 8 sessions (2 GB KV << pool) | **114 ms (2.5x shorter)** | Hits within the pool — the cache works normally |
| Revisit beyond the pool | Revisit 80 sessions (~20 GB > pool ~15 GB), both halves | 285 ms (same as the first visit) | **Hits disappear completely — LRU cyclic eviction (thrash)** |
| Revisit after restart | Server restart → the 8 sessions that were in the pool | 284 ms | **Cache lost entirely** (qml survives through disk preservation — section 4-1) |

The claim made in sections 4-1 to 4-3, "the caching benefit of stock engines is conditional on prefix sharing, pool capacity, and process lifetime", is
confirmed numerically at all four stages.

## 4-6. Cross-engine session migration demonstration — llama.cpp → vLLM → llama.cpp, one session, correct at every stage (2026-08-04)

The product's headline claim (session continuity across engine boundaries) was demonstrated end to end for the first time. **The substance of
the migration is not KV blob movement but logical replay based on the token log** — KV tensor bytes are not portable across engines,
and the only portable currency is the token sequence (exactly as defined in the design document).

Setup: same weights (Llama-3.2-3B FP16 = GGUF/HF of the same checkpoint), 1 session with 1 planted needle
(`alice::mig-1`), and the store directory (SQLite+LMDB) **moved between stages by file copy between hosts**
— llama.cpp ran on the NVMe host and vLLM in the container (same GPU, so launched one after the other).

| Stage | Engine and host | Action | Result |
|---|---|---|---|
| 1 | llama.cpp @ NVMe host | Create the session with an 8K history, 8,017 token log rows recorded | **Correct** (601 ms) |
| 2 | vLLM @ container | Pass the token log of the same session_id as prompt token IDs (cache_salt convention), rebuild with 1 prefill, then ask | **Correct** (957 ms) — the connector saved 501 blocks of vLLM-format KV + an extended token log (8,027) on the same session row |
| 3 | llama.cpp @ NVMe host | Send only the new question (session_id); auto-continue prepends the extended token log | **Correct** (695 ms, prefill 8,046) |

Confirmed facts:
- **Tokenizer compatibility**: the token IDs of the GGUF conversion and the HF original are identical — the IDs recorded by llama.cpp were
  passed to vLLM as is and gave the correct answer.
- **The store is a movable carrier**: the session was kept while moving between the two hosts by tar copy.
- **Migration cost = 1 prefill on the target engine** (at 8K, vLLM 957 ms / llama.cpp 695 ms).
  Later reconnects are restored quickly from each engine's saved data.

### Defect caught by the demonstration — when another engine extends a session, a stale blob is applied wrongly

The first attempt at stage 3 died with **HTTP 500 ("Invalid input batch")**. Cause: the blob suitability check
reads the row shared by engines (`qml_session.n_tokens`), and even after vLLM updated that value to 8,027,
the llama-side blob was still for 8,017 tokens — the stale blob passed the check and was applied under an 8,027-token
context.

Fix (the commit after llama.cpp `76c3d45cb`): added `qml_restore_covers()`, which compares **the engine's actual restored range**
(`llama_memory_seq_pos_max + 1`) with the token log length right after restore, to all 3 restore
paths — on a mismatch it clears the state and falls back to recomputation (correctness first).
Rerun log: `covers 8017 token(s) but the token log holds 8027 -- discarding, will
recompute` → correct. **It showed that a class of defects that appears only in cross-engine operation really exists, and the demonstration
itself became the regression test for it** (`tests/session-handover/bench_migration.py`).

### 4-6 addition (2026-08-04 evening): the SGLang → llama.cpp direction also works

After resolving task #37 (engine-sglang.md section 4): llama.cpp took over a session (3K needle) that SGLang had created with `cache_salt`,
receiving only the new question from the same store — 3,027 token log tokens prepended, correct,
385 ms. With this, **all three engines take part in session recording, and llama.cpp↔vLLM in both directions +
SGLang→llama.cpp are demonstrated**. (The direction "into" SGLang or vLLM is, by the API semantics of those two engines,
a form in which the client resends the history, and identity continuity is kept with cache_salt/extra_key.)

## 4-7. Three-engine qml overhead panel — VRAM 0, time cost +10~44% (2026-08-04)

With the same model (3B FP16) and the same load (12 sessions × about 1,500-token needle, sequential), with qml and without qml
were compared per engine. All configurations correct 12/12. llama.cpp on the NVMe host, vLLM and SGLang in the container (same GPU).

| Engine | VRAM (without/with) | KV capacity | Mean latency (without→with) | Overhead |
|---|---|---|---|---|
| llama.cpp | 6,945 / **6,945 MiB (0)** | Fixed | 170 → 197 ms | +16% |
| vLLM | 21,355 / **21,355 MiB (0)** | GPU KV cache size 122 identical | 175 → 252 ms | **+44% — largest, next optimization candidate** |
| SGLang (hierarchical cache only) | 20,919 / 20,971 MiB | 122,626 tokens identical | 175 → 173 ms | ≈0 (the hierarchical cache itself) |
| SGLang (+qml backend) | 20,947 MiB | Identical | 175 → 192 ms | +10% |

Verdict: **the overhead in VRAM, KV pool, and GPU compute (total prefill identical across all configurations) is 0 on all three engines.** The time
cost arises only in the save path (+17~77 ms per request), and what it buys is session persistence (reconnect 4.8 s→1.08 s, etc.,
from earlier measurements). Footnote: the lower total prefill of llama.cpp without qml (16,591 vs 18,240) is a by-product of sequential requests
reusing the common prefix of the same slot — exactly the characteristic of stock, which has no session isolation.

## 5. Remaining items (whole project)

| Item | Basis |
|---|---|
| No-regression measurement (first evaluation priority) | A qualification requirement that is missing on all three engines |
| Formalizing the session density, RTO, and request size metrics | Primary metrics for P1–P3 |
| SGLang #37 (session_id) → inclusion in cross-engine | Feature gap |
| Retry SGLang speed | After the engine's page-size bug is fixed |
| ~~Measure vLLM prefix-cache hit rate~~ **Done (2026-08-02, results below section 4-4)** | Measured — see section 4-5 below |
| **Upstream issue reports for 2 LMCache defects** (posted 2026-08-02 — awaiting maintainer confirmation) | Defect (1) found in measurement (engine death from FileNotFoundError on eviction) has the same stack as the public unresolved issue #1981 — single-GPU and 0.5.2 reproduction information can be added. Defect (2) (in warm reconnects, KV from other requests is mixed in and 90% of answers are wrong — cross-session contamination) is judged unreported upstream; reproduction script, log, and diagnosis (the wrong answers are the correct answers of other sessions) secured (internal record, not published). Confirmed there is no fix in the latest release 0.5.2. Posted: new issue #4385 + reply on #1981 (KPST-LAB) |

## Three-engine startup footprint measurement (2026-08-06, dev container = same physical GPU)

Same: 3B Instruct model (F16 class), 8K context class, same GPU, ready = first health response.
Different: engine (llama.cpp uses GGUF + KV q4_0, vLLM/SGLang use HF safetensors + default KV).
Each measured 2 times (startup time); RSS in a separate single pass. Script `tests/bench_startup.sh` (internal script, not published),
log: internal record, not published.

| Engine | Time to ready | host RSS (at ready) | GPU memory (at ready) |
|---|---|---|---|
| llama.cpp (llama-server) | **2.6~3.1 s** | **1,166 MiB** | 6,900 MiB |
| vLLM 0.26.0 | 67.5~111.0 s | 4,015 MiB | 21,310~21,914 MiB |
| SGLang | 57.3~58.4 s | 5,510 MiB | 21,784 MiB |

How to read this (no exaggeration): startup time (about 20~40x) and host RSS (3.4~4.7x) can be claimed as measured differences
in engine properties. **The GPU column is not an efficiency claim but a policy difference** — vLLM/SGLang
intentionally pre-reserve the KV pool (0.85 / 0.833 respectively), and llama.cpp allocates only as much as -c.
The causal claim "it is fast because it is C/C++" is outside the scope of measurement — only "consistent with the property of being
a single C++ binary". Operationally meaningful in resource-constrained environments with frequent restarts or a shared GPU.

### 4-6 addition (2026-08-12): migration across multiple physical machines — x86_64 4090 ↔ aarch64 Orin

The paper's 3-hop protocol (bench_migration.py) was repeated on **2 physical machines and 2 ISAs**.
KV bytes do not cross the machine boundary — what moved is the session row + token log (tens of KB of JSON, scp).

| Stage | Machine | Path | Time | prefill | Correct |
|---|---|---|---|---|---|
| 1 build | RTX 4090 host (x86_64) | Session creation | 2,318 ms | 8,006 | OK |
| 2 continue | Jetson AGX Orin (aarch64, after reinstall: kernel 6.8/CUDA 13.2) | token-log rebuild (blob not transferred) | 5,456 ms | 8,036 | OK |
| 3 return | RTX 4090 host | After clearing the slot with PURGE_CACHE, prefix restore — `blob covers 8017 of 8077 -- prefix restore, tail recomputes` | 356 ms | 79 | OK |

Conditions: both sides use llama.cpp b10141 + the qml patch (same source; Orin rebuilt with gcc13/CUDA 13.2),
Llama-3.2-3B F16, q4_0 KV. Orin power mode pmode:000011 (12 cores), NVMe 1.2 GB/s (measured directly,
after the ASPM repair — confirmed absence of dmesg completion-polled). **Caution**: in the first hop-3 attempt the session
was resident in a 4090 slot, so it came out with prefill 19 — a response from the resident state is not a migration verification, so it was
discarded, and only the rerun that forced the restore path with PURGE_CACHE is recorded (8 fillers did not evict the 8K session:
LCP similarity assigned all fillers to slot 5).
