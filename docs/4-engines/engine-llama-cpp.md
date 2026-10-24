# Engine implementation: llama.cpp

One of the three engine documents. [engine-vllm.md](engine-vllm.md) · [engine-sglang.md](engine-sglang.md)

Reference date 2026-07-31. Every number here is measured, and the conditions are stated with it.

---

## 1. Why only this engine needs a source patch

There is **no** official plugin extension point. Handling session state requires access to `server_slot::prompt.tokens`
(type `server_tokens`), which is a C++ object statically compiled into the server binary, so
LD_PRELOAD cannot reach it in principle.

The core of the problem is what `update_slots()` does on every request:

```cpp
n_past = slot.prompt.tokens.get_common_prefix(input_tokens);   // reuse only the common prefix
// then truncate everything after n_past with seq_rm
```

Even if bytes are written into the KV cache with `llama_state_seq_set_data()`, `slot.prompt.tokens` is empty, so
the common prefix is computed as 0 and what was just restored is truncated on the spot. **Restore works only when the KV cache and the server's
C++ bookkeeping are aligned together.**

llama.cpp's own `/slots?action=restore` handler does exactly those two things together
(`load state → prompt.clear() → tokens.insert()`). Our patch reuses that procedure as is.

## 2. 3-layer structure

| Layer | Location | Role |
|---|---|---|
| L1 storage (C, engine-agnostic) | `core/`, `storage/sqlite/`, `storage/lmdb/` | SQLite metadata + LMDB blob. Shared by the three engines |
| L2 llama.cpp-aware wrapper | `hooks/llama.cpp/qml_llama_server_hook.cpp` | C++ layer that knows llama.cpp types. Control command dispatch, persister thread |
| L3 server patch | `third_party/llama.cpp` branch `qml-session` | 10 files including `server-context.cpp`. All under `#ifdef LLAMA_QML_SESSION` |

The build target is **`third_party/llama.cpp`** (branch `qml-session`, base `0d47ea742`). The patch source and
the regeneration procedure are in [../patches/README.md](../../patches/README.md). `$HOME/llama.cpp` was removed
from this project as of 2026-07-31.

```
cd third_party/llama.cpp
cmake -S . -B build -DLLAMA_QML_SESSION=ON -DQML_ROUTER_DIR=$REPO -DGGML_CUDA=ON
cmake --build build --target llama-server -j
```

With `LLAMA_QML_SESSION=OFF` (default) the build is identical to the stock build. At one point a relaxation of
`private:` → `public:` in 3 headers under `src/` was outside the `#ifdef`, but it was **removed on 2026-07-31** — it was left over from the investigation phase, and
the current code does not access those internals (Section 7).

## 3. CLI flags

| Flag | Meaning |
|---|---|
| `--qml-session-store PATH` | SQLite metadata store. If not given, the whole feature is off |
| `--qml-lmdb-store PATH` | LMDB blob store |
| `--qml-restore-under-load` | Attempt restore even when there is no free slot |
| `--qml-restore-on-start ID[,ID...]|auto` | Restore the given sessions into slots **at startup** (the 43 ms in Section 5 comes from this). With `auto`, MRU sessions are selected automatically, as many as there are slots |

The request body gets `session_id` / `user_id` fields. When absent they are empty strings, so existing
clients are not affected.

## 4. Implementation core

### 4-1. `qml_restore_into_slot()` — the single entry point for the restore procedure

The lazy path (on request arrival), the `RESTORE` control command and `--qml-restore-on-start` **all use this one function**.
The procedure is identical to the native `SLOT_RESTORE` handler:

```
load state → slot.prompt.clear() → prompt.tokens.insert(restored) → create_checkpoint()
```

The restore logic being duplicated at three points of the request lifecycle was the **common cause** of bugs #52, #60 and #62.
That is why the consolidation is a correctness measure, not a performance optimization.

### 4-2. auto-continue — the only path where the client can send just the new turn

The server prepends the tokens of the saved/resident session so that `get_common_prefix()` works normally.
Stock `/slots` restore cannot do this — if only the new turn is sent, the restored part cannot become the prefix and
it falls back to a full re-prefill (this is why the "file restore" row of the table in Section 5 sends the full context).

The duplicate-prepend decision is not an "exactly identical prefix" check but a **measurement of the actual overlap** (#62):

```cpp
while (overlap < cmp_n && task.tokens[overlap] == stored[overlap]) ++overlap;
const bool already_prefixed = overlap >= 16 && overlap * 2 >= qml_n_hist;
```

It decides correctly even when the saved part contains generated tokens the client has not seen (7,972 vs 7,960).

### 4-3. Asynchronous persister

In `release()`, only `llama_state_seq_get_data` (memcpy on the model thread, fast) is done synchronously, and
the SQLite/LMDB writes are handed to a background thread queue. At least one save per turn is guaranteed,
while disk I/O does not block the generation hot path.

### 4-4. SWA policy

A windowed SWA (`n_swa != 0`) blob contains only each layer's window, so generation cannot continue after injecting it
— **restoring the same blob twice produced different continuation text.** qml refuses restore under this condition
and falls back to a re-prefill based on the token log. With `--swa-full` (forces the server's `n_swa` to 0),
gemma-3 is also bit-exact.

Stock does not refuse under the same condition — it reports success and then silently recomputes (`prompt_n=3998`).

### 4-5. Other

| Feature | Description |
|---|---|
| Control channel | SQLite command queue + inotify event-driven. `MONITOR_PING` median 5.9 ms |
| `RESTORE`/`MIGRATE`/`BACKUP_NOW`/`PREWARM` | Direct KV cache manipulation. live RESTORE 17.2 ms, MIGRATE 138 ms |
| model fingerprint | An actual content hash, not the file name — detects a different checkpoint at the same path |
| `qml_token_archive` | Preserves tokens that context shift discards (#61) |
| Slot selection | Fast path for a slot whose `session_id` matches |

## 5. Measurement — resume TTFT

Llama 3.1 8B Q4_K_M · q4_0 KV · needle in the middle of a 7,950-token context · blob 280 MiB ·
**server restart after the save completes** · streaming TTFT · answer verified · save excluded from the measurement

| Method | Recovery time point | Client sends | TTFT | prompt_n |
|---|---|---|---|---|
| No restore | — | Full 7,960 tokens | 791 ms | 7,960 |
| File restore (`/slots`) | Admin call (eager) | Full 7,960 tokens | 147 ms | 10 |
| qml lazy | Inside the first request | New turn 11 tokens | 248 ms | 11 |
| **qml eager** (`--qml-restore-on-start`) | **At startup** | New turn 11 tokens | **43 ms** | 11 |

All rows answer correctly. Startup log: `restored session 'ttft-1' into slot 0 at startup
(198.0 ms, 7950 tokens)` — the 198 ms recovery is absorbed into startup time, so the client does not pay for it.
`set_data` itself takes 43 ms for 280 MiB.

**Conclusion: it was not the storage medium (LMDB vs file) but the recovery time point (lazy vs eager) that decided everything.**

Side fact: blob size decides session-vs-prefill. f16 **128 KB/token**, q8_0 **68**, q4_0 **36**.
`--cache-ram` (default 8 GiB) made no difference in this benchmark in an A/B test, ±4 ms (every run starts with `/slots erase`,
so it has no chance to intervene) — keeping the default is correct.

## 6. Fixed bugs

| # | Symptom | Cause |
|---|---|---|
| #50 | Re-save after restore truncated KV outside the SWA window | The re-save path recorded only the window |
| #51 | Stale tail in the token log, `pos_max` off-by-one | `create_checkpoint(n)` → `llama_memory_seq_pos_max()` (inclusive) |
| #52 | lazy restore gap + no SWA policy | Section 4-1/Section 4-4 |
| #60 | **When there is no free slot, a saved session is served without history** (wrong answer) | auto-continue trigger condition |
| #61 | Tokens discarded by context shift are permanently lost in the disk tier as well | Added `qml_token_archive` |
| #62 | auto-continue duplicate prepend → 1,228 ms (slower than no restore) | Prefix decision method (Section 4-2) |
| #63 | Restore time point was at decode time | `--qml-restore-on-start` (Section 4-1) |

## 7. Remaining items (updated afternoon of 2026-07-31)

- ~~`--qml-restore-on-start auto`~~ **Done** — when the value is `auto`,
  MRU sessions, as many as there are slots, are restored automatically with `qml_sqlite_recent_sessions()` (updated_at DESC).
  Verification: after a restart the server found `ttft-1` on its own and restored it in 199.5 ms, resume TTFT 51 ms, correct answer,
  prompt_n=11.
- ~~`private:` → `public:` 4 lines~~ **Removal done** — left over from the investigation phase; compilation confirmed that the current code does not
  access those internals. Now every change in the patch is inside `#ifdef LLAMA_QML_SESSION`.
- ~~#22~~ Closed (already resolved by #48/#53).
- ~~Concurrency re-measurement~~ **Done** — 5 controls + `third_party` binary, 12 sessions × 3 series (base/c1/c2),
  4 slots, sessions ~2,089 tokens, q4_0. **kv_restore fired 12/12 on every request** (unlike the earlier invalid measurement):

  | | session_restore (new turn only) | stateless_prefill (full resend) | Ratio |
  |---|---|---|---|
  | median | 1,362 ms (prompt_n=9) | 2,781 ms (prompt_n=2,093) | **2.04x** |
  | p90 | 2,246 ms | 3,822 ms | 1.70x |
  | whole batch | 2,305 ms | 3,890 ms | 1.69x |
  | sequential baseline | 361 ms | — | — |

  The earlier 2.4x/2.63x/3.83x are discarded and replaced by these numbers. **Caveat**: in the determinism check,
  c1==c2 12/12 (the load path is self-consistent) but base==c1 7/12 — the existing observation that the continuations
  of sequential and concurrent runs diverge deterministically was reproduced. All mismatched outputs are alternative
  continuations that fit the context (presumed to come from batch-composition numerics), and prompt_n=9 + 12/12 restores confirm
  that the history itself was supplied correctly. Finding the cause remains a deferred item.

## 7-1. Density benchmark (late night 2026-07-31) — honest reversal data

**What was tested**: "Can one GPU (4 slots) serve 150 registered users within the SLO?"
150 sessions (each ~1.5K tokens, a unique needle per session — **all 500 requests are self-verifying**), Zipf(1.1)
arrivals, concurrency 8. Same trace, same seed on two servers (`tests/bench_session_density.py` (internal script, not published)):

| | Session method (new turn only, qml server) | Full-resend method (server without qml, default settings) |
|---|---|---|
| Accuracy | 500/500 | 500/500 |
| p50 / p90 / p99 | 1,282 / 2,429 / 3,309 ms | **1,020 / 1,191 / 1,379 ms** |
| throughput | 5.32 req/s | **7.80 req/s** |
| SLO (p90 < 2 s) | **FAIL** | **PASS** |

**At this operating point the qml session method loses.** It looks contradictory to the 2.04x win in the 12-session test, but
the log breakdown explains both results consistently:

- On the full-resend side, **265 of 500 requests (53%) had a prefill of 500 tokens or less** — the resent history was
  absorbed by llama.cpp's own cache (slot LCP reuse + `--cache-ram` default 8 GiB RAM tier).
  The working set of this workload (7.9 GiB) fits exactly inside the default RAM cache.
- In the 12-session test there was **no repetition** (each session looked up once) → cache ineffective → all actual prefill
  → session method won by 2.04x. In the density test, Zipf repetition warmed the cache and reversed the result.
- On the qml side, 417/500 (83%) were disk blob restores, and each restore is serialized on the model thread as
  `set_data` (+`get_data` for the evicted session), blocking decoding on the other slots.

**Structural finding**: qml's operating model ("the client sends only the new turn") **disables the engine's own prompt
cache** — because no history text to compare against reaches the server. P2 (new turn only) and
"free reuse from the engine cache" are a trade-off.

**Where qml still wins** (what this benchmark did not change): on restart, `--cache-ram` is
entirely lost — recovery after restart (RTO) remains qml's strength. And scales where the working set exceeds RAM
were not measured (next measurement target).

This result **does not replace the 12-session 2.04x but adds a condition to it**: that number is the value for "cold traffic
without repetition", and for hot traffic with heavy repetition + a working set that fits in RAM, llama.cpp with default settings
is better.

## 7-2. Proof roadmap steps 2 and 3 measurement (2026-08-01)

Proof structure defined by the user: (1) state persistence (done) (2) qml optimization → more concurrent users
(3) GPU computation + validity of first- and second-level caching. This measurement fills the first half of (2) and (3).

### (2)-a. Reduction of GPU computation — proven (1/44)

With the same 150 sessions, 500 requests and same seed, the **number of tokens the GPU actually computed in prefill**
(sum of `timings.prompt_n`) was captured:

| | Session method (new turn only, qml) | Full-resend method (no qml, default settings) |
|---|---|---|
| Tokens computed by the GPU (sum over 500 requests) | **5,480** (average 11 per request) | **239,187** (average 478 per request) |
| Accuracy | 500/500 | 500/500 |
| p50 / p90 | 1,082 / 1,374 ms (SLO PASS) | 1,013 / 1,188 ms (SLO PASS) |
| throughput | 6.96 req/s | 7.82 req/s |

**GPU prefill computation per request goes from 478 → 11 tokens, 1/44** (the full-resend side is also a value where its own cache
absorbed most of the history, so compared with a cold condition without cache it is ~1/150). Restore brings the history back
with memcpy instead of GPU computation. This is the measured proof of "applying qml reduces GPU computation".

Note (difference from the first run in Section 7-1): the p90 of 2,429 ms in the first session-method run overlapped with
disk exhaustion at the end of the run. In a rerun with free disk space, p90 is 1,374 ms and passes the SLO. Both values
are recorded — the fact that the session method's tail latency gets much worse when storage space runs out is itself
valid operational information.

### (2)-b. "→ more concurrent users" does not yet hold at this operating point

Computation is 1/44, yet throughput is actually 11% lower. At this operating point (1.5K-token sessions, repeated traffic with a warmed
cache), **prefill computation is not the bottleneck** — on both sides the bottleneck is decoding and
scheduling, and that is the same. The saved computation converts into more users only in regions where prefill is the
bottleneck: long sessions (prefill superlinear), cold traffic, right after restart, working sets that exceed the cache. Measuring
capacity in that region is the remaining half of (2).

### (3). Validity of the tiers (L1/L2/L3) — structure proven, value of the RAM tier conditional

The same ~1.5K-token session was forced into each tier and resume latency was measured. **The source of every sample was verified with server log
events**, and measurements that came from the wrong tier were discarded (`tests/bench_tier_latency.py` (internal script, not published)):

| Tier | How it was forced | Resume latency (median) | Valid samples |
|---|---|---|---|
| L1 — resident in a VRAM slot | Made resident by the previous request | **127 ms** | 5/5 |
| L2 — RAM restore | Slot eviction with 5 fillers with unique prefixes | 161 ms | 1/5 |
| L3 — disk cold restore | Eviction + LMDB pages evicted with `posix_fadvise(DONTNEED)` | 158 ms | 2/5 |

- Cost of losing the slot = **+~30 ms** (`set_data` memcpy of a 52 MB blob — consistent with the earlier measured 6.5 GB/s).
- **Honest finding: at this session size there is no difference between L2 and L3** — NVMe sequential reads are fast enough that
  the value of the RAM tier (PREFETCH_RAM) is limited to large blobs or slow-disk conditions.
- Why the gate pass rate is low (1/5, 2/5): llama-server's slot selection is **LCP (similarity) first**, so
  forcing eviction with fillers is unreliable — this fact itself is an engine behavior confirmed by measurement this time.

## 7-3. The remaining half of (2): right after restart + working set larger than the cache (2026-08-01)

In (2)-b we left the note "the saved GPU computation converts into more users only in regions where prefill is the bottleneck".
The two candidate regions were then measured. Scenario: 150 users each continue their own session
(turns are serial per session, 8 different sessions concurrent), and every response is verified for correctness with a needle.
The comparison is always between the same two — **session method** (the client sends only the new turn, qml server) vs
**full-resend method** (the client resends the full history, default llama-server without qml).

### Right after restart (1.5K tokens × 150 sessions, 500 requests right after the server is killed)

| | Session method (qml) | Full-resend method (default server) |
|---|---|---|
| p50 / p90 | 1,071 / **1,277 ms PASS** | 1,006 / **1,197 ms PASS** |
| Accuracy | 498/500 | 500/500 |
| GPU prefill (average per request) | 11 tokens | 477 tokens |

**Verdict: at this size there is no restart penalty on either side.** Batch prefill of 1.5K tokens is too
fast (a few hundred ms), so even right after a restart, full resend does not get slower. The restart advantage holds only in
the region where "the history is long and prefill recompute takes several seconds" — 43 ms restore vs several seconds of recompute
on a single 8K-token session was already measured in Section 5, but in the 150-session density condition, 1.5K is not in that region.

### Working set larger than the cache (2.5K tokens × 150 sessions = 13.3 GiB > --cache-ram 8 GiB)

| | Session method (qml) | Full-resend method (default server) |
|---|---|---|
| p50 / p90 / p99 | 1,741 / 3,521 / 5,247 ms **FAIL** | 644 / 1,100 / 1,615 ms **PASS** |
| throughput | 2.41 req/s | 6.45 req/s |
| Accuracy / server errors | 500/500 / 0 | 500/500 / 0 |
| GPU prefill (average per request) | **11 tokens** | 420 tokens (**1/38 difference**) |
| Response source | Disk restore 425, resident reuse 75 | Most absorbed by the engine's own cache |

**Verdict: the reversal holds even when the working set exceeds the RAM cache.** The reason is the nature of caching —
in Zipf(1.1) traffic the frequently arriving sessions (hot set) fit inside 8 GiB, and --cache-ram captures exactly
that hot set. The session method, in contrast, pays a restore + eviction save of a 90–180 MB blob for 85% of requests.
On this hardware (single GPU, prefill ~9K tok/s), a 2.5K-token prefill (~a few hundred ms) is
cheaper than moving a large state. **The 1/38 GPU computation still holds under this condition, but that
saving in computation does not convert into a latency or throughput advantage** — because GPU computation is not the bottleneck.

### Product bug found by this test: lost turns caused by a race between asynchronous save and restore (P5 violation)

During the density test, the maximum n_tokens of the session method (2,899) was smaller than that of full resend (4,119) —
**recent turns were silently disappearing from server-side sessions.** Because the needle is at the beginning of the history,
this type of error is not caught by the accuracy check. Cause: the save at turn completion goes into the asynchronous persister queue, and
if the next turn of the same session arrives after slot eviction **before the queue commits**, the old blob in the store is
restored (no read-your-writes, last-writer-wins). Fix: the 3 restore paths now look up the
persister queue and in-flight jobs before the store (`pending_save_lookup`,
`hooks/llama.cpp/qml_llama_server_hook.cpp`). After the fix the maximum n_tokens is 4,945, the same as full resend
— turn preservation confirmed. This fix increased the session-method wall time from 141→207 s; **the faster
number before the fix is invalid because it restored a shorter (stale) state.**

### Operational facts confirmed along the way

- **LMDB file growth**: as eviction and re-save repeat and blobs grow, one 500-request run takes it from 13.5→18 GB
  (+35%). Storage capacity planning needs a growth factor and a compaction policy.
- **Slot context overflow cliff**: at 4,096 tokens per slot (-c 16384, -np 4), when a hot session grew
  to 4,119 tokens, the full-resend method failed 38 of 500 requests with a server error ("exceeds the
  available context size"). Workloads where sessions grow must size -c to include the growth.
- **Do not send concurrent turns to the same session**: serializing turns per session is a client/gateway
  protocol requirement (naturally satisfied, since a real user cannot send two turns at the same time).

## 7-4. Proof centered on first-level caching (VRAM slots): GPU maximum capacity calculation and slot expansion (evening of 2026-08-01)

User instruction: the recovery/restore path has already been proven, so **calculate the maximum resident
capacity from the GPU state (number of slots, VRAM) and prove the system effect of first-level caching in that configuration.**

### GPU maximum capacity calculation (24GB GPU, Llama-3.1-8B Q4_K_M, KV q4_0)

| Item | Value | Basis |
|---|---|---|
| Total VRAM / fixed part (model+buffers) | 24,564 / 5,242 MiB | measured with nvidia-smi |
| KV per token | 36 KiB (q4_0) | measured |
| Tokens that can stay resident in VRAM | ≈534K | (24,564−5,242−512) ÷ 36KiB |
| **Maximum slots** at 4,096 tokens per slot | **≈130 (operated safely at 112, settles at 22.9GB)** | -c 458752 -np 112 |

### First-level cache hit rate and performance by number of slots (session method, 150 sessions × 1.5K tokens, 500 requests)

| Configuration | Hit rate (resident reuse) | p50 / p90 | Throughput | Verdict |
|---|---|---|---|---|
| 16 slots, engine cache default | 53% (264/500) | 1,066 / 1,615 ms | 4.37 req/s | PASS |
| 112 slots, engine cache default | 92% (461/500) | 1,323 / 3,466 ms | 2.96 req/s | **FAIL — conflict below** |
| **112 slots, --cache-ram 0** | **92% (459/500)** | **619 / 854 ms** | **7.88 req/s** | **PASS, best of the whole campaign** |

### Finding: double-caching conflict with the engine RAM prompt cache → product configuration rule

With 112 slots + default settings, the server-internal computation is p90 696 ms but the client p90 is 3,466 ms —
the difference is outside model computation. Cause: the engine's own RAM prompt cache (--cache-ram default 8 GiB) performs a
54 MB state copy and eviction per turn in the server loop (eviction events: 62 with 16 slots vs 861 with 112 slots;
150 sessions×54 MB=8.1 GiB exceeds the cache limit, so every insert causes an eviction). qml already handles the RAM/disk
tiers, so this cache is a double burden. **Rule: servers with qml use --cache-ram 0.**
Turning it off brought p90 from 3,466→854 ms.

### Final comparison at the maximum configuration (112 slots) — each with its best settings

| | Session method (qml, --cache-ram 0) | Full-resend method (default server) |
|---|---|---|
| p50 / p90 | 619 / 854 ms PASS | 569 / 725 ms PASS |
| Throughput | 7.88 req/s | 8.34 req/s |
| GPU prefill (per request) | **11 tokens** | 228 tokens (**20x difference**) |
| Accuracy / errors | 500/500 / 0 | 500/500 / 0 |
| Survives restart, preserves history | **Yes** | No (slots and cache all lost on restart) |

**Verdict: at the VRAM maximum configuration, the session method reached the same speed class as full resend (p90 difference
~130 ms), with 20x less GPU computation, and only qml survives restarts and preserves history.** Going 4→16→112
slots, the first-level cache hit rate (16→53→92%) determines system performance — the original frame of "first-level caching for
as much as VRAM holds, disk preservation for whatever overflows" is supported by measurement.

## 7-5. Store growth problem and chunked storage — verification as a compaction replacement (2026-08-02)

Question: can the qml logic replace the LMDB compaction function?
Measurement method: run 500-request churn (restore + re-save) twice on a 150-session (7.9 GB) store and
record the file page usage and freelist (pages waiting to be reused) at each point with `mdb_stat`.

### Cause analysis (measurements of the single-blob storage method)

If session state is overwritten as one value per session, the value size always changes because the session grows every turn.
LMDB writes a value whose size changed to new pages with copy-on-write, so every save reallocates the entire
blob (~54 MB = 13,000+ pages). Measured: one churn run adds +2.14 GB to the file, of which
94% is freelist. In addition, on the second run p90 jumped to 49,931 ms and 3 responses were lost,
and this was reproduced (the suspected cause is the sharply increased cost of finding a contiguous free page range for large value writes when
the freelist has grown).

### Measure: chunked storage in fixed 4 MiB pieces (`qml_lmdb_put_blob/get_blob/delete_blob`)

- The blob is split into fixed 4 MiB pieces stored under keys of the form `<key>#000000`, and the base key holds only a 16-byte
  header (magic + total size). **When overwriting a value of the same size, LMDB (WRITEMAP) overwrites the pages
  in place**, so no copy-on-write occurs — even as the session grows, only the last piece is
  reallocated.
- A read returns a malloc'd copy assembled from the chunks in order and **closes the read transaction immediately**
  (previously the transaction stayed open during the GPU upload and prevented page reclamation).
- The existing single-blob format is still supported for reads (backward compatible); it switches to the chunk format on the next save.

### Results — before/after comparison with the same probe

| Metric | Single-blob storage | Chunked storage |
|---|---|---|
| File growth, churn run 1 | +2.14 GB | **+0.46 GB (4.7x less)** |
| freelist, churn run 1 | 2.01 GB | **0.27 GB (7.4x less)** |
| File growth, churn run 2 | +0.82 GB | +0.26 GB |
| Revisit p90, run 2 | **49,931 ms, 3 responses lost** | **1,800 ms, 500/500 correct** |
| Actual file size after 2 churn runs | 11 GB | 8.7 GB |
| Size after `mdb_copy -c` compaction | 8.2 GB (2.7 GB reclaimed) | 8.3 GB (**only 0.4 GB reclaimable**) |

### Verdict

- **qml's chunked storage logic replaces compaction.** Even after churn the store keeps ~96% density,
  so there is almost no space for compaction to reclaim. The remaining growth (+0.26~0.46 GB per run) is
  data from sessions that actually grew plus reallocation of last pieces, and it has an upper bound.
- The p90 spike anomaly in run 2 (49,931 ms) also disappeared with the switch to chunks — this causally supports the hypothesis that
  freelist bloat was the cause.
- `mdb_copy -c` (measured 83 s / 11 GB, offline) is needed only to shrink the file itself after deleting many sessions,
  and even then the qml persister can run it itself during idle time — no external
  compaction daemon is needed.

## 7-6. Restore vs recompute for long contexts — three lengths measured (2026-08-02)

Background: in agent workloads the context of one session grows to hundreds of thousands of tokens. In this region
the cost of the two ways to bring a session back to the GPU was measured under the same conditions
(`tests/reconnect-restore/bench_long_context.py`). Conditions: 1 session, needle at the 10% point, temperature 0,
every response verified correct, restore in a **disk cold** state with a server restart + even the OS page cache removed with `posix_fadvise`
(worst case), source verified with restore events in the server log.

| Length | State size | qml restore (disk cold) | Recompute (full prefill) | Ratio |
|---|---|---|---|---|
| 8K tokens | 0.27 GiB | 1,083 ms | 884 ms | recompute 1.2x faster |
| 32K tokens | 1.10 GiB | 4,011 ms | 4,523 ms | **crossover point — restore 1.1x faster** |
| 128K tokens | 4.40 GiB | **15,107 ms** | **39,132 ms** | **restore 2.6x faster** |

Observations:
- Recompute speed drops as length grows (8K: 9.0K tok/s → 128K: 3.3K tok/s,
  attention cost increases). Restore is proportional to state size and its speed is constant
  (~0.27-0.29 GB/s — the disk cold read speed of this machine is the upper bound).
- Therefore **the disk-cold crossover point is about 30K tokens**, and above it the advantage of restore widens
  as length grows. Restore of a state already in RAM is bounded by memcpy speed
  (earlier measured 6.5 GB/s, 43 ms for 280 MiB), so about 1 second for 4.4 GiB is
  expected, but the numbers in this table are disk-cold measurements only.
- **The 128K state (4.40 GiB) exceeds the LMDB single-value limit (4 GiB)** — saving it at all requires the chunked
  storage of Section 7-5, and this measurement was the first to pass save, restore and answer
  verification of a blob over 4 GiB.
- 1M tokens (state ~37 GiB) does not fit even one slot on this GPU (24 GB), so it cannot be measured on this
  machine. Using the measured speeds above gives ~2 minutes restore vs ~8+ minutes recompute, but this is not a
  measurement; it is a calculation that plugs in measured speeds.

### Restore time breakdown and fix of the peek path (same day)

To break down the composition of restore time, disk cold and page-cache warm were measured separately on a 128K session;
8.5 seconds remained even when warm, and tracing found an inefficiency introduced by the switch to chunks:
the lookup (peek) path that only needs the token list before restore was reusing `qml_session_snapshot_load`
and **unnecessarily assembling and copying the whole 4.4 GiB blob**. A token-only lookup,
`qml_session_snapshot_load_tokens` (reads only the blob header to check existence, keeping the same consistency checks), was
added and peek was switched to it.

| 128K restore | Before fix | After fix |
|---|---|---|
| Disk cold | 15,107 ms | **12,480 ms (−2.6 s)** |
| Page cache warm | 8,452 ms | **5,899 ms (−2.6 s)** |

Main components of the remaining 5.9 s warm (estimated, not instrumented): blob assembly in the restore body (4.4 GiB malloc
first touch + memcpy), `set_data` (4.4 GiB), SELECT of 128,010 token log rows. There is room for further
optimization, but in order, instrumenting the next bottleneck comes first.

Meaning of restore time (what was established by measurement):
1. **Restore does not occupy the GPU** — the 39 s recompute monopolizes the GPU so no other session can be served,
   but restore is disk/CPU work, so the GPU can process other requests in the meantime.
2. **Restore time is a linear cost of state size ÷ storage medium speed**, so it is predictable (RTO guarantee).
   The disk cold read of this machine is measured at 0.54 GB/s — with a current NVMe (~7 GB/s) the crossover point drops to the range of a few
   thousand tokens.
3. **The value of the RAM tier is proportional to state size** — for 1.5K sessions L2≈L3, but at 128K
   the difference between cold and warm was measured at 6.6 s.

## 7-7. Mixed load: does reduced GPU computation lead to serving more users? (2026-08-02)

Question: if qml reduces GPU computation, can that headroom be used to process more requests from other users?
Method (`tests/mixed-load/bench_mixed_load.py`): while 6 long 32K-token sessions continuously return on 4 slots
(each return causes a slot swap), 3 workers of short requests (~40 tokens each, including a unique needle) keep sending requests
as "other users". 153-second window, every response verified correct, same load for both configurations.

### First, the measured contribution of GPU computation time (1.5K context, re-analysis of the 112-slot logs)

- Amount of computation (token count): 5,500 vs 113,892 — **the 20.7x reduction is real.**
- But in terms of time, prefill computation time is 37.7 s vs 53.9 s, **only a 1.4x difference** —
  when only the new turn is computed, the fixed cost per request (small-batch prefill ~75 ms/request) dominates, and
  server-loop stalls on the save path eat into the time. The time breakdown established why the reduced computation did not
  convert into throughput for short contexts.

### Mixed load results (32K long sessions — the region where prefill takes seconds)

| | With qml (restore) | Without qml (full resend) |
|---|---|---|
| Long-session returns completed | **53** (p50 2,980 ms) | 32 (p50 4,730 ms) |
| Short requests completed | 114 (p50 3,686 / p90 5,799 ms) | **139** (p50 4,040 / p90 4,768 ms) |
| Sum of GPU prefill tokens | **4,778** | 960,640 (**201x**) |
| Accuracy | long 53/53, short 114/114 | long 32/32, short 139/139 |

### Verdict — half holds, and the barrier for the other half has been identified

- **Long-session service rate is 1.66x** (53 vs 32): the return cost itself is clearly cheaper with restore.
- **GPU arithmetic is reduced 201x**: the side without qml spent practically all of its 153 s on prefill
  (955K tokens ÷ 153 s ≈ 6,245 tokens per second — the GPU stays occupied). On the qml side the GPU is mostly idle.
- Yet **short-request throughput was actually 18% lower** (114 vs 139). Cause: llama-server's
  restore (disk read + set_data) and eviction save (get_data) **run synchronously in the single model
  loop**, so decoding on the other slots also stops during them. In contrast, recompute prefill proceeds in batches
  interleaved with the other slots' decoding, so short requests keep flowing.
- **Conclusion: the last barrier of "less computation → idle GPU → serving more users" is asynchronous restore.**
  GPU idle time exists by measurement (prefill reduced 201x). What prevents other users from using that idle time
  is not the GPU but the server loop structure. Moving the disk read (most of the restore time) and the
  state copy out of the loop (to a background thread) makes it a structure that gets both metrics at once
  — the next engineering item.

## 7-8. Asynchronous restore implementation — completing "less computation → more concurrent users" (2026-08-02)

This resolves the barrier identified in Section 7-7 (restore blocks the server loop). Implementation in two steps:

1. **Parallel blob assembly** (`qml_lmdb_get_blob`, 4 workers): applied after measuring that reads of data in the
   page cache scale from 5.4→15.2 GB/s. It was also confirmed by measurement that it has no effect on disk cold reads, which are at the device limit
   (0.52 GB/s). Because per-worker transactions may see different snapshots, each worker revalidates the header in its own transaction,
   and on mismatch it retries on the single-transaction
   path. warm 32K restore median 2,218 → 1,752 ms.
2. **Asynchronous fetch + synchronous apply** (hook fetch thread + llama-server patch):
   when `launch_slot_with_task` finds a saved session it starts a background fetch, and
   `update_slots` defers only that slot until the fetch finishes (STARTED gate, resumed by a wake
   callback) while continuing to serve the other slots. Only `llama_state_seq_set_data` (~0.17 s per GB)
   remains in the loop. llama_context is not
   thread-safe, so set_data and the eviction capture (get_data) stay in the loop — this split is the core of the design.

### Mixed load re-measurement (same conditions: 6 long 32K sessions returning continuously + 3 workers of short requests, 150 s)

| | With qml + asynchronous restore | Without qml (full resend) | Synchronous restore of Section 7-7 (reference) |
|---|---|---|---|
| Short requests completed | **882** (p50 184 / p90 1,527 ms) | 145 (p50 3,572 / p90 4,761 ms) | 114 |
| Long-session returns completed | **46** (p50 3,697 ms) | 32 (p50 4,749 ms) | 53 |
| Sum of GPU prefill tokens | 32,862 | 941,594 (**29x**) | — |
| Accuracy | 882/882, 46/46 | 145/145, 32/32 | — |

**Verdict: the user's proposition "applying qml reduces GPU computation and increases concurrent users" was proven in this
configuration.** While long-session returns continue, short-request processing is **6.1x** that of the no-qml side
(882 vs 145), and long-session returns are also 1.4x (46 vs 32) — an advantage on both metrics at once.
The GPU idle time that loop blocking consumed in the synchronous-restore period (114) was converted by making it asynchronous into actual user
processing. Stated condition: this gain is for the long-context (32K) region where the return cost takes
seconds; for short 1.5~2.5K sessions, the verdicts of Section 7-2 (b) and Section 7-7 (does not hold) remain valid.

## 7-9. The KV blob is tied to the slot configuration, but the session is not (2026-08-02)

llama.cpp's per-sequence state save writes `n_stream` at the very start of the blob
([llama-kv-cache.cpp](../../third_party/llama.cpp/src/llama-kv-cache.cpp) state_write:2129),
and on restore throws an exception if it differs from the current context (state_read:2202 `n_stream mismatch`).
Since `n_stream = kv_unified ? 1 : n_seq_max`, **it is effectively the `-np` value**. Therefore if `-np` is changed or
the server switches to `--kv-unified`, **existing KV blobs cannot be reused.**

**But the session is not lost — it is recovered automatically through an already implemented path.** This is because qml keeps
the token sequence (`qml_token_log`) and session metadata separately from the KV blob:

| Step | Behavior | Code |
|---|---|---|
| 1. Request arrives | Reads the saved token sequence and prepends it to the request. For the blob it checks **only existence (header)**, so it passes even if the slot geometry differs | `qml_session_snapshot_load_tokens` → `qml_lmdb_blob_stat` |
| 2. KV restore attempt | slot geometry mismatch → `state_seq_set_data` returns 0 → warning log, then the slot is cleared | server-context.cpp:3919 |
| 3. Automatic fallback | The full history is already prepended to the request, so **the session is recovered by a normal prefill** | existing auto-continue path |
| 4. Re-save | At the end of the turn, **the blob is rewritten with the new slot geometry (number of slots × context size per slot)** | asynchronous persister |

That is, the cost of a slot geometry change is **one recompute per session, the first time**, and neither data loss nor a migration tool
is involved. With stock llama.cpp the session disappears the moment the slot geometry changes, but qml has the token sequence
and brings it back — **a real case where qml fills in what the engine cannot do.**

Side confirmation: this path has not appeared in the logs until now because every measurement created a new store
(0 mismatches). As an optional improvement, recording the slot geometry (n_stream, slot
context size) in the metadata would let a mismatch be predicted before reading the blob (not required).

**Verification method**: save in shared-pool mode (`--kv-unified`) → restart → restore → if needle recall
passes, the whole chain above (blob restore when the slot geometry matches, recompute fallback when it does not) is verified at once.

## 7-10. Support for the shared KV pool (unified) and slot-geometry independence of the blob (2026-08-02)

Result of executing steps 1~3 of (internal record, not published).

### Finding: llama.cpp already had a shared KV pool mode

`n_stream = kv_unified ? 1 : n_seq_max` ([llama-kv-cache.cpp:82]). With `--kv-unified` on,
the context per slot becomes **the whole n_ctx** instead of `n_ctx / number of slots`, and a session occupies only
as much as it uses. All of our measurements specified `-np N` and ran with this mode off — "llama.cpp has
fixed slots and is at a disadvantage at large scale" was **true only with default settings**.

### Step 1 measurement — do qml save and restore work in the shared pool?

3B F16 model, q4_0 KV, 3 sessions (506 / 4,002 / 16,008 tokens), needle recall verified on every response:

| Step | Configuration | Result |
|---|---|---|
| Save | `--kv-unified -np 4` | `n_ctx_slot = 65536` (4 slots, yet each session can use the whole context), store 633 MB — **proportional to the actual token count**, not to slot capacity |
| Restart with the same configuration | `--kv-unified -np 4` | 3/3 restored, GPU computed 11 tokens, answer OK |
| **Restart with a different number of slots** | `--kv-unified -np 16` | 3/3 restored, GPU computed 11 tokens — **saved sessions are independent of the number of slots** |
| Control group | no unified, `-np 4` | `n_stream mismatch` 3 times → 0 KV restores, but **3/3 correct** (automatic fallback to token-log recompute) |

Restore was also faster than recompute (167/265/655 ms vs 255/353/1,374 ms).

### Step 2 — removing wasted reads with slot geometry metadata

`n_stream` and `n_ctx_seq` are recorded in `qml_llama_session_ext`. The values are derived from the public API
(if `n_ctx_seq == n_ctx` then n_stream=1, otherwise `n_seq_max`) — no server patch needed. When the saved value and the current value differ, the restore path
rejects **before reading the blob** and hands over to token-log recompute
(removes a 4.4 GiB wasted read for a 128K session). This check is not put in the peek path — self-recovery works only if the history is
prepended. Existing rows have the value 0 (unknown), so they behave as before.

Found and fixed during verification: the asynchronous fetch thread read the blob **before any restore entry point**,
so after a restart exactly 1 first session passed with the geometry unset. Fixed by synchronizing once in the server's `init()`;
0 mismatches.

### Step 3 — slot-geometry independence by rewriting the blob header

A per-session blob has the structure `[magic][seq_id][n_stream][n_stream sections]`, and exactly 1 section
holds data. If only that section is kept and n_stream in the header is rewritten, a context with a different configuration accepts it
**without touching a single byte of KV data**.

| Direction | Verdict |
|---|---|
| **N → 1 (restore into unified)** | **Verified** — 3/3 correct, actual KV restore (GPU computed 11 tokens), reproduced repeatedly |
| 1 → N (restore into non-unified) | **Disabled because unproven.** Wrong answers were observed, but the verification script at the time did not wait for asynchronous saves to complete between steps, so the saved state itself may have been inconsistent. Rejected and replaced by recompute until the cause is found |

That is, **a session saved under non-unified can be restored on a unified server without recompute** —
the store does not have to be discarded when switching to shared-pool mode.

### Product defect fixed at the same time — a path that answered without context

When a saved session was larger than the current slot context (e.g. slots became smaller because `-np` was increased), the server
could not prepend the history and **answered from the question alone**. In the measurement, when a 16,008-token session
met an 8,192 slot, it produced an answer made up without context — the same type of silent wrong answer we pointed out in LMCache.
Now it rejects and returns an actionable message ("increase --ctx-size, reduce -np, or
use --kv-unified"). Note: in unified mode the context per slot is the whole context, so this
situation almost never occurs.

### Operational recommendation

Turn on `--kv-unified`. Reasons: (1) a session occupies only as much as it uses (variable) (2) saved sessions become
independent of the number of slots (3) rejections due to exceeding the slot practically disappear. The verification script is
`tests/bench_unified_compat.py`, and the run procedure is (internal record, not published).

## 7-11. NVIDIA AGX Orin measurement — on the edge the crossover point moves below 8K (2026-08-03)

The item previously written only as a calculated value, "on Orin an advantage of tens of times is expected by calculation", was
replaced with measurements. **This is a comparison where the model file, scripts, KV settings and verification method are kept completely identical and only the hardware
changes**, and Orin was measured in 2 power modes.

### Measurement conditions (identical for all three conditions)

| Item | Value |
|---|---|
| Model | Llama-3.2-3B-Instruct **F16 (the same file transferred with scp)** |
| KV cache | q4_0 4-bit (measured at 31 KiB per token, identical in all three conditions) |
| Server settings | `-ngl 99 -fa on -np 1 --cache-ram 0` |
| restore condition | after save, **server restart + disk cold** (worst case) |
| Verification | needle recall on every response — all 6 measurements correct |

Machine specs in summary:

| Hardware condition | GPU / CPU | Storage read |
|---|---|---|
| Dev server | **RTX 4090 (VRAM 24,564 MiB)** + Intel i7-5820K 6 cores 12 threads, system memory 62 GiB | 0.55 GB/s (disk cold, no NVMe installed) |
| Orin power-limited mode | Jetson AGX Orin Developer Kit (32 GB-class unified memory, about 29 GiB usable, compute capability 8.7, CUDA 12.6), **8 cores online** (the mode ID was not logged at the time of measurement) | **NVMe 0.92 GB/s** |
| Orin MAXN | Same machine, **MAXN (`pmode:0000`), 12 cores online, CPU 2.2 GHz** | **NVMe 0.92 GB/s** |

Note: the earlier record that described Orin as "8 cores" gave not the machine spec but the number of cores that were
on in that power mode. AGX Orin has 12 cores, all of which are online in MAXN.

### Results — all three conditions

| Hardware condition | Context | prefill (without qml) | restore (with qml) | qml advantage |
|---|---|---|---|---|
| Dev server | 8,000 | **0.64 s** | 0.81 s | None (prefill 1.27x better) |
| Dev server | 32,000 | 3.15 s | **2.72 s** | 1.15x |
| Orin power-limited | 8,000 | 13.81 s | **2.08 s** | **6.6x** |
| Orin power-limited | 32,000 | 88.89 s | **4.49 s** | **19.8x** |
| **Orin MAXN** | 8,000 | 4.34 s | **1.33 s** | **3.3x** |
| **Orin MAXN** | 32,000 | 23.89 s | **2.56 s** | **9.3x** |

The GPU computation on the qml side is 11 tokens (the new question) in all 6 measurements.

### Effect of MAXN on the two costs — the size of the improvement differs

| Context | prefill improvement | restore improvement |
|---|---|---|
| 8,000 | **3.18x** (13.81 s → 4.34 s) | 1.57x (2.08 s → 1.33 s) |
| 32,000 | **3.72x** (88.89 s → 23.89 s) | 1.75x (4.49 s → 2.56 s) |

Raising the power mode improves prefill much more, because GPU computation is the whole of the prefill cost.
restore also improved 1.6~1.8x, which means the CPU-side work remaining in restore (memory copy,
metadata lookup) benefited from more cores and a higher clock. Even so, **even in MAXN the
crossover point is below 8,000 tokens**, and qml is 3.3x better.

### Sensitivity of the two costs to a hardware change — verification of the structural claim

Splitting the performance difference of Orin MAXN relative to the dev server into the two costs:

| Cost | Dev server → Orin MAXN | Dependent resource |
|---|---|---|
| prefill | **6.8x slower** at 8K, **7.6x slower** at 32K (Orin MAXN prefill throughput: about 2,000 tokens per second at 8K, about 1,391 tokens per second at 32K / 667 and 372 tokens in power-limited mode) | GPU computation |
| restore | 1.64x slower at 8K, **at 32K Orin is actually 1.07x faster** (2.56 s vs 2.72 s) | I/O bandwidth |

The reason the edge device beat the dev server on 32K restore is simple — **Orin's NVMe (0.92 GB/s) is
1.7x faster than the dev server's disk cold read (0.54 GB/s).** This is the result that most directly shows the structural claim of
Sections 7-7 and 7-6 that restore depends not on GPU computation but on storage bandwidth.
Even though GPU computation performance differs by more than 7x, restore time is won by the side with faster storage.

### Implications

| Judgment | Basis |
|---|---|
| The previously recorded boundary "below about 30K tokens qml is at a disadvantage" is a **condition limited to high-performance dGPUs** | On Orin, qml is 3.3x better at 8K even in MAXN |
| Edge and small on-premises machines are qml's main target | GPU computation is the scarcest resource, and qml removes that cost |
| When citing a boundary value, **always state which GPU it was measured on** | With the same model and the same script, changing only the hardware moved the crossover point from the 30K range → below 8K |
| On edge devices, the choice of storage dominates restore performance | eMMC 0.27 GB/s vs NVMe 0.92 GB/s (Section 7-9); on NVMe it is faster than the dev server's disk |

### 3 portability problems found in the Orin build (reusable record)

The build was done without sudo privileges, and in the process problems that need to be reflected in the repository were found.

| # | Problem | Resolution |
|---|---|---|
| 1 | `find_package(OpenSSL)` is called first in llama.cpp's vendored subproject (cpp-httplib), so the `OpenSSL::Crypto` imported target is not visible in our directory scope | **CMakeLists.txt changed** — if the library is found but the target does not exist, create it from the variables (independent of build order) |
| 2 | Debian-family systems put `opensslconf.h` in an architecture-specific path (`include/aarch64-linux-gnu/openssl/`) | In a setup that extracts and uses the development packages directly, it must be linked into the standard include path |
| 3 | `libsqlite3.so` and `liblmdb.so` extracted from the development packages are symbolic links whose targets are in the runtime packages, so they become broken links | Relinked to the system's actual runtime files (`.so.0`, `.so.3`) |

The Orin-side procedure is in `tests/orin/`. The measurement script reads the power mode from `/var/lib/nvpmodel/status`
and writes it to the log (no sudo needed), and also records the number of online cores and the maximum CPU clock —
this time the power mode changed the results by more than 3x, so these are mandatory record items.

## 7-12. Same GPU and CPU, only the storage device differs — NVMe measurement and an error in the calculation model (2026-08-03)

To answer "what happens if NVMe is added to the dev server", restore time was first broken down into storage I/O and
CPU-side work and **calculated**. Then it was **measured** on the NVMe path of the same physical machine, and the calculation
model was found to be wrong and corrected. This section keeps the whole process.

### Comparison conditions — same silicon, only the storage device differs

| Item | Dev server | NVMe host |
|---|---|---|
| GPU | RTX 4090 — **GPU UUID `<gpu-uuid>`** | **Same UUID (same physical GPU)** |
| CPU / memory | i7-5820K 6 cores 12 threads / 62 GiB | Same |
| Relationship | A **container** running on this host (Ubuntu 22.04, glibc 2.35) | **Host** (Ubuntu 18.04, glibc 2.27) |
| Storage device | overlay file system (SATA SSD based), **sequential read 0.55 GB/s** (identical in all 3 runs) | **Samsung PM9A1 NVMe, PCIe 3.0 x4**, sequential read median **1.63~1.90 GB/s** |

The model file, scripts, KV settings (q4_0) and verification method are completely identical. **The GPU and CPU are literally the same
silicon, so the only variable in this comparison is the storage device.**

### Measurement defect found first — "cold" was not cold

In the first measurement on the NVMe host, cold restore came out at 8K 439 ms / 32K 1,178 ms, and the difference from warm
was only 34 ms / 176 ms. That amounts to reading 0.258 GB in 34 ms (7.6 GB/s) and 1.031 GB in 176 ms (5.9 GB/s),
which **is impossible with this device's measured bandwidth (1.6~1.9 GB/s)**.

Cause: `posix_fadvise(POSIX_FADV_DONTNEED)` **does not evict pages that are still dirty.** Right after the save
the pages were dirty, and the measurement labeled cold was actually served from RAM.

Measure — the measurement was made to prove by itself that it is cold (`tests/reconnect-restore/bench_long_context.py`):

| Measure | Description |
|---|---|
| fadvise after `os.sync()` | Dirty pages must be written back first before they can be evicted |
| **Measure and record resident bytes with `mincore()`** | Every cold measurement leaves `resident 246 MiB -> 0 MiB (cold confirmed)` in the log. If 8 MiB or more remains, **that measurement is marked invalid** |
| Bandwidth measured separately with a real file | The LMDB `data.mdb` is a sparse file with an apparent size of 64 GiB / 0.96 GiB of actual blocks. Reading the whole file is mostly hole reads that do not go through the device, so it **wrongly comes out at 2.7 GB/s** |

Re-measuring the dev server reproduced the earlier value (783/808 ms) as 797 ms even after verification — **the dev server
measurements were valid all along, and the dirty page problem occurred only on the NVMe host.**

### Measured results — two machines, cold verification passed, needle recall correct on every response

| Context | Machine | page cache verification | restore cold | restore warm (CPU-side lower bound) | stock prefill | Verdict |
|---|---|---|---|---|---|---|
| 8,000 | Dev server | 246 MiB → 0 MiB | 797 ms | 393 ms | 653 ms | prefill 1.22x better |
| 8,000 | **NVMe host** | 246 MiB → 0 MiB | **390 ms** | 379 ms | 596 ms | **qml 1.53x better** |
| 32,000 | Dev server | 986 MiB → 0 MiB | 2,809 ms | 1,042 ms | 3,209 ms | qml 1.14x better |
| 32,000 | **NVMe host** | 986 MiB → 0 MiB | **1,179 ms** | 984 ms | 3,021 ms | **qml 2.56x better** |

**Effect of NVMe: restore became 2.04x faster at 8K (797→390 ms) and 2.38x faster at 32K (2,809→1,179 ms).**
And **at 8K the crossover point is passed and qml wins** — on the dev server, prefill was 1.22x better.

Self-check of the measurement: since the GPU and CPU are the same, the CPU-side lower bound and stock prefill must match on the two
machines. 8K lower bound 393 vs 379 ms (4%), 32K 1,042 vs 984 ms (6%), stock prefill 8K 653 vs 596
ms (10%), 32K 3,209 vs 3,021 ms (6%) — they matched within a 4~10% difference that appears to be container overhead.

### The calculation model was wrong — the two costs do not add, they overlap

The original model was `restore = CPU-side lower bound + blob ÷ bandwidth`. Compared with the measurements:

| Context | Machine | Additive model | max model | **Measured** | Which model is right |
|---|---|---|---|---|---|
| 8,000 | Dev server | 862 ms | 469 ms | **797 ms** | additive |
| 8,000 | NVMe host | 515 ms | 379 ms | **390 ms** | **max** |
| 32,000 | Dev server | 2,916 ms | 1,874 ms | **2,809 ms** | additive |
| 32,000 | NVMe host | 1,616 ms | 984 ms | **1,179 ms** | **max** |

**On a slow device the two costs add; on a fast device only the slower one remains.** The blob is read as 4 MiB chunks
in parallel by 4 workers, so I/O and the CPU-side copy can run overlapped. If storage is fast enough, pages
arrive before they are needed and the I/O time is absorbed into the CPU work; if it is slow, every access waits and the costs add.
(The likely explanation is that when mmap page faults get ahead of readahead, every fault becomes a wait, but
**this was not confirmed by profiling.**)

Therefore `cold − warm` must not be called "I/O time". That value is only **the I/O that could not hide behind the
CPU work** (195 ms at NVMe 32K corresponds to 5.3 GB/s, while the device does 1.6 GB/s).

How this error changed the conclusion — the model was wrong **in a direction unfavorable to qml**:

| Context | Model prediction (additive) | Measured | Model error |
|---|---|---|---|
| 8,000 | 528 ms (at 1.74 GB/s) | **390 ms** | measured 1.35x better |
| 32,000 | 1,700 ms | **1,179 ms** | measured 1.44x better |

The model also said "an improvement of 2.06x at 8K / 2.49x at 32K is the upper bound, reachable only with infinitely fast storage", but
**in reality 2.04x / 2.38x was already reached at 1.6~1.9 GB/s.** The upper bound itself was right, but the bandwidth
needed to reach it was greatly overestimated.

### Conclusion

| Judgment | Basis |
|---|---|
| **Adding NVMe makes restore 2.0~2.4x faster** | 8K 797→390 ms, 32K 2,809→1,179 ms (same GPU and CPU) |
| **On NVMe, qml wins even at 8K (1.53x)** | restore 390 ms vs prefill 596 ms — the crossover point moves below 8K |
| The reason 8K was unfavorable to qml on the dev server was the **storage device** | The GPU and CPU are the same and only storage changed, yet the verdict reversed. Section 7-11 separated the GPU factor, and this section separated the storage factor |
| **The next bottleneck is CPU-side work** | On NVMe, restore time is almost at the CPU-side lower bound (8K 379 / 32K 984 ms). Faster storage cannot bring it below this value |
| 32K CPU-side lower bound 984 ms = about 1 GiB per second to process 0.96 GiB | It is low compared with pure memcpy speed (several GiB per second), so there seems to be room for improvement, but **no profiling was done** |

### Portability problems found in the NVMe host build (Ubuntu 18.04 / glibc 2.27)

| # | Problem | Measure |
|---|---|---|
| 1 | The system gcc 7.5 has no `<filesystem>` (used by llama.cpp's `common.cpp` and `server-tools.cpp`) | Set up a conda-forge gcc/gxx 11 toolchain in `/nvme/qml-router/env`. The latest Miniconda requires glibc ≥ 2.28, so version 23.5.2 was installed instead, and the environment was created with micromamba |
| 2 | `vendor/sheredom/subprocess.h` calls **`posix_spawn_file_actions_addchdir_np()`** — a function first introduced in glibc 2.29, so compilation fails. `llama-server` always links `mtmd`, so it cannot be turned off | **Added a glibc version guard** — on 2.29 and later the existing code runs unchanged, and below that it returns `ENOSYS` at runtime. No regression confirmed on glibc 2.35. Also applies to RHEL/CentOS 7 (glibc 2.17) |
| 3 | When the measurement script exported the conda env's `lib` through the global `LD_LIBRARY_PATH`, **the system `curl` broke**, the health check failed, and a server that was running normally was reported as "SERVER FAILED" | Limited the library path to the llama-server invocation only (`env LD_LIBRARY_PATH=... llama-server`) |

The scripts are in `tests/orin/build_nvme_host.sh` and `tests/orin/run_nvme_host_test.sh`, and the raw logs are in
`tests/orin/results/`.

## 7-13. Profiling the CPU-side lower bound — the cause is a checkpoint that upstream would not have created (2026-08-04)

The question left by Section 7-12 ("where does the CPU-side lower bound of 984~1,042 ms go?") was settled by step-by-step instrumentation of the restore
path, and the cause was fixed. **32K warm restore went from 1,042 ms → 537 ms, and needle recall stayed
correct on every response.**

### Instrumentation (one log line per restore, permanently in the code)

| Log | Location | What it measures |
|---|---|---|
| `[qml][timing] load` | core/qml_session.c | SQLite metadata lookup vs LMDB blob read |
| `[qml][timing] fetch` | hook fetch worker | Whole blob load (background thread) |
| `[qml][timing] apply` | hook state_apply | `set_data` (loop thread) |
| `[qml][timing] capture` | hook enqueue_save | State capture for saving, `get_data` (every turn, loop thread) |
| bench `server compute:` line | server response timings | Separation of model computation (prompt/predict) from the non-computation remainder |

### Breakdown (32K warm restore, before the fix)

| Segment | Time | Verdict |
|---|---|---|
| SQLite metadata lookup | 0.2 ms | Negligible |
| LMDB blob read (0.96 GiB, page cache) | ~200 ms | Normal |
| `set_data` | ~120 ms | Normal (matches the earlier measured 0.17 s/GB) |
| **`create_checkpoint` — recapture of the state just applied** | **~502 ms** | **Confirmed as waste** |
| 11-token prefill + 12-token generation | ~160 ms | Normal |
| (Separate) save capture `get_data` at release every turn | ~530 ms | Not included in the response; consecutive requests wait this long — next optimization target |

How the unexplained segment was identified: a `hot` mode was added to the bench that sends a second question to the same server (the session is already in the slot, so no
restore) — the hot prompt is 28~104 ms while the prompt right after restore is in the 700 ms range
→ this established that the difference is inside the restore path → instrumentation was added to `tokens.insert` (0.0 ms) and `create_checkpoint` (502 ms)
to name it.

### Why it is waste — the upstream source is the basis

The condition under which upstream llama.cpp creates a checkpoint (do_checkpoint in update_slots):
it creates one **only for models where partial `seq_rm` is impossible (FULL/RS) or for SWA models**. Ordinary non-SWA
causal models can be truncated at any point with `llama_memory_seq_rm`, so it neither creates nor
reads checkpoints. The guard in our restore path was `n_swa == 0` (create only on non-SWA), **the exact opposite** —
it was spending 502 ms on every restore to create a checkpoint that upstream would not create.

### One rejected fix (recorded)

A method that copies the blob into the checkpoint to avoid the recapture was implemented first, but **at 553 ms it was the same as the recapture.**
On this CPU (i7-5820K), copying 1 GiB into new pages is bound by page faults at ~2 GB/s (measured 493~524 ms).
The wall is the copy itself, not the source of the copy. → It was discarded, and the creation condition itself was
replaced with the upstream condition (in all 3 restore paths).

### Measurements after the fix (dev server, cold verification passed, every response correct)

| Context | Measurement | Before fix | **After fix** | prefill under the same conditions | Verdict |
|---|---|---|---|---|---|
| 8,000 | restore cold | 797 ms | 804 ms (cold is dominated by disk reads, so no change) | 661 ms | prefill 1.22x better (slow storage only) |
| 8,000 | restore warm (**CPU-side lower bound**) | 393 ms | **245 ms** | — | — |
| 32,000 | restore cold | 2,809 ms | **2,240 ms** | 3,159 ms | **qml 1.41x better** (1.14x before the fix) |
| 32,000 | restore warm (**CPU-side lower bound**) | 1,042 ms | **537 ms (1.9x shorter)** | — | — |

Converted to the NVMe environment (max-behavior model of Section 7-12): 32K restore is expected to drop to around the CPU-side lower bound of 537 ms,
**about 4~5x better than prefill at 3,021 ms** — to be confirmed by re-measurement.

Note: the restore numbers in Sections 7-6, 7-11 and 7-12 are values from **before** this fix. Under the rule against deleting measured rows they are
left as they are, and the new numbers are used only in this section and later measurements.

### Remaining optimization target (not started)

The save capture of ~530 ms at release every turn is now the largest single item. It was confirmed that this llama.cpp version has the
`LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` (capture within the GPU) flag — a candidate for reducing the device-to-host
path, but **not verified**.

## 7-14. NVMe re-measurement after the checkpoint fix — qml 4.6x better at 32K, prediction confirmed (2026-08-04)

The fix of Section 7-13 was applied on the NVMe host and re-measured. Conditions are identical to Section 7-12 (same physical GPU and CPU,
only the storage device is NVMe, cold verified with mincore, needle recall correct on every response).

### Results — all four conditions (3B F16, KV q4_0, same GPU silicon)

| Condition | 8K restore cold | 8K prefill | 8K verdict | 32K restore cold | 32K prefill | 32K verdict |
|---|---|---|---|---|---|---|
| Dev server (0.55 GB/s), before fix | 797 ms | 653 ms | prefill 1.22x | 2,809 ms | 3,209 ms | qml 1.14x |
| Dev server, after fix | 804 ms | 661 ms | prefill 1.22x | 2,240 ms | 3,159 ms | qml 1.41x |
| NVMe (1.6~1.9 GB/s), before fix | 390 ms | 596 ms | qml 1.53x | 1,179 ms | 3,021 ms | qml 2.56x |
| **NVMe, after fix** | **275 ms** | 580 ms | **qml 2.11x** | **661 ms** | 3,033 ms | **qml 4.59x** |

warm (CPU-side lower bound): 8K 223 ms / 32K 604 ms — the difference from cold is only 52/57 ms, so on NVMe
the storage read hides almost completely behind the CPU-side work (exactly the max behavior of Section 7-12).

### Prediction check — the model was right at a second hardware point as well

Compared with the prediction calculated with the max-behavior model in Section 7-13:

| Context | Prediction | Measured | Error |
|---|---|---|---|
| 8,000 | ~250 ms (qml 2.4x) | 275 ms (qml 2.11x) | 10% |
| 32,000 | ~650 ms (qml 4.6x) | 661 ms (qml 4.59x) | **2%** |

### Cumulative effect — how much this campaign reduced restore

| Step | 32K restore cold | Means of improvement |
|---|---|---|
| Starting point (dev server, slow disk) | 2,809 ms | — |
| + NVMe (storage device replaced) | 1,179 ms | Hardware |
| + checkpoint fix (software) | **661 ms** | Guard replaced with the upstream condition |

In total **4.2x shorter**, and the verdict reversed from prefill being better at 8K → **qml 2.1x better even at 8K**. The remaining lower bound is
the CPU-side 604 ms, and its largest item is the save capture every turn (~530 ms, Section 7-13) — the next optimization target.

## 7-15. Speculative PREWARM demonstration — if the event comes 2 seconds earlier, a 32K reconnect takes 248 ms (2026-08-04)

We measured how much restore latency is hidden if restore is started before the actual request, using a client event (opening the chat window, starting to type) as the
signal. Both steps used **control commands that were already implemented** as they are —
INSERTing a PENDING row into the `qml_control_cmd` table wakes the server immediately through inotify (external contract).

| Step | Command | What it does | Where it runs |
|---|---|---|---|
| 1 | `PREFETCH_RAM` | Warms disk → RAM (page cache) | persister thread (does not touch VRAM or the model loop) |
| 2 | `PREWARM` | Full restore into an idle slot (fetch+apply) | Safe point inside update_slots on the model thread |

### Measurement (dev server = slow disk 0.55 GB/s, 32K session, server restart + cold verification per scenario, every response correct)

| lead (interval from event to question) | No preparation | PREFETCH_RAM | PREWARM |
|---|---|---|---|
| — | **2,231 / 2,261 ms** | — | — |
| 0.5 s | — | 1,766 ms | 1,949 ms |
| 1 s | — | 1,326 ms | 1,473 ms |
| 2 s | — | **528 ms (4.2x)** | **248 ms (9.0x)** |
| 5 s | — | 507 ms | **186 ms (12x)** |

Verdict:

| Fact | Basis |
|---|---|
| When the lead exceeds the disk read (~1.7 s), step 1 hides the whole I/O part | From 2 s on, at the level of the warm lower bound (537 ms) |
| When the lead exceeds the whole restore (~2.2 s), step 2 also hides the CPU-side work | 248 ms at 2 s, 186 ms at 5 s — the level of a hot request |
| When the lead is too short, the gain is partial | 1.2~1.3x even at 0.5 s |
| On NVMe deployments the gain of step 1 is already known to be small (Section 7-12) | cold−warm is only 52~57 ms — **this is a feature for slow-storage deployments** |

Operational recommendation: fire PREWARM on the chat UI's session-list click / start-of-typing event, and if there is no idle slot,
downgrade to PREFETCH_RAM (PREFETCH_RAM uses no slot or VRAM at all, so it is always safe).

### Product defect found and fixed at the same time — the session lock of a dead process remained forever

In the first run, PREWARM succeeded only the first time and then all FAILED. Cause: when the owner of the session lock
(`qml_session.owner_pid`) died from `kill -9`, the lock remained, so **every later control command
restore (RESTORE/PREWARM) was rejected forever** (measured: LOCKED remained with owner_pid=10211 dead).
The request-path restore does not go through the lock, so this was not caught by any measurement until now.

Fix (`storage/sqlite/qml_sqlite_store.c`): on a lock conflict, check whether the owner is alive with `kill(pid, 0)` and
**take over only the lock of a dead owner (ESRCH)**. EPERM (alive but another user) is kept as a legitimate
conflict. The WHERE clause of the takeover UPDATE rechecks the existing owner pid, so it is safe even in a race with a third
process. It is a same-host design in which all qml processes share one embedded SQLite file, so
checking whether the pid is alive is valid. In the rerun after the fix, PREWARM worked correctly all 4 times.

## 7-16. Re-estimating head-selective storage — valid only on the edge (slow storage) (2026-08-04, analysis, not implemented)

The expected gain of the DuoAttention (Xiao et al. 2024) approach (store full KV only for retrieval heads) was re-estimated with measured
numbers. **It was not implemented; what follows is a calculation.**

Premise: the behavior established in Sections 7-12 and 7-14 — on fast storage restore ≈ max(CPU-side lower bound, I/O), so
shrinking the blob has almost no effect; on slow storage it is additive, so it shrinks by the I/O part. A 3x blob reduction is assumed
(depends on the retrieval head ratio, not verified).

| Storage (32K blob 1.03 GB) | I/O part | I/O at 3x reduction | Expected restore improvement |
|---|---|---|---|
| Dev server 0.55 GB/s | ~1,700 ms | ~570 ms | 2,240 → about 1,100 ms (2.0x) |
| Orin NVMe 0.92 GB/s | ~1,120 ms | ~370 ms | Significant (additive region) |
| **Orin eMMC 0.27 GB/s** | **~3,800 ms** | **~1,270 ms** | **Largest — the real target of this feature** |
| NVMe host 1.6~1.9 GB/s | Hidden (max behavior) | Hidden | **Almost 0 — not a target** |

Conditions to start (it is not implemented until all are met): (1) there is an actual deployment requirement that uses eMMC-class storage,
(2) an accuracy gate is designed — at least on par with the INT8 verification (needles at several depths and several prompts + greedy generation
match comparison), (3) the cost of classifying retrieval heads per model (incurred for every model added) is accepted. Its priority is
lower than the ~530 ms save capture every turn (Section 7-13, ON_DEVICE candidate) — because that one has an
effect on every turn on every hardware.

## 7-17. Two VRAM measurements — qml's VRAM overhead is 0, answers stay correct even when the pool is exceeded (2026-08-04)

Two VRAM-related questions that were open in the deployment review were measured on the NVMe host (3B F16, KV q4_0,
`-c 65536 -np 8 --kv-unified`).

### A. qml's own VRAM overhead

| Configuration | VRAM after startup |
|---|---|
| Without qml (same flags) | 8,979 MiB |
| With qml (`--qml-session-store` added) | **8,979 MiB — difference 0** |

qml's save and restore buffers are all in system RAM and on disk, so this is the value expected from the structure, but now
there is measured evidence. **Deployment wording: "qml uses no additional VRAM (measured 0 MiB)."**

### B. Operating beyond the unified pool — every response correct even at 1.5x the pool

Six 16K sessions (96K in total) were created in sequence on a 64K-token shared KV pool, and then only a new question was sent to all 6
(revisit). Measurement of the current behavior for "what happens when the pool is full", asked by step 4 B of the handover:

| Step | Result |
|---|---|
| 6 creations | **All succeeded, all correct, 0 rejections or errors** — VRAM stable at 9,001→9,017 MiB (the pool is reserved at startup, so it does not depend on the number of sessions) |
| Revisit — 3 resident | **0.4~0.5 s, prefill 20 tokens** (sessions that remained in the pool) |
| Revisit — 3 evicted | **4.0~4.4 s, prefill 16,041 tokens** — transparently rebuilt by replaying the token log, **answers stay correct** |

Three key verdicts:

1. **Exceeding the pool is safe.** No request failed or gave a wrong answer. The cost of an evicted session is
   only one recompute (about 4.4 s at 16K).
2. **This safety is provided by qml.** The revisiting client sent only the new question — with a stock server
   without qml, an evicted session would have answered wrongly without history (the class of defect measured in the past), or the client would have had to
   resend the full history. qml's token log prepend transparently recovered the evicted sessions.
3. It is recorded that the evicted sessions were recovered by the **recompute path**, not by a blob restore (11~20-token prefill)
   — with the pool full, `set_data` appears to have had no free cells to use, so the restore failed and was
   downgraded to recompute, but **the cause was not confirmed by profiling.** To get the full restore gain in configurations with heavy
   residency contention, the eviction/admission rules (step 4 C of the handover) must handle this point.

That is, step 4 C (queueing and size-aware assignment) is reclassified as **a performance optimization, not a requirement for safety**
— because the current behavior already guarantees correctness.

## 7-18. Comparison of the first-level caching effect across 3 tiers + capacity of fixed partition vs unified pool (2026-08-05)

Measurements for two user requests. Environment: standard (NVMe host), 3B F16, KV q4_0, needle recall
verified on every response.

### Test 1 — how long the same reconnect question takes depending on "where the memory is"

Three 32K sessions were created, and the same question ("What was the password?") was repeated in three states.

| State | Where the memory is | Response time (3 sessions) | GPU computation |
|---|---|---|---|
| (1) Resident reconnect (first-level cache hit) | As is in GPU memory | **259 / 736 / 725 ms** | 19 tokens |
| (2) Restore reconnect (disk → GPU) | Only on disk (restart + cache cleared) | 797 / 1,362 / 1,480 ms | 19 tokens |
| (3) Recompute (without qml) | None — full history resent | 3,031 / 2,880 / 2,883 ms | 28,809~32,018 tokens |

How to read it: (1) is the value of first-level caching (**4~11x** compared with recompute), (2) is the safety net when the resident copy has been evicted
(2~3.8x compared with recompute), and (3) is the world without qml. Sessions 1 and 2 being slower than session 0 is because they overlapped with waiting while the
state save work of the previous request (the ~0.5 s capture every turn of Section 7-13) occupied the loop — this is reported as is, as the actual perceived value under
consecutive requests. In (3), the GPU computation dropping to 28,809 for sessions 1 and 2 is because the same slot reused the common
preamble across sessions (about 3,200 tokens before the needle); this is simply a property of running without qml.

### Test 2 — 8 sessions of mixed sizes: does the unified pool do what the fixed partition cannot?

8 sessions (300×4, 1,500×2, 8,000×1, 16,000×1 = about 27K tokens in total) were run on two configurations with the same VRAM budget
(65,536 tokens' worth). This is also the measurement for step 4 B of the handover ("does capacity increase with mixed sizes?").

| | Configuration A: fixed partition (-np 8, 8,192 tokens per slot) | Configuration B: unified pool (--kv-unified) |
|---|---|---|
| Admission | **7/8 — 16,000-token session rejected** (larger than the slot, explicit rejection with HTTP 400) | **All 8/8 admitted** |
| Reservation waste | The actual 12,342 tokens used by the 7 admitted sessions occupy 57,344 tokens of slot reservation — **waste rate 78.5%** | No reservation — only the 28,352 tokens used are occupied, the rest is free space |
| Reconnect accuracy | 7/7 (a reconnect to the rejected session gives a wrong answer because nothing was saved — see note below) | **8/8** |
| Reconnect speed | 112~145 ms | 144~278 ms (small overhead of the unified pool, +20~30 ms) |

Note: in A, if a reconnect is attempted to session 7, whose creation was rejected, nothing is saved, so it answers without history and
gives a wrong answer — whether to reject reconnects to a session that does not exist is left as an operational policy item (the server cannot know
that session, so it currently handles it as an ordinary request).

Verdict: **the unified pool resolved, by measurement, the two problems of the fixed partition (rejection of large sessions, small sessions occupying an entire slot),
at a cost of +20~30 ms on reconnect.** The operational recommendation (Section 7-10) to adopt unified by default is supported by the numbers.
The waste rate of 78.5% is far above the decision criterion required by handover task (1) (over 30% means making it variable is well
worth it).

## 7-19. Sustained mixed load: fixed partition vs unified pool + a defect where sessionless requests destroyed sessions (2026-08-05)

A measurement that closes the paper's unmeasured gap ("geometry comparison under sustained load"). What is the same: standard environment,
3B F16, KV q4_0, `-c 65536 -np 8 --cache-ram 0`, both configurations llama.cpp+qml, same load.
What differs: only the KV geometry (`-np 8` fixed partition vs `--kv-unified` shared pool).
Load: after creating 12 sessions (300×4, 1,500×4, 6,000×3, 7,500×1 ≈ 33K tokens, a size both geometries can
admit), for 150 seconds 4 revisit workers (send only the question, serial per session — following the rule of no concurrent turns
on the same session) + 2 workers of short requests **without session_id** ("other users").
needle verification on every response. Scripts `tests/bench_geometry_sustained.py`,
`tests/orin/run_geometry_ab.sh`, raw logs `tests/orin/results/results_geometry_ab_*.log` (internal scripts and logs, not published).

### Product defect found first — a sessionless request overwrote the slot's session record (P5 violation)

Results of the first run (recorded as is, under the rule against deleting measured rows):

| First run (before fix) | A: fixed partition | B: shared pool |
|---|---|---|
| Revisits completed / **wrong answers** | 840 / **145** | 199 / **24** |
| Short requests completed / wrong answers | 450 / 0 | 106 / 0 |

The wrong answers were **all concentrated in sessions 4 and 5** (practically every revisit of those two sessions was wrong), and the content of the wrong answers was
always the needle of a sessionless short request (`TK-...`). Comparing with the server log: the saved data of sessions 4 and 5 had been
replaced by **57 tokens** (the size of one short request), and every later revisit answered with that foreign content
prepended.

Cause (established from the source): `launch_slot_with_task` kept the slot's existing session binding for a request **without**
a session_id (a convenience rule: "omitting it keeps the binding"). When a sessionless request reused that
slot, the capture at release **saved the content of the sessionless request under the previous session's name**
— the blob and token log were replaced entirely by foreign content. Slot assignment is based on similarity and LRU, so
there is no guarantee at all that "it comes to the same slot", so this convenience rule was a contract that never held in the first place.

Fix (`server-context.cpp`): the binding is **always reassigned** — when session_id is empty it is released, so
sessionless requests do not go through the save path. Rerun after the fix: **0 wrong answers in both configurations**
(A 826+424, B 188+95, all correct). Why earlier measurements did not catch it: the short requests of the existing load scripts
all had their own session_id — the mix of traffic without session_id and session
traffic was measured for the first time here.

### geometry results (after the fix, numbers with 0 wrong answers)

| Completed within the 150-second window | A: llama.cpp+qml fixed partition | B: llama.cpp+qml shared pool |
|---|---|---|
| Revisits completed | **826** (p50 707 / p90 882 ms) | 188 (p50 3,170 / p90 5,522 ms) |
| Short requests completed | **424** (p50 703 ms) | 95 (p50 3,280 ms) |
| Wrong answers | 0 | 0 |
| Rejections (HTTP 400) | **81** — all explicit rejections after the 7,500-token session grew past the 8,192 slot through accumulated revisits | **0** |
| Admitting a 16K session after the window ends | Rejected | **Admitted, correct (1.8 s)** |

### Cause breakdown — set_data of the shared pool explodes under churn

Comparison of the server's permanent instrumentation ([qml][timing]). 12 sessions rotate over 8 slots, so every revisit causes a slot
swap (eviction capture + restore):

| Stage (sum over 150 s) | A: fixed partition | B: shared pool |
|---|---|---|
| apply (`set_data`) | p50 **16 ms**, total 10.4 s | p50 **554 ms**, total **120.4 s** |
| capture | p50 52 ms, total 55.7 s | p50 64 ms, total 17.1 s |
| fetch | p50 20 ms, total 13.9 s | p50 14 ms, total 2.8 s |

For sessions of the same size, `set_data` on the shared pool is **about 35x** slower — 120 of the 150 seconds are
spent on apply in the loop, which causes the 4.4x throughput difference. The cost of writing state back into the shared pool appears to be
sensitive to the pool occupancy state (fragmentation), but **the cause inside llama.cpp was not
confirmed by profiling** (next optimization target).

### Verdict — conditional downgrade of the operational recommendation in Section 7-10

| Workload | Recommended geometry | Basis |
|---|---|---|
| Mixed-size, growing sessions, admission comes first | **Shared pool** | 16K admitted, 0 rejections, 0 reservation waste (7-18) |
| Fixed size, high-frequency revisit churn | **Fixed partition** | 4.4x sustained throughput, p50 707 vs 3,170 ms. However, explicit rejections occur when a session grows past the slot (81) — headroom for growth is mandatory (configuration rule 3) |

"Always `--kv-unified`" (7-10) is withdrawn by this measurement and replaced by the table above. Re-evaluate when the shared pool
`set_data` cost is resolved.

## 7-20. template-identity shared prefix — demonstration of faster first visits for new sessions (2026-08-05)

Implementation and demonstration of a feature designed as a safe alternative to the proposal to remove session boundaries (content-based matching).
A `template_id` is added to the request: **only when there is no saved record for the session_id**, the template session's
token log is prepended and its blob is restored, and the release save is then recorded **under its own session_id**
(happens once per session). Because the name is specified explicitly, there is no content-matching step, so the contamination class of errors
cannot occur by construction. Implementation: `server-task.h` (+`template_id`),
`server-schema.cpp` (request field), `server-context.cpp` (peek branch in launch +
restore key separated with `qml_restore_key`).

Conditions: standard environment, 3B F16, KV q4_0, `--kv-unified -np 8 --cache-ram 0`. template =
4,000 tokens (planted fact ZK-8800). needle verification on every response, cold verified with mincore (0.0 MiB).

| Method for a new session's first visit | Response time | GPU prefill | Correct |
|---|---|---|---|
| A. No handling — the client resends the whole template body | 362 ms | **4,004 tokens** | OK |
| B. template seeding, store warm | **181~221 ms** | **11 tokens** | OK |
| C. template seeding, store cold (verification passed) | **196 ms** | **11 tokens** | OK |

Isolation check: user1 and user2 were seeded from the same template and given different new facts (TK-7001/7002)
→ on revisit after a restart, each answered only the template fact + its own fact, and **the absence of the other's fact was
confirmed**. Measured that template_id is ignored for a session that already has a record (its own record is used from the 2nd turn on).

Verdict: on a new session's first visit, GPU prefill is **1/364** (4,004→11), and response is 1.7~1.8x faster.
Difference from the engine's own prefix cache: this gain **persists across restarts** (C at 196 ms when cold), and
there is no upper limit on the number of templates as long as the store can hold them. Scripts `tests/bench_template_seed.py`,
`tests/orin/run_template_seed.sh`, log `tests/orin/results/results_template_seed.log` (internal scripts and log, not published).

## 8. Honest positioning

qml is **not faster or more accurate than stock llama.cpp used as designed.** `--slot-save-path`
+ `/slots` is designed as a prefill cache where "the client sends the full context and the server skips
recompute", and in that usage it is accurate and fast (147 ms).

The value of qml is the **operating model**: the client sends only the new turn, the session is referred to by
`session_id` rather than by file name + slot, there is user/session metadata, sessions can be handed over between engines, and history that the engine discards
is preserved.
