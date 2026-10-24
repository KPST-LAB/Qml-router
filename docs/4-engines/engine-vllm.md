# Per-engine implementation: vLLM

One of the three engine documents. [engine-llama-cpp.md](engine-llama-cpp.md) · [engine-sglang.md](engine-sglang.md)

As of 2026-07-31, vLLM 0.26.0. Every number here is measured, and the conditions are stated with it.

---

## 1. Extension method — 0 source patches

An external module is imported at runtime with `KVConnectorBase_V1` + `kv_connector_module_path`.
Searching the installed `vllm` package tree for the string `qml` returns **0 matches** — no vLLM source was
modified by our work.

```
--kv-transfer-config '{
  "kv_connector": "QmlVllmConnector",
  "kv_connector_module_path": "qml_vllm_connector",
  "kv_role": "kv_both",
  "kv_buffer_device": "cpu",
  "kv_connector_extra_config": {
    "qml_sqlite_path": "...", "qml_lmdb_path": "...",
    "qml_self_url": "http://127.0.0.1:8210",
    "qml_read_threads": "8"
  }
}'
```

There are only two implementation files: `hooks/vllm/qml_vllm_connector.py` (connector),
`hooks/vllm/qml_store_bindings.py` (ctypes bindings to the C storage).

## 2. Semantics imposed by the engine — there is no session continuation

The question the connector receives is not "restore this session" but **"how many more tokens of this prompt's prefix
can you provide"** (`get_num_new_matched_tokens`). Therefore:

- The client **always sends the full context.** vLLM has nothing that corresponds to llama.cpp's auto-continue ("send only the new turn").
- The only performance comparison axis is **with qml vs. without qml**, and client behavior is identical on both sides.
- `MIGRATE` (external eviction) is structurally impossible — there is no eviction handle. Only inducing LRU is possible.
  **(Formally accepted 2026-08-02: confirmed as not implementable until the engine provides a handle. Classified as an "accepted limitation"
  in the remaining issue list, to be revisited if an eviction API appears upstream in vLLM.)**
- **Automatic storage format detection (2026-08-02, improved at the user's request)**: the first implementation added an
  `@q8` suffix to the key name to distinguish the format, but then every existing FP16 store misses.
  Fix: removed the suffix and **detect the format (FP16/INT8) at read time from the stored size of the block**, loading
  each range in its format (the two formats of the same layer differ in size, so the storage layer's size
  check acts as the discriminator). Verification: a server with compression on restored as is a session (FP16) saved by a server with compression off
  (answer correct), and in a mixed store (FP16 session + INT8 session) both restored correctly after restart.
  The faster restore on the INT8 side (337 vs 541 ms) is also consistent with half the bytes.
- **Loss verification of INT8 storage compression (2026-08-02)**: the accurate description is "lossless at the task and generation level,
  slight loss at the numeric level (~0.2-0.4% rounding error per value)". Measured evidence: (1) needle recall (a task of finding and answering with information planted in the context) all 514
  correct, (2) on the same 32K context, a greedy 128-token continuation with computed KV and with INT8 round-trip KV
  **matches exactly** (95/95 words, no divergence — note that the sample is 1 prompt). Why it differs from vLLM's FP8 KV
  mode (10.4% wrong): with FP8 the attention computation itself is 8-bit, so error accumulates, whereas
  this compression quantizes only once per storage round trip and the computation is FP16.
- **user_id passing convention (added 2026-08-02)**: vLLM's Request does not pass user fields through to the connector,
  so if the client gives `cache_salt` in the form `"userID::sessionID"`, the
  connector records the part before `::` as the user_id of qml_session. The whole salt is still used as the
  session and isolation key, so KV matching behavior does not change, and a salt without `::` behaves the same as
  before (backward compatible). Execution verified (2026-08-02): salt `alpha::sess-9` → qml_session.user_id `'alpha'` recorded, salt without the convention `plain-sess-7` → empty value — both cases correct.
- `RESTORE`/`PREWARM` are implemented as a workaround with `cache_salt` + `max_tokens=1` self-completion
  (every entry point of KVCacheManager requires a live `Request` object, so there is no lower-level shortcut).

The session identifier uses vLLM's own `cache_salt`. However, `NewRequestData` does not have this field, so it is
captured in `on_new_request()`, where the full `Request` is visible, and consumed later (a bug we actually hit).

## 3. Key design

| Item | Content |
|---|---|
| cache seed (#53) | model id/revision, model dtype, KV cache dtype, TP/PP, block_size, number of layers, num_kv_heads, head_size + fingerprint of the local checkpoint files |
| block chain (#28) | `chain[i] = sha256(chain[i-1] | block_i tokens)`, the seed is the cache seed. **Partial prefix reuse** comes from this |
| sharing scope (#54) | `qml_share_scope` = `none` (default, session_id included in the seed) / `tenant` / `global`. global brings back a cross-session timing side channel, so it is an explicit opt-in |
| manifest | Per-block sentinel keys. "How many prefix blocks are cached" is decided with these keys |
| safeguard | On load, if the blob byte length differs from the expected value for that layer, stop at that point (treated as a cache miss; no garbage injection) |

Multimodal requests are skipped entirely — hashing only the placeholder tokens makes different images collide in the hash and
**silently serves KV of the wrong image** (#32).

## 4. Measurement — resume TTFT

`google/gemma-3-4b-it` bf16 · `--max-model-len 8192 --enforce-eager` · 6,617 tokens (needle
`QX-4471` in the middle) · block_size 16 · store 935 MB (141 KB/token, 14,455 blocks) · 1 request after restart ·
streaming TTFT · saving excluded from measurement

| Condition | Without qml | With qml (per block) | With qml (batched) |
|---|---|---|---|
| cold — first request after restart | 737 ms | **7,612 ms** | 741 ms |
| warmed — after 1 short unrelated request | 671 ms | — | **592 ms** |

Every row matches the correct answer. In the qml rows, 6,608/6,617 tokens are supplied from the external store each time
(`vllm:prompt_tokens_by_source{source="external_kv_transfer"}`); in the rows without qml, 0.

`--warmup` is a benchmark option. The first request of a fresh process pays one-time costs such as kernel compilation;
both sides pay it, so mixing the two hides the difference — **turn it on for both sides or off for both.**

### Break-even

This GPU prefills at 6,617 tok / 0.737 s ≈ **9,000 tok/s**, and gemma-3-4b bf16 KV is
**141 KB/token**. Therefore, to tie with recomputation, **141 KB × 9,000 = 1.27 GB/s** must be sustained all the way to the GPU.
The first implementation was 125 MiB/s, 10x short, and it is now 3.0 GB/s.

llama.cpp puts the same store in with **1 call** of `llama_state_seq_set_data` at 6.5 GB/s — this is
the upper bound for this path, and a gap that cannot be narrowed as long as the vLLM connector API works in block×layer units.

## 5. Batched I/O (task #65) — the most important part of this document

The problem was not the storage medium but **the number of round trips**. Per block and per layer: 1 LMDB read txn +
1 SQLite recency UPDATE + 2 host copies + 1 H2D transfer of 64 KiB → 14,042 of each.

| Action | Implementation | Effect |
|---|---|---|
| Batched read | `qml_lmdb_get_many()` — 1 txn per layer, direct memcpy into a pinned memory buffer | |
| Batched H2D | 1 transfer of 26 MiB per layer + 1 `index_put` | to-GPU 3,037 → 113 ms |
| Manifest scan | `qml_lmdb_exists_run()` with 1 txn, once per request (`_match_cache` passed from the scheduler hook → meta build) | 101+93 → 14+0 ms |
| Read parallelization | `qml_read_threads` (default 4). ctypes releases the GIL, so mmap page faults are actually parallelized | |
| **LRU rank recording** | Batched (`qml_sqlite_block_touch_many`, 1 transaction) + background writer | read 377 → 166 ms |

```
before:     7,013 ms (store read 3,756 + to-GPU 3,037) 14,042 block-layers ->   125 MiB/s
after:        293 ms (store read   166 + to-GPU   113) same bytes          -> 3,035 MiB/s
```

The LRU item was unexpected — the bookkeeping for cache reclamation cost more than the reads themselves. For reference, vLLM itself
**does not record** recency on each hit (`BlockPool.touch()` only removes the block from the free queue, and
order is expressed only by position in an intrusive doubly linked list). Our tier has to survive process restarts,
so it needs timestamps; the right conclusion was "move it off the critical path", not "remove it".

### The save path had the same problem

`block_put()` issued 1 LMDB write txn + 1 SQLite commit per block, and did so inside the forward pass.
Now: `qml_lmdb_put_many()` (1 write txn per layer) + one extraction per layer
(1 D2H, written directly from the host tensor's `data_ptr()` with the numpy detour removed) + manifest 414 entries → 1.

| | Cumulative `save_kv_layer` (1 request) | LMDB write txn | SQLite commits |
|---|---|---|---|
| Per block (`QML_SAVE_BATCH=0`) | **5,767 ms** | 14,456 | 14,042 |
| Batched | **616 ms** | 35 | 34 (background) |

Because the byte-producing path changed, the round trip was verified: after restart 6,608 tokens supplied + correct answer,
TTFT 595 ms (same as 592 before the change), same block and token log counts. This cost does not show up in resume TTFT, but
**every request pays it.**

Instrumentation is enabled with `QML_LOAD_TIMING=1` (default off) and reports match/meta/load/save separately.
The previous save path is kept as `QML_SAVE_BATCH=0` for comparison.

## 6. Bugs fixed

| # | Symptom | Cause |
|---|---|---|
| #29 | `_block_token_bytes` range error | bytearray slicing |
| #30 | Tensor reshape / device mismatch | The layer shape is `(pages, kv_heads, page_size, head_dim)`, not `(pages, 2, ...)` as in the example comment |
| #32 | Multimodal hash collision → wrong image KV | Section 3 |
| #64 | **EngineCore dies on an 8K prompt** | Chunked prefill. The chain covers the whole prompt, but `block_ids` holds only this step's allocation → `IndexError`. `scheduled_cached_reqs` (second and later chunks) were not handled at all, so even if the crash were prevented, only the first chunk would have been saved. `_store_progress` + `_emit_store_chunk` now save **only blocks whose computation has finished**, and the session row and token log are written once at the last chunk (`ReqMeta.total_blocks`) |
| #66 | **A session preempted during prefill is silently saved half-way + orphan chain** | A preempted request comes in **again** through `scheduled_new_reqs`, but the salt was consumed with `_request_cache_salt.pop()` on its first appearance → on reappearance it was saved to a **different chain** with a session=None seed (94 orphan blocks), the original chain was cut at the preemption point, and the session row and token log were not recorded. Fix: the salt is read with `get` and cleaned up at `finished_req_ids`; a reappearing store request **replaces** its block list + keeps `saved_blocks`; the cached branch also replaces for `resumed_req_ids` (exactly as the vLLM docstring says) |

#64 did not surface because all earlier vLLM tests used prompts of 2K or less, which fit in one step.

Reproducing #66 was itself half of the work (`tests/stress_preempt_store_vllm.py` (internal script, not published)): vLLM
**avoids admissions that would cause preemption** with `scheduler_reserve_full_isl` (default True), so only after turning that option
off (`--no-scheduler-reserve-full-isl`) and overlapping "a session that grows while holding blocks
+ multi-chunk prefills" with 512-token chunks and a 500-block pool did a single preemption finally occur. Before the fix: the preempted session's
external +1,488 (stopped at 93 blocks), 94 orphan blocks, no session row. After the fix: **+2,976 (all)**,
block count 20,020 = (386+186)×35 exact match (0 orphans), session row and token log present, 3-criterion probe regression
passed. Another lesson from this reproduction: unless the needle is placed at the 85% point of the prompt (the region after preemption), the contamination
passes the correctness check.

## 7. Follow-up measurements and verdict (2026-07-31 afternoon)

### Crossover point: the context length at which qml starts to win ≈ 4.6K tokens

Same protocol (restart, warmed on both sides, correctness check, size included in the salt to prevent cache contamination across sizes),
gemma-3-4b bf16, block 16:

| Prompt tokens | Without qml (full prefill) | With qml (restore) | Difference |
|---|---|---|---|
| 1,667 | **267 ms** | 366 ms | without +99 |
| 3,317 | **338 ms** | 389 ms | without +51 |
| 6,617 | 621 ms | **544 ms** | with −77 |

Linearly interpolated crossover point **≈ 4,600 tokens**. Restore has a fixed component of ~330 ms (scheduling + match + injection overhead),
so prefill wins on short contexts. This is for this GPU and model; since prefill is superlinear, the gap keeps widening on longer
contexts.

### block_size 16 → 64: blocked by the engine for this model

gemma-3 is a hybrid (SWA+full-attention) model, and when a KV connector is attached vLLM disables the hybrid KV cache
manager. In that state both `--block-size 64` and `32` fail at launch:
`ValueError: Hybrid KV cache manager is disabled but failed to convert the KV cache specs to
one unified type.` **Only block 16 can launch** — judged not measurable. It could be tried on a full-attention model,
but changing the model also changes KB/token and prefill speed, which would invalidate the comparison, so it is on hold.

### Handling of 2 things learned from vLLM itself

- **Evict tail blocks first**: adopted. `rowid DESC` tie-breaker in `QML_SQL_KV_BLOCK_SELECT_LRU` —
  batched touch stamps the same timestamp in chain order, so on a tie the larger rowid = the chain tail
  is evicted first and the prefix lives longer. The same heuristic vLLM implements by freeing in `reversed()` order.
- **Partial block matching (`cache_partial_block`)**: **rejected.** Maximum saving = block_size−1 = 15 tokens =
  ~1.7 ms at 9,000 tok/s — smaller than run-to-run variation (±30 ms). Only the implementation complexity would remain.

## 8. Honest positioning

Restore is **correct** (6,608 tokens supplied externally + matching correct answer). Speed at 6.6K tokens is effectively a tie when cold
and 12% ahead when warmed — not a big win, and other context lengths are unverified. The main value of the current vLLM path
is not speed but **session persistence and the cross-engine token log**.
