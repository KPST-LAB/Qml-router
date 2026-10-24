# Per-engine implementation: SGLang

One of the three engine documents. [engine-llama-cpp.md](engine-llama-cpp.md) · [engine-vllm.md](engine-vllm.md)

As of 2026-07-31 (evening: after lifting the hold, reuse across restart **proven**, batch I/O applied).
Speed still loses to recomputation — the purpose of this document is not to blur that fact.

---

## 1. Extension method — 0 source patches

A `HiCacheStorage` subclass is loaded dynamically with `--hicache-storage-backend`. Searching the installed `sglang`
package tree for the string `qml` returns **0 matches**. The implementation is only `hooks/sglang/qml_sglang_storage.py`.

Required launch options (both found only through actual launch failures):

| Option | Why it is needed |
|---|---|
| `--enable-hierarchical-cache` | **Without it, our backend is not created at all** |
| `--disable-cuda-graph` | Avoids an SGLang/FlashInfer shape bug unrelated to our code in the combination of the gemma3 multimodal code path + CUDA graph capture (`q.shape[0] does not match qo_indptr[-1]`) |
| `"backend_name"` in extra-config | **Required** (as of 0.5.16; without it, launch fails with `ValueError`) |
| Keep the default `--page-size` | Both 16 and 64 **segfault even without qml** — an engine bug of this version + gemma-3 combination (section 6) |

One real bug caught during implementation: `_create_dynamic_backend` in `backend_factory.py` calls
`backend_class(storage_config, kwargs)`, and this `kwargs` is `create_backend()`'s own
`**kwargs` (effectively an empty dict), not the content of `--hicache-storage-backend-extra-config`.
At first we read it the other way around and "fixed" the constructor, which surfaced as a `ValueError` at launch, so we reverted it
(reading from `storage_config.extra_config` is correct).

## 2. Implementation status

| Method | Status |
|---|---|
| `set` / `batch_set` | **Verified live** — single commit with `qml_lmdb_put_gather()` (section 5) |
| `get` / `batch_get` | **Verified live** — 1,556/1,567 reused after restart (section 3), single txn with `qml_lmdb_get_scatter()` |
| `exists` / `batch_exists` | **Verified live** — `qml_lmdb_exists_run()` |
| Control command polling | **None** (llama.cpp and vLLM have it) |
| Session/token log recording | **Verified live** (resolved in section 4 — write-path recorder) |

## 3. ~~Unresolved~~ Resolved (1): reuse after restart — **proven (2026-07-31 evening)**

**The earlier conclusion ("SGLang does not query our backend", the radix tree hypothesis) was wrong.** The real
cause was the gate in `scheduler._prefetch_kvcache()` → `prefetch_from_storage()`:

```python
if (not self.enable_storage
        or prefetch_length < self.prefetch_threshold     # default 256 tokens!
        or self.cache_controller.prefetch_rate_limited()):
    return
```

**The default `prefetch_threshold` is 256 tokens**, and the earlier verification used a 34-token prompt — it silently returned
before reaching the backend. An empty radix tree on a cold restart is actually the pass
condition (`last_host_node is root_node`).

Demonstration (gemma-3-4b, 1,557-token needle prompt, relaunched with the same store after kill, identical request):
`cached_tokens = 1,556 / 1,567` (after batch I/O was applied; effectively all supplied from the external store) + correct answer.
**P1 (survives restart) is proven on all three engines.**

Note: the **`backend_name` field became required** in `--hicache-storage-backend-extra-config`
(in this SGLang version; without it, launch fails).

## 4. ~~Unresolved~~ Resolved (2): session_id — completed by hooking the hicache write path (task #37, 2026-08-04)

| Question | Answer |
|---|---|
| Is there a session isolation key such as `cache_salt`? | **Yes.** `Req.extra_key` / `RadixKey.extra_key` already takes part in the hash computation — unlike vLLM, where a separate `on_new_request()` hook had to be built, there is nothing for us to do |
| Is session_id accessible when the backend is called? | **No.** `get/set/batch_get/batch_set/exists` receive only opaque page hash keys and `prefix_keys`. The plaintext `cache_salt` is not passed |

As a result, the SGLang path **can (in theory) do pure KV reuse but leaves nothing in `qml_session` /
`qml_sglang_session_ext` / `qml_token_log`** — that is, it is excluded from the session and user management shown by
`scripts/qml_manager.py` (internal script, not published), and it cannot take part in cross-engine portability
(a→b, b→a, c=a+b were verified only for llama.cpp↔vLLM) either.

**Resolution (2026-08-04)**: the storage ABC does not receive plaintext, but one layer up,
`node.key` (RadixKey) of `HiRadixCache.write_backup_storage(node)` carries **token_ids and extra_key
together**. At creation time our backend wraps this one method (`_install_identity_recorder`,
qml_sglang_storage.py) and records `qml_session` (engine_type=SGLANG) + `qml_token_log` —
keeping 0 lines of SGLang source patches. Completeness argument: the invariant of the write-through policy ("a backed-up node is a
gap-free prefix from the root") guarantees that accumulated segments form the full token sequence, and since it uses INSERT OR REPLACE, re-backup is
idempotent too. Every access is wrapped in a version guard, and on a mismatch only the recording is skipped (no effect on KV storage).

Demonstration (2026-08-04, test record: internal record, not published): 3K needle session with `cache_salt="alice::sg-1"` →
`qml_session` row (user_id='alice', engine_type=2, n_tokens=3,027) + 3,027 token log rows recorded →
**llama.cpp received only the new question from the same store, prepended the 3,027 tokens, and answered correctly** (385 ms).
SGLang→llama.cpp handover worked for the first time.

Core defect fixed at the same time: `qml_session_snapshot_load_tokens` assumed a llama extension row + blob, so
a session created by another engine (only the base row + token log) was treated as "not found" → **wrong answer without the history**
(confirmed live in the first verification). Now, if there is no llama snapshot, it falls back to the token log — this is
the general path for cross-engine handover.

## 5. ~~Unresolved~~ Resolved (3): batch I/O — **applied (2026-07-31 evening)**

SGLang gives a **separate host tensor** as the destination for each page, so `get_many/put_many`, which are for contiguous buffers,
cannot be used. Pointer-array variants were added in C: `qml_lmdb_get_scatter()` /
`qml_lmdb_put_gather()` — the whole batch is 1 read txn / 1 write commit. `batch_exists` is
implemented with `qml_lmdb_exists_run()` (count of consecutive prefixes, 1 txn).

Real bug caught during implementation: passing the tensor's `data_ptr()` directly to C **segfaults on a non-contiguous view (a slice of the
host pool) or a CUDA tensor** — it is now passed after `_host_contig()` guarantees CPU + contiguous, and if a copy
occurred the read result is copy_()'d back to the original tensor (we saw the scheduler die with exit -11 live).

Effect: reuse completeness 1,154/1,567 → **1,556/1,567** (prefetch now completes in time).
For total time, however, see section 6 — the remaining bottleneck is not us.

## 6. Measurement — resume after restart (product test, 2026-07-31)

Same protocol (restart → warmed → identical request → correctness verification), gemma-3-4b, comparison is **with qml
vs. without qml (only the storage backend removed from the same launch flags)**:

| Prompt | Without qml (full recomputation) | With qml (reuse from the store) | Reused amount |
|---|---|---|---|
| 1,567 tokens | **1,187 ms** | 3,045 ms (after applying 8 threads) | 1,556/1,567 |
| 6,629 tokens | **1,279 ms** | 4,052 ms (same) | 2,965/6,629 |

**Honest verdict: on SGLang, restore is correct but still slower than recomputation.** Where the bottleneck lies was
confirmed by instrumentation, not inference (after applying `QML_LOAD_TIMING=1`, `qml_read_threads=8`):

```
qml sglang batch_get: 3~5 ms per call (128 keys, 17.0 MiB) ≈ 4 GB/s
```

Our store reads total **~50 ms** for the whole session — about 1.5% of the total resume time of 3,045 ms. The remaining
~2.9 s is spent in SGLang's own prefetch pipeline (page=1, so per-token host index allocation, radix node
insertion, and GPU loading). 8 read threads improved it only 3,306→3,045 ms (8%) — because our part
is already small. Both `--page-size 16` and `64` **segfault even without qml (exit -11)** — an engine bug of this SGLang
version (0.5.16) + gemma-3 + hierarchical cache combination, so the standard approach of larger pages
is blocked.

## 7. Remaining items

| # | Item | Notes |
|---|---|---|
| 1 | task #37 — path to obtain session_id (section 4) | Feature gap. Needs hooking in the serving layer where the plaintext is visible before hashing |
| 2 | Inclusion in cross-engine portability | Follows from 1 |
| 3 | Retry on speed | Re-measure when an SGLang version upgrade fixes the page-size bug |
| 4 | Control command (RESTORE/PREWARM) polling | The other two engines have it |
| 5 | Scripting the launch options (including the required `backend_name`) | Exists only in the documentation |

## 8. Current status in one line

**"The conversation survives a restart" (P1) is proven on all three engines.** On SGLang, save, restore, and correctness
are all confirmed, and only speed loses to recomputation because of an engine constraint (page=1 forced). Session metadata (#37)
does not take part yet.
