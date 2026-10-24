<!--
SPDX-License-Identifier: AGPL-3.0-only
Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
-->

# qml-router — API Overview

qml-router is a control-plane session-persistence layer for LLM serving on
resource-constrained (on-premises / edge) nodes. It persists two things per
session, keyed by **identity** (`user::session`), never by content:

| Record | Store | Portability |
|---|---|---|
| Token log (the conversation's token-id sequence) | SQLite (`qml.db`) | Engine-neutral — the migration carrier |
| KV tensor blobs (the computed cache bytes) | LMDB (`qml_lmdb/`) | Engine-local — rebuilt per engine from the token log |

All engine integrations converge on one C core (`core/`, `storage/`), exposed
through three surfaces described below.

## 1. C session API (`include/qml_router.h`)

The complete public surface. Every function documents `in:`/`out:` in the
header; the groups are:

| Group | Functions | Role |
|---|---|---|
| Lifecycle | `qml_store_init / _shutdown / _compact` | Open/close the SQLite+LMDB pair |
| Session turn cycle | `qml_session_begin / _end`, `_append_tokens`, `_track_decode`, `_get_last_pos`, `_has_new_tokens` | Record the token log as the engine computes |
| Snapshot (KV blob) | `qml_session_save`, `_save_tokens_only`, `_load`, `_load_blob`, `_load_blob_any`, `_load_tokens`, `_load_done` | Capture/restore engine KV state; `_save_tokens_only` is the delta-threshold cheap path (`QML_CAPTURE_MIN_DELTA`, default 512 tokens) |
| Geometry & locks | `qml_session_set_geometry`, `_acquire_lock / _get_lock / _release_lock`, `_restore_done` | Validate stored-vs-live engine config before a restore; cross-process session ownership |
| RAM staging | `qml_session_prefetch` | Stage a blob into the OS page cache (L2) without touching the GPU |
| Control plane | `qml_control_put / _claim_next / _update_status / _get_blocks` | The SQLite command bus (`PREWARM`, `BACKUP_NOW`, `PREFETCH_RAM`, `PURGE_CACHE`, `COPY_SESSION`) that any process can enqueue and the engine polls |
| Router scope | `qml_router_init / _shutdown`, `_execute_query`, `_select_target_engine`, `_update_kv_mesh` | The routing layer above the persistence core (multi-engine target selection; the paper measures the persistence core) |

Memory-tier vocabulary used throughout: **L1** = session resident in a GPU
slot, **L2** = blob bytes in system RAM (page cache / `PREFETCH_RAM`),
**L3** = disk only.

## 2. Engine attach points

| Engine | Mechanism | Files |
|---|---|---|
| llama.cpp | Server patch (no upstream extension point exists). Applied as `patches/llama.cpp-qml-session.patch` on upstream commit `0d47ea742` (build b10141) | `hooks/llama.cpp/qml_llama_hook.cpp` (+ `include/qml_llama_hook.h`), optional `LD_PRELOAD` shim `hooks/llama.cpp/qml_llama_preload.cpp` |
| vLLM | Official `KVConnector` V1 plug-in — zero source changes. Identity arrives via the `cache_salt` request field; blocks are saved by a writer thread, loads are pipelined and batched | `hooks/vllm/qml_vllm_connector.py`, ctypes store bindings `hooks/vllm/qml_store_bindings.py` |
| SGLang | Official `HiCacheStorage` backend — zero source changes. Session identity + token log recorded one layer above the opaque page hashes | `hooks/sglang/qml_sglang_storage.py`, `hooks/sglang/qml_sglang_hook.py` |

Engine-side llama.cpp flags added by the patch: `--qml-session-store <qml.db>`
and `--qml-lmdb-store <dir>`. A request binds to a session by carrying
`session_id` (llama.cpp `/completion`) or the forwarded identity headers via
the gateway (below). The client then sends **only the new turn**; the server
prepends stored history itself (auto-continue).

## 3. Session layer for vLLM and SGLang (`router/qml_openai_gateway.py`, stdlib-only)

An OpenAI-compatible front end that owns the session for vLLM (with the qml-router connector)
and SGLang (with the qml-router backend): it keeps the token log, trims the window (TAWT) and
recalls archived turns (RIG) through the C core, and sends the engine token ids with the
session key (`cache_salt` for vLLM, `extra_key` for SGLang). The engine modules then handle
the KV payload only.

| Endpoint | Role |
|---|---|
| `POST /v1/chat/completions` | With `session_id`: the last user message is the new turn (a `system` message is used only for a new session); the response carries a `qml` object with what the session layer did. Without `session_id` the request is passed through statelessly. |
| `GET /health` | Liveness |

```bash
python3 router/qml_openai_gateway.py --engine vllm --engine-url http://127.0.0.1:8000 \
  --model MODEL_NAME --tokenizer MODEL_DIR --store-db STORE/qml.db \
  --store-lmdb STORE/qml_lmdb [--window 3000] [--port 8090]
```

## 4. Store schema (SQLite)

`qml_session` (identity row: n_tokens, engine_type, lock, timestamps),
`qml_token_log` (session_id, pos, seq_id, token_id), `qml_token_archive`
(trimmed history), `qml_kv_block` (vLLM/SGLang block index),
`qml_llama_session_ext` / `qml_vllm_session_ext` (per-engine metadata),
`qml_control_cmd` (+`_blocks`) (command bus). The schema is declared in
`storage/sqlite/qml_sqlite_schema.h` and is SQL-queryable at rest
("this user's sessions and usage?").

## 5. Options

Defaults are those of the code in this repository. Engine options are environment
variables read once when `llama-server` starts.

### 5.1 Request fields (llama.cpp `/completion`)

| Field | Type | Default | Effect |
|---|---|---|---|
| `session_id` | string | empty | Binds the request to a stored session (`user::session`). Empty = stateless request, not saved. |
| `user_id` | string | empty | Owner recorded with the session (usage, quota, audit queries). |
| `qml_transient` | bool | false | The user turn and its answer are not kept in the session (KV and token log are cut back before the save). Needs the optional patch (5.6). |
| `qml_recall_query` | string | empty | Query used for archive retrieval on this turn instead of the full turn text. Needs the optional patch (5.6). |

### 5.2 Window trim (TAWT)

Turns beyond the restore window move to the per-session archive (`qml_token_log_archive`,
indexed in `qml_archive_fts`) at a turn boundary; the system prefix is kept.

| Where | Option | Default | Effect |
|---|---|---|---|
| Gateway | `--restore-window N` | 32768 | Window in tokens; trim runs before the request is forwarded (`trim_session_window`: `PURGE_CACHE` + `DROP_BLOB` + token-log rewrite + archive insert). |
| Gateway | `user_prefs.restore_window` (SQL row per user) | unset | Per-user window, overrides `--restore-window`. |
| Gateway | `--archive-after N` | 0 (off) | Delete the KV payload of sessions unused for N seconds; the token log is always kept. |
| Engine | `QML_TRIM_WINDOW` | 0 (off) | Window in tokens for the engine's own trim on the restore path. |
| Engine | `QML_TRIM_EOT` | unset | End-of-turn token id that marks turn boundaries (Llama-3: 128009, Qwen3/ChatML: 151645). Required with `QML_TRIM_WINDOW`. |
| Engine | `QML_TRIM_HEAD` | 1 | 0 = do not keep the system prefix (measurement switch). |
| OpenAI session layer | `qml_openai_gateway.py --window N` | 0 (off) | Window in tokens for vLLM and SGLang (`qml_window_trim` in the core). The trim also drops the KV blocks the session stored through the vLLM connector or the SGLang backend (`qml_kv_owner`, `qml_store_drop_kv`): positions are renumbered and K is stored after RoPE, so those blocks cannot match again. |

### 5.3 Archive retrieval and injection (RIG, Recall Injection Gate)

| Option | Default | Effect |
|---|---|---|
| `QML_ARCHIVE_RECALL` | off | `1` = retrieve archived turns (FTS5, BM25) and inject them between the history and the new turn. |
| `QML_ARCHIVE_RECALL_BUDGET` | 1024 | Upper limit of injected tokens per turn. |
| `QML_ARCHIVE_RECALL_TOPK` | 3 | Candidates retrieved per turn (upper bound 64). |
| `QML_ARCHIVE_RECALL_CONF` | 0.2 | Relative-confidence threshold of the gate; 0 = gate off (always inject). |
| `QML_ARCHIVE_RECALL_NMIN` | 4 | Minimum effective words in the question; shorter questions get a proportionally lower score. |
| `QML_ARCHIVE_RECALL_PAIR_TURNS` | 1 | Index a user turn and its answer as one document. |
| `QML_ARCHIVE_RECALL_SPREAD_W`, `_SPREAD_K` | 0, 2 | Position window and count for spreading candidates over the history (0 = off). |
| `QML_ARCHIVE_RECALL_FACTS` | 0 | Retrieve extracted fact rows instead of turns. |
| `QML_ARCHIVE_RECALL_TRIGGER` | unset | Inject only when the turn matches this trigger (unset = every question). |
| `QML_EMBED_URL` / `QML_EMBED_MODEL`, `QML_EMBED_NGL` | unset | Embedding server or in-process model for hybrid (lexical + embedding) retrieval. |

### 5.4 Persistence

| Option | Default | Effect |
|---|---|---|
| `QML_CAPTURE_MIN_DELTA` | 512 | A turn that adds fewer tokens than this saves the token log only; the stored KV payload restores as a prefix and the tail is recomputed. |
| `--qml-session-store`, `--qml-lmdb-store` | none | Store paths; without them the engine runs as stock llama.cpp. |

### 5.5 Control commands (`qml_control_cmd` table)

Any process inserts a row; the engine polls the table and sets the status.

| Command | Effect |
|---|---|
| `PREWARM` | Restore a stored session into a free slot before the client returns. |
| `PREFETCH_RAM` | Read the KV payload into the page cache without using the GPU. |
| `BACKUP_NOW` | Save the resident session now, regardless of `QML_CAPTURE_MIN_DELTA`. |
| `MIGRATE` | Save, then free the slot (the session continues on another engine from the token log). |
| `PURGE_CACHE` | Drop the session's resident KV in the engine. |
| `DROP_BLOB` | Delete the stored KV payload; the token log stays and the next visit rebuilds by prefill. |
| `COPY_SESSION` | Copy a session record to a new session id. |

### 5.6 Optional llama.cpp patch

`hooks/llama.cpp/server-context-resident-dedup.patch` applies on top of
`patches/llama.cpp-qml-session.patch` (it is not applied by `scripts/setup_llama_cpp.sh`):

```bash
git -C third_party/llama.cpp apply ../../hooks/llama.cpp/server-context-resident-dedup.patch
```

It adds `qml_transient` and `qml_recall_query` (5.1), and removes the archived turns
injected by 5.3 from the KV cache and the token log before each save, so injected
material is used for that turn only and does not accumulate in the session. The patch
measured in the paper (Table 15) is `patches/llama.cpp-qml-session.patch` alone; among
the logs in `artifact/`, none was made with this patch applied.

### 5.7 Options used by the paper's runs

| Run | Trim | Retrieval and injection | Questions |
|---|---|---|---|
| Table 5, llama.cpp + qml-router | Gateway procedure (5.2) executed by the benchmark client (`trim_llama()`) | Engine, `QML_ARCHIVE_RECALL=1` | Removed from the session after scoring (`drop_rows_after()`) |
| Table 5, vLLM / SGLang + qml-router | Client keeps the resent text window | Client: archive sentences containing the question's name, up to 800 tokens | Not stored by the engine |
| Section V-B1, 20,000 turns | Gateway procedure (5.2) executed by the benchmark client (`trim()`) | Engine, `QML_ARCHIVE_RECALL=1`, `_TOPK=10`, `_BUDGET=1024`, `_PAIR_TURNS=1`, `_CONF` as stated per condition | Each question on its own copy of the finished session (`clone_session()`) |
| Fig. 6, LongMemEval | None; the benchmark's `ingest` mode writes each history into the archive and the FTS index with SQL | Engine, k = 32 | One session per question |

The weather program (`tests/multi-turn/common/bench_weather_longevity.py`) asks its questions
in one of two ways, selected by `WX_QUESTION_MODE` on the `ask` stage:

| Value | How the questions are asked |
|---|---|
| `conversation` (default) | The questions continue the fed conversation in order, on one copy of it. Question and answer turns stay in the history, and the overflow trim runs as during the feed. |
| `clone` | Each question is asked on its own copy of the fed session, made by the `clone` stage. This is the procedure of the Section V-B1 runs; the scripts in `tests/paper/secVB1_session_beyond_limits/` set it. |

### 5.8 Notes on reported numbers

The paper text is not changed after acceptance. Where a number in the paper needs context
that the paper does not give, the note is kept here.

| Item | Paper text | What the measurements show |
|---|---|---|
| Cost model (Abstract, Section I, Section IV, Section V-A4, Section VII) | The cost model matches the measured restore/recompute boundary "within 2-10% at the measured points (not a general error bound)", "across three hardware classes". | On the final build (the build of every Table 7 row), the predictions were within 10% of the measurements on the RTX 4090 host with NVMe: 250 ms predicted against 275 ms measured at 8K tokens (10%) and 650 ms against 661 ms at 32K tokens (2%). The restore-or-recompute verdict follows the model on all three hardware classes of Table 7 (RTX 4090 with 0.55 GB/s storage, the same GPU with NVMe, Jetson AGX Orin). |
