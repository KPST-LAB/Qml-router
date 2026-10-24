// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
//
// qml_router.h
#ifndef QML_ROUTER_H
#define QML_ROUTER_H

#include "qml_sqlite_store.h" // qml_control_cmd_t / qml_control_cmd_block_t

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// QML Router initialization (sets SQLite and LMDB paths)
//  in: sqlite_path — session-metadata SQLite db path
//  in: lmdb_path — KV-blob LMDB store path
// out: intended 0 on success / -1 on failure (declared routing-plane
//      interface; router/src/qml_router.c is currently an empty stub)
int qml_router_init(const char* sqlite_path, const char* lmdb_path);

// ---- in-process embedding engine (router/src/qml_router.c, 2026-08-29) ----
// Serves the embedding model through the llama C API in the same process, without a
// separate embedding server — the goal is to keep a single-process topology (1 engine + N agents).

// Load the embedding model (once; returns success as a no-op if already loaded)
//  in: model_path — embedding gguf path (e.g. bge-m3)
//  in: n_gpu_layers — 0 = CPU only (recommended), 99 = all layers on GPU
// out: 0 success / -1 failure
int qml_embed_init(const char * model_path, int n_gpu_layers);

// Get the embedding dimension
// out: number of dimensions (0 if not initialized)
int qml_embed_dim(void);

// Embed one text
//  in: text — UTF-8 (truncated beyond 1,024 tokens)
//  in: out / out_cap — result buffer and its capacity (number of floats)
// out: 0 success (L2-normalized vector) / -1 failure
int qml_embed_text(const char * text, float * out, int out_cap);

// Execute a QML query
//  in: qml_script — QML query text to execute
// out: intended 0 on success / -1 on failure (declared interface; no
//      in-tree implementation yet)
int qml_router_execute_query(const char* qml_script);

// KV Mesh token-slot update (zero-copy)
//  in: session_id — numeric session identity the tokens belong to
//  in: token_ids — token id array, num_tokens entries
//  in: num_tokens — number of entries in token_ids
//  in: engine_type — QML_ENGINE_* value (see inline comment below)
//  in: slot_ptr — engine-native VRAM/RAM slot address (opaque; recorded,
//      not dereferenced by the storage layer)
// out: intended 0 on success / -1 on failure (declared interface; no
//      in-tree implementation yet)
int qml_router_update_kv_mesh(
    uint64_t session_id,
    const int32_t* token_ids,
    size_t num_tokens,
    uint32_t engine_type, // 0: llama.cpp, 1: vLLM, 2: SGLang
    uintptr_t slot_ptr    // VRAM/RAM memory address pointer
);

// Look up target engine routing
//  in: token_ids — token id array to route on
//  in: num_tokens — number of entries in token_ids
// out: intended selected engine id (QML_ENGINE_*), negative on failure
//      (declared interface; no in-tree implementation yet)
int32_t qml_router_select_target_engine(const int32_t* token_ids, size_t num_tokens);

// QML Router shutdown
//  in: (none)
// out: none (void)
void qml_router_shutdown();

//------------------------------------------------------------
// QML LLaMA Session Bridge
//
// Engine-agnostic bridge between the LD_PRELOAD hook layer
// (hooks/llama.cpp/qml_llama_preload.cpp) and the storage layer
// (storage/sqlite, storage/lmdb). Only raw byte buffers cross this
// boundary — no llama.cpp types appear here, so this header stays
// includable from plain C storage code.
//
// Implementation: core/qml_session.c
//------------------------------------------------------------

// opaque per-context live session tracking handle
typedef struct qml_llama_session qml_llama_session_t;

// opaque LMDB read-txn handle (full definition in storage/lmdb/qml_lmdb_store.h) --
// forward-declared here so callers of the snapshot API below don't need to
// include the storage layer's own header just to hold this pointer around
typedef struct qml_lmdb_txn qml_lmdb_txn_t;

//------------------------------------------------------------
//Func : qml_store_init
//opens the sqlite metadata db and lmdb blob store
//returns 0 on success, -1 on failure
//  in: sqlite_path — session-metadata SQLite db path
//  in: lmdb_path — KV-blob LMDB store path
//------------------------------------------------------------
int qml_store_init(const char* sqlite_path, const char* lmdb_path);

//------------------------------------------------------------
//Func : qml_store_close
//closes the sqlite/lmdb stores opened by qml_store_init
//  in: (none)
// out: none (void)
//------------------------------------------------------------
void qml_store_close();

//------------------------------------------------------------
//Func : qml_session_begin
//llama_init_from_model POST-hook entry point.
//looks up (user_id, session_id) in storage. if a prior state exists:
//  - *out_state_data / *out_state_size point ZERO-COPY into the LMDB
//    mapping (caller feeds them straight into llama_state_seq_set_data)
//  - *out_last_pos receives the highest restored position
//if nothing is found, *out_state_data = NULL, *out_last_pos = -1, and a
//fresh (empty) tracking handle is still returned.
//model_hash is compared against any stored value; pass "" to skip the check.
//always returns a non-NULL handle.
//  in: user_id — owner identity stored with the session
//  in: session_id — session key to look up / create
//  in: seq_id — engine sequence id the state belongs to
//  in: model_hash — fingerprint validated against the stored row ("" = skip)
//  in: out_state_data / out_state_size — (output) zero-copy LMDB state view
//      (NULL/0 when no prior state exists)
//  in: out_last_pos — (output) highest restored position (-1 when none)
// out: non-NULL tracking handle, always; restore results come back via the
//      output parameters above
//------------------------------------------------------------
qml_llama_session_t* qml_session_begin(
    const char* user_id,
    const char* session_id,
    int32_t     seq_id,
    const char* model_hash,
    const void** out_state_data,
    size_t*      out_state_size,
    int32_t*     out_last_pos
);

//------------------------------------------------------------
//Func : qml_session_restore_done
//releases the zero-copy LMDB read handle opened by qml_session_begin
//(no-op if qml_session_begin found no prior state). call this once the
//caller is done reading *out_state_data.
//  in: sess — handle returned by qml_session_begin (stays valid afterwards)
// out: none (void)
//------------------------------------------------------------
void qml_session_restore_done(qml_llama_session_t* sess);

//------------------------------------------------------------
//Func : qml_session_track_decode
//llama_decode PRE-hook bookkeeping. token_ids/pos/seq_ids are parallel
//arrays of n_tokens length. pos[] is rewritten IN PLACE, shifted by the
//session's restore offset, so newly decoded tokens are appended after
//the already-restored KV range instead of colliding with it (pos 0..N-1).
//  in: sess — tracking handle from qml_session_begin
//  in: token_ids — n_tokens token ids being decoded
//  in: pos — n_tokens positions; REWRITTEN IN PLACE (restore-offset shift)
//  in: seq_ids — n_tokens sequence ids, parallel to token_ids
//  in: n_tokens — length of the three parallel arrays
// out: none (void)
//------------------------------------------------------------
void qml_session_track_decode(
    qml_llama_session_t* sess,
    const int32_t* token_ids,
    int32_t*       pos,
    const int32_t* seq_ids,
    size_t         n_tokens
);

//------------------------------------------------------------
//Func : qml_session_end
//llama_free PRE-hook. persists state_data/state_size (already obtained
//by the caller via llama_state_seq_get_data) plus the accumulated token
//log to SQLite/LMDB, then frees the tracking handle. sess is invalid
//after this.
//  in: sess — tracking handle (freed by this call)
//  in: state_data / state_size — full KV blob from llama_state_seq_get_data
// out: none (void)
//------------------------------------------------------------
void qml_session_end(qml_llama_session_t* sess, const void* state_data, size_t state_size);

//------------------------------------------------------------
//Func : qml_session_has_new
//returns 1 if at least one token has been tracked via
//qml_session_track_decode since qml_session_begin, 0 otherwise. lets a
//caller skip an expensive llama_state_seq_get_data() call (e.g. before
//evicting an idle slot) when there is nothing new to persist -- mirrors
//the same gate qml_session_end() applies internally.
//  in: sess — tracking handle from qml_session_begin
// out: 1 if new tokens were tracked, 0 otherwise
//------------------------------------------------------------
int qml_session_has_new(qml_llama_session_t* sess);

//------------------------------------------------------------
// QML Session Snapshot API (stateless)
//
// For hosts that already hold a complete token array + KV blob at a
// single well-defined point (e.g. llama-server's update_slots()/
// release(), which unlike the LD_PRELOAD/llama-cli path never needs
// incremental per-decode-call tracking). These do not use
// qml_llama_session_t at all -- no lifecycle, just load-then-done and
// a one-shot save. See hooks documentation / llama-server integration
// notes for usage.
//------------------------------------------------------------

//------------------------------------------------------------
//Func : qml_session_load
//looks up (session_id) in storage. if found and model_hash matches
//(pass "" to skip the check):
//  - *out_state_data is a malloc'd contiguous copy of the state blob
//    (chunk-assembled; the LMDB read txn is already closed when this
//    returns, so no reader pins pages while the caller uploads to GPU)
//  - *out_token_ids is malloc'd, holding *out_n_tokens token ids in
//    position order
//release both through qml_session_load_done.
//returns 0 if found, 1 if not found (or model_hash mismatch / missing
//blob -- treat identically to "start fresh"), -1 on error. On any
//non-zero return, out params are all NULL/0 and there is nothing to
//release.
//  in: session_id — stored session key
//  in: model_hash — fingerprint to validate ("" = skip)
//  in: out_state_data / out_state_size — (output) malloc'd contiguous blob
//  in: out_token_ids / out_n_tokens — (output) malloc'd token id array + count
//------------------------------------------------------------
int qml_session_load(
    const char*  session_id,
    const char*  model_hash,
    void**       out_state_data,
    size_t*      out_state_size,
    int32_t**    out_token_ids,
    size_t*      out_n_tokens
);

//------------------------------------------------------------
//Func : qml_session_load_tokens
//token list + the same model_hash / token-log-vs-meta pairing checks as
//qml_session_load, and a blob existence check (header only) --
//but WITHOUT copying the state blob. For peek paths that only need the
//token array. *out_token_ids is malloc'd; free with
//qml_session_load_done(NULL, token_ids).
//returns 0 if found, 1 if nothing usable, -1 on error.
//  in: session_id — stored session key
//  in: model_hash — fingerprint to validate ("" = skip)
//  in: out_token_ids / out_n_tokens — (output) malloc'd token array + count
//------------------------------------------------------------
int qml_session_load_tokens(
    const char* session_id,
    const char* model_hash,
    int32_t**   out_token_ids,
    size_t*     out_n_tokens
);

//------------------------------------------------------------
//Func : qml_session_load_blob
//blob + metadata only, skipping the token-log SELECT (the caller already has
//the tokens from an earlier peek). *out_n_tokens_meta lets the caller verify
//that its tokens and this blob describe the same state. Release with
//qml_session_load_done(state_data, NULL).
//returns 0 if found, 1 if nothing usable, -1 on error.
//  in: session_id — stored session key
//  in: model_hash — fingerprint to validate ("" = skip)
//  in: out_state_data / out_state_size — (output) malloc'd contiguous blob
//  in: out_n_tokens_meta — (output) token count the stored metadata claims
//------------------------------------------------------------
int qml_session_load_blob(
    const char*  session_id,
    const char*  model_hash,
    void**       out_state_data,
    size_t*      out_state_size,
    int64_t*     out_n_tokens_meta
);

//------------------------------------------------------------
//Func : qml_session_load_any
//load_blob without the slot-geometry guard, additionally reporting the
//n_stream the blob was saved under (0 = unknown/legacy row). Only the
//llama hook should use this: it can re-frame a mismatched blob instead of
//discarding it. Release with qml_session_load_done(data, NULL).
//  in: session_id — stored session key
//  in: model_hash — fingerprint to validate ("" = skip)
//  in: out_state_data / out_state_size — (output) malloc'd contiguous blob
//  in: out_n_tokens_meta — (output) token count the stored metadata claims
//  in: out_n_stream — (output) n_stream the blob was saved under (0 = unknown)
// out: 0 if found (outputs filled), 1 if nothing usable, -1 on error
//------------------------------------------------------------
int qml_session_load_any(
    const char*  session_id,
    const char*  model_hash,
    void**       out_state_data,
    size_t*      out_state_size,
    int64_t*     out_n_tokens_meta,
    uint32_t*    out_n_stream
);

//------------------------------------------------------------
//Func : qml_session_load_done
//frees the state buffer and token array returned by the loads above.
//safe to call with either pointer NULL (no-op for whichever is NULL).
//  in: state_data — blob pointer from a load above, or NULL
//  in: token_ids — token array from a load above, or NULL
// out: none (void)
//------------------------------------------------------------
void qml_session_load_done(void* state_data, int32_t* token_ids);

//------------------------------------------------------------
//Func : qml_session_set_geometry
//Tells the session layer the slot geometry of the live engine context:
//n_stream (= kv_unified ? 1 : n_seq_max) and n_ctx_seq (per-slot context).
//llama.cpp records n_stream at the head of every per-sequence state blob and
//hard-checks it on restore, so knowing the current value lets the restore
//path reject a blob it already knows the engine will refuse -- WITHOUT
//reading it (a 4.4 GiB read at 128K tokens). Call once after the context is
//created; llama-server owns exactly one context per process. Passing 0
//disables the check (behaviour before this existed).
//  in: n_stream — live engine stream count (kv_unified ? 1 : n_seq_max);
//      0 disables the geometry check
//  in: n_ctx_seq — live per-slot context length
// out: none (void)
//------------------------------------------------------------
void qml_session_set_geometry(uint32_t n_stream, uint32_t n_ctx_seq);

//------------------------------------------------------------
//Func : qml_session_save
//one-shot upsert of a complete session: persists state_data/state_size
//(the full current KV blob) and the complete token_ids array (position
//== array index) to SQLite/LMDB, replacing whatever was stored before
//for this session_id. Unlike qml_session_end, this takes no tracking
//handle and has no "nothing new to persist" gate -- callers who already
//know they have nothing new to save should simply not call this.
//returns 0 on success, -1 on failure.
//  in: session_id — session key to upsert
//  in: user_id — owner identity stored alongside ("" if unknown)
//  in: model_hash — fingerprint stored with the row
//  in: seq_id — engine sequence id the state belongs to
//  in: state_data / state_size — full current KV blob to persist
//  in: token_ids / n_tokens — complete token array (position == array index)
//------------------------------------------------------------
int qml_session_save(
    const char*    session_id,
    const char*    user_id,
    const char*    model_hash,
    int32_t        seq_id,
    const void*    state_data,
    size_t         state_size,
    const int32_t* token_ids,
    size_t         n_tokens
);

//------------------------------------------------------------
//Func : qml_session_save_tokens
//capture-throttle variant: persists the session row and token log but leaves
//the blob/ext untouched (blob goes prefix-stale; restore applies the prefix
//and recomputes the tail). returns 0 on success, -1 on failure.
//  in: session_id — session key to upsert
//  in: user_id — owner identity stored alongside ("" if unknown)
//  in: model_hash — fingerprint stored with the row
//  in: seq_id — engine sequence id the tokens belong to
//  in: token_ids / n_tokens — complete token array (position == array index)
//------------------------------------------------------------
int qml_session_save_tokens(
    const char*    session_id,
    const char*    user_id,
    const char*    model_hash,
    int32_t        seq_id,
    const int32_t* token_ids,
    size_t         n_tokens
);

//------------------------------------------------------------
// QML Control Plane Bridge (see docs/udf-control-plane-proposal.md)
//
// Thin engine-agnostic passthroughs onto storage/sqlite/qml_sqlite_store.h's
// qml_sqlite_control_cmd_* functions -- kept here (rather than having
// hooks/llama.cpp, hooks/vllm etc. include qml_sqlite_store.h directly) for the
// same reason the snapshot API above is: engine-specific hook code should
// only ever depend on this one header, not on storage-layer internals.
// qml_control_cmd_t / qml_control_cmd_block_t are defined in
// qml_sqlite_store.h, included above.
//------------------------------------------------------------

//------------------------------------------------------------
//Func : qml_control_put
//see qml_sqlite_control_put's docs (storage/sqlite/qml_sqlite_store.h)
//  in: target_engine — engine name the command targets (e.g. "llama")
//  in: target_pid — pid of the process expected to claim the command
//  in: cmd_type — command verb (RESTORE/MIGRATE/BACKUP_NOW/PREWARM/...)
//  in: session_id — session the command applies to
//  in: block_hashes / n_block_hashes — optional list payload (PREWARM only);
//      pass NULL/0 otherwise
//  in: out_cmd_id — (output) new row's cmd_id
// out: 0 on success, -1 on failure
//------------------------------------------------------------
int qml_control_put(
    const char*          target_engine,
    int32_t              target_pid,
    const char*          cmd_type,
    const char*          session_id,
    const char* const*   block_hashes,
    size_t               n_block_hashes,
    int64_t*             out_cmd_id
);

//------------------------------------------------------------
//Func : qml_control_claim
//see qml_sqlite_control_claim's docs. Intended to be polled
//periodically (e.g. from a hook's own persister/background thread) with
//target_pid == the calling process's own pid.
//  in: target_pid — calling process's own pid
//  in: out_cmd — (output) claimed command row
// out: 0 if a row was claimed (*out_cmd filled), 1 if none pending, -1 on failure
//------------------------------------------------------------
int qml_control_claim(int32_t target_pid, qml_control_cmd_t* out_cmd);

//------------------------------------------------------------
//Func : qml_control_set_status
//moves a command row's status forward; see qml_sqlite_control_set_status
//  in: cmd_id — command row id
//  in: status — 'DONE' | 'FAILED' (normally a terminal state)
// out: 0 on success, -1 on failure
//------------------------------------------------------------
int qml_control_set_status(int64_t cmd_id, const char* status);

//------------------------------------------------------------
//Func : qml_control_blocks
//loads a command's block_hash list payload (PREWARM); see
//qml_sqlite_control_blocks
//  in: cmd_id — command row id
//  in: out_hashes / out_n — (output) malloc'd array (caller free()s) + count
// out: 0 on success (out_n may be 0), -1 on failure
//------------------------------------------------------------
int qml_control_blocks(int64_t cmd_id, qml_control_cmd_block_t** out_hashes, size_t* out_n);

//------------------------------------------------------------
//Func : qml_store_compact
//see qml_sqlite_compact's docs. Backs the COMPACT_NOW control command.
//  in: (none)
// out: 0 on success, -1 on failure (blocks the caller for the VACUUM duration)
//------------------------------------------------------------
int qml_store_compact(void);

//------------------------------------------------------------
//Func : qml_store_drop_kv
//drops every KV block a session stored through the vLLM connector or the
//SGLang backend (qml_kv_owner) and no other session recorded: their records
//and their LMDB values. Called by qml_window_trim, since K is stored after
//RoPE and a trim renumbers positions, so those blocks cannot match again
//  in: owner — session id
// out: number of LMDB values deleted (0 when the session stored none); -1 on error
//------------------------------------------------------------
long qml_store_drop_kv(const char * owner);

//------------------------------------------------------------
//Func : qml_session_prefetch
//page-cache warming tier for a stored session (task #58). Costs no engine
//slot and mutates no live KV, so a wrong guess wastes only page cache --
//unlike PREWARM, which occupies one of the engine's physical slots and can
//evict a real session. Backs the PREFETCH_RAM control command.
//returns 0 if touched (and *out_bytes set), 1 if nothing stored, -1 on error.
//out_bytes may be NULL.
//  in: session_id — stored session whose blob pages to warm
//  in: out_bytes — (output, may be NULL) number of bytes touched
//------------------------------------------------------------
int qml_session_prefetch(const char* session_id, size_t* out_bytes);

//------------------------------------------------------------
//Func : qml_session_lock / qml_session_unlock / qml_session_lock_state
//split-brain guard (docs/udf-control-plane-proposal.md's blocker #2). See
//qml_sqlite_lock's docs (storage/sqlite/qml_sqlite_store.h)
//for exact return-code semantics.
//  in: session_id — session row to lock/unlock/inspect (must already exist
//      as a qml_session row for acquire to succeed)
//  in: pid — calling process's pid (acquire/release)
//  in: out_owner_pid / out_locked — (output, get_lock) owner pid and
//      1 = LOCKED / 0 = FREE
// out: acquire — 0 acquired (previously FREE, or already owned by this same
//      pid), 1 held by a DIFFERENT pid (caller must not proceed), -1 failure
// out: release — 0 on success (releasing a lock not owned is a no-op, not an
//      error), -1 on failure
// out: get_lock — 0 found (outputs filled), 1 session not found, -1 failure
//------------------------------------------------------------
int qml_session_lock(const char* session_id, int32_t pid);
int qml_session_unlock(const char* session_id, int32_t pid);
int qml_session_lock_state(const char* session_id, int32_t* out_owner_pid, int* out_locked);

//------------------------------------------------------------
//Func : qml_session_last_pos / qml_session_append
//position-offset precision for cross-engine token log continuity. See
//qml_sqlite_append_tokens's docs (storage/sqlite/qml_sqlite_store.h)
//-- use qml_session_append instead of qml_session_save's
//own token-writing (which always starts at pos 0, correct for a full
//within-engine resend) when token_ids is an INCREMENTAL addition to a
//session another engine (or a prior turn) already has qml_token_log
//rows for.
//  in: session_id — session key
//  in: out_last_pos — (output, get_last_pos) highest recorded position
//      (-1 when the session has never recorded a token)
//  in: token_ids / n_tokens — (append_tokens) incremental tokens, appended
//      starting at the stored last_pos + 1
//  in: seq_id — (append_tokens) engine sequence id recorded with the rows
// out: get_last_pos — 0 found (*out_last_pos filled), 1 session unknown,
//      -1 on failure
// out: append_tokens — 0 on success, -1 on failure
//------------------------------------------------------------
int qml_session_last_pos(const char* session_id, int32_t* out_last_pos);
int qml_session_append(const char* session_id, const int32_t* token_ids, size_t n_tokens, int32_t seq_id);

//------------------------------------------------------------
//Func : qml_window_trim (core/qml_window.c)
//TAWT (Turn-Aligned Window Trim) in the core: when the session's token log
//exceeds `window` tokens, keep [system prefix + most recent turns], move the
//excised middle losslessly to qml_token_log_archive, renumber the live log,
//drop the now-stale KV blob, and invalidate the session's FTS index rows.
//Call where no live KV exists for the session (the restore/admission path).
//  in: session_id — session to consider
//  in: window — restore-window token budget (0 = disabled)
//  in: eot_token — end-of-turn token id (turn-boundary marker)
// out: tokens archived (0 = no-op), -1 on storage error
//------------------------------------------------------------
long qml_window_trim(const char * session_id, size_t window,
                           int32_t eot_token);

//------------------------------------------------------------
//Func : qml_rig_query / qml_rig_n_words / qml_rig_gate /
//       qml_rig_match_count
//(core/qml_rig.c) retrieval policy shared by every engine hook: the FTS5
//MATCH query builder (role/Korean-function-word/all-digit exclusions), the
//effective-word counter, the RIG injection verdict
//(rig_score = relative-sim x info, threshold conf; conf <= 0 = gate off),
//and the absolute overlap counter (distinct query-eligible words of the
//question that literally occur in a candidate text — the literal-match
//floor addressing TEST-LEDGER defect #11's relative-sim limit).
//  in: text — raw new-turn text (build_fts_query / n_eff)
//  in: score / best_score / n_eff / n_min / conf — see qml_rig.c contract
//  in: question / candidate — see qml_rig.c contract (query_match_count)
// out: build_fts_query — malloc'd MATCH string or NULL (caller free());
//      n_eff — non-stopword word count; rig_gate — 1 inject, 0 skip;
//      query_match_count — matched distinct query-word count (>= 0)
//------------------------------------------------------------
char * qml_rig_query(const char * text);
int    qml_rig_n_words(const char * text);
int    qml_rig_gate(double score, double best_score,
                       int n_eff, int n_min, double conf);
int    qml_rig_match_count(const char * question, const char * candidate);
int    qml_rig_contains(const int32_t * hay, size_t n_hay,
                             const int32_t * needle, size_t n_needle);
size_t qml_window_cut(const int32_t * toks, size_t n, size_t lo, int32_t eot);

#ifdef __cplusplus
}
#endif

#endif // QML_ROUTER_H