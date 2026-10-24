// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.

// storage/sqlite/qml_sqlite_store.h
//
// SQLite-backed session metadata + token log manager.
// Heavy tensor blobs stay in LMDB (see storage/lmdb/qml_lmdb_store.h);
// this store only ever touches small, queryable rows.

#ifndef QML_SQLITE_STORE_H
#define QML_SQLITE_STORE_H

#include "qml_format.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------
//qml_llama_session_meta_t
//mirrors one row of the qml_session table. embeds the universal
//qml_session_meta_t (see qml_format.h) as its first member -- `.base`
//holds user_id/session_id/engine_type/seq_id/last_token/last_pos/
//n_tokens/updated_at; the fields below are llama.cpp-specific.
//------------------------------------------------------------
typedef struct
{
    qml_session_meta_t base;
    char    model_hash[65]; // 64 hex chars (SHA256) + NUL -- see core/qml_model_hash.c
    char    lmdb_key[256];
    int64_t state_size;
    // slot geometry the KV blob was written under (2026-08-02). n_stream is
    // the value llama.cpp stores at the head of every per-sequence blob
    // (= kv_unified ? 1 : n_seq_max) and hard-checks on restore, so keeping
    // it here lets the restore path reject a doomed blob WITHOUT reading it.
    // 0 = unknown (written before these columns existed) -> attempt anyway.
    int64_t n_stream;
    int64_t n_ctx_seq;
} qml_llama_session_meta_t;

//------------------------------------------------------------
//qml_token_rec_t
//mirrors one row of the qml_token_log table
//------------------------------------------------------------
typedef struct
{
    int32_t pos;
    int32_t seq_id;
    int32_t token_id;
} qml_token_rec_t;

//------------------------------------------------------------
//Func : qml_sqlite_open
//opens (creating if absent) db_path, enables WAL mode, runs schema DDL
//returns 0 on success, -1 on failure
//------------------------------------------------------------
int qml_sqlite_open(const char * db_path);

//------------------------------------------------------------
//Func : qml_sqlite_close
//------------------------------------------------------------
void qml_sqlite_close(void);

//------------------------------------------------------------
//Func : qml_sqlite_copy_session
//engine-side SQL record copy of one session (base row + token log) to a new
//session_id, in ONE write transaction on the ENGINE'S OWN connection. This is
//the supported way to duplicate/import a session: an external process writing
//these tables directly was found unreadable by the running engine on the
//standard host (mixed-SQLite-runtime WAL visibility, recorded in the project's defect notes). The new
//row starts FREE/unowned; no ext row and no blob are copied -- the first visit
//rebuilds KV from the token log (the normal self-healing path).
//returns 0 ok, 1 = src missing or dst already exists, -1 error.
//------------------------------------------------------------
int qml_sqlite_copy_session(const char * src_session_id, const char * dst_session_id,
                            const char * new_user_id /* NULL/"" = keep original */);

//------------------------------------------------------------
//Func : qml_sqlite_delete_llama_ext
//Deletes only the llama ext row (session and token log are kept) -- the SQLite half of DROP_BLOB (cold
//storage). The blob itself is deleted by the caller with qml_lmdb_delete_blob.
//------------------------------------------------------------
int qml_sqlite_delete_llama_ext(const char * session_id);

//------------------------------------------------------------
//Func : qml_sqlite_uses_blob
//1 when a session row or a llama ext row refers to the LMDB blob key (or on
//error); 0 when nothing does -- the blob is an orphan.
//------------------------------------------------------------
int qml_sqlite_uses_blob(const char * key);

//------------------------------------------------------------
//Func : qml_sqlite_put_session
//insert or update the single metadata row identified by session_id
//returns 0 on success, -1 on failure
//------------------------------------------------------------
int qml_sqlite_put_session(const char * session_id, const qml_llama_session_meta_t * meta);

//Updates only the session base row -- ext (blob record) is kept (for capture-saving save)
int qml_sqlite_put_session_base(const char * session_id, const qml_session_meta_t * base);

//------------------------------------------------------------
//Func : qml_sqlite_get_session
//returns 0 if found (out_meta filled), 1 if not found, -1 on failure
//------------------------------------------------------------
int qml_sqlite_get_session(const char * session_id, qml_llama_session_meta_t * out_meta);

//------------------------------------------------------------
//Func : qml_sqlite_recent / qml_sqlite_recent_free
//lists up to `limit` session_ids of the given engine_type, most
//recently updated first. exists for --qml-restore-on-start auto,
//which needs "the N sessions most likely to be asked for again"
//without the operator having to name them. release the array with
//the _free() call. returns 0 on success, -1 on failure.
//------------------------------------------------------------
int  qml_sqlite_recent(int engine_type, size_t limit, char *** out_ids, size_t * out_n);
void qml_sqlite_recent_free(char ** ids, size_t n);

//------------------------------------------------------------
//Func : qml_sqlite_put_vllm_session
//insert or update the base row (qml_session) + vLLM extension row
//(qml_vllm_session_ext) for session_id, in one transaction. mirrors
//qml_sqlite_put_session's llama-specific counterpart.
//returns 0 on success, -1 on failure
//------------------------------------------------------------
int qml_sqlite_put_vllm_session(const char * session_id, const qml_vllm_session_meta_t * meta);

//------------------------------------------------------------
//Func : qml_sqlite_get_vllm_session
//returns 0 if found (out_meta filled), 1 if not found, -1 on failure
//------------------------------------------------------------
int qml_sqlite_get_vllm_session(const char * session_id, qml_vllm_session_meta_t * out_meta);

//------------------------------------------------------------
//Func : qml_sqlite_delete_session
//removes the base row, both engines' extension rows (whichever exists
//is affected, the other is a no-op delete), and all token_log rows for
//session_id. returns 0 on success, -1 on failure
//------------------------------------------------------------
int qml_sqlite_delete_session(const char * session_id);

//------------------------------------------------------------
//Func : qml_sqlite_put_tokens
//appends n_tokens rows in a single transaction
//returns 0 on success, -1 on failure
//------------------------------------------------------------
//Func : qml_sqlite_block_touch / _total_bytes / _evict_lru / _evict_free
//LRU accounting for content-addressed KV blocks held in LMDB (task #56).
//qml_sqlite_block_touch() must be called on every block_put AND every hit so
//eviction is LRU and not FIFO. qml_sqlite_block_evict_lru() removes the coldest
//rows until the tracked total fits max_bytes and hands the caller the keys to
//delete from the blob store (it never touches LMDB itself); release the array
//with qml_sqlite_block_evict_free(). max_bytes <= 0 disables eviction.
//all return 0 on success, -1 on failure.
//------------------------------------------------------------
//qml_sqlite_block_touch_many() is the same statement over n_keys of equal size
//inside ONE transaction; a restore touches thousands of blocks at once and
//per-statement commits there cost more than the KV read itself.
int  qml_sqlite_block_touch(const char * block_key, int64_t n_bytes);
int  qml_sqlite_block_touch_many(const char * const * block_keys, size_t n_keys, int64_t n_bytes);

//Func : qml_sqlite_owner_add / qml_sqlite_owner_take / qml_sqlite_free_keys
//block ownership (qml_kv_owner): add records that a session stored these
//blocks; take removes the session's records plus the accounting rows of the
//blocks no other session recorded, and returns those keys for LMDB deletion
int  qml_sqlite_owner_add(const char * owner, const char * const * keys, size_t n);
int  qml_sqlite_owner_take(const char * owner, char *** out_keys, size_t * out_n);
void qml_sqlite_free_keys(char ** keys, size_t n);
int  qml_sqlite_block_total_bytes(int64_t * out_total);
int  qml_sqlite_block_evict_lru(int64_t max_bytes, char *** out_keys, size_t * out_n);
void qml_sqlite_block_evict_free(char ** keys, size_t n);

//------------------------------------------------------------
//Func : qml_sqlite_archive_rewrite
//preserves token history the ENGINE discarded (llama-server context shift)
//before a full-snapshot save overwrites qml_token_log with its shortened
//window. No-op when the stored log is a prefix of the new window (pure growth).
//returns the number of tokens archived, or -1 on failure (callers should still
//continue with the save -- see the .c-side comment).
//------------------------------------------------------------
int qml_sqlite_archive_rewrite(const char * session_id, const int32_t * new_tokens, size_t n_new);

//------------------------------------------------------------
//Func : qml_sqlite_replace_tokens
//atomically makes the stored token log for session_id EXACTLY the given
//n_tokens rows (DELETE + INSERT in one transaction). Use for full-snapshot
//saves -- qml_sqlite_put_tokens leaves a stale tail behind when a
//later save is shorter than an earlier one (see its .c-side comment for the
//live corruption bug this prevents).
//returns 0 on success, -1 on failure
//------------------------------------------------------------
int qml_sqlite_replace_tokens(const char * session_id, const qml_token_rec_t * tokens, size_t n_tokens);

//------------------------------------------------------------
//Func : qml_sqlite_save_session
//full save in ONE transaction: session row (+ llama ext row when meta is not
//NULL), archive of rows a context shift discards, and the complete token log.
//A failure rolls everything back, so a session row is never left pointing at
//an empty token log. meta NULL -> base row only (capture-throttle save).
//clear_when_empty: when n_tokens is 0, 1 clears the stored log, 0 leaves it.
//returns 0 on success, -1 on failure (reason logged on stderr)
//------------------------------------------------------------
int qml_sqlite_save_session(const char * session_id,
                            const qml_llama_session_meta_t * meta,
                            const qml_session_meta_t * base,
                            const int32_t * token_ids, size_t n_tokens,
                            int32_t seq_id, int clear_when_empty);

//------------------------------------------------------------
int qml_sqlite_put_tokens(const char * session_id, const qml_token_rec_t * tokens, size_t n_tokens);

//------------------------------------------------------------
//Func : qml_sqlite_get_tokens
//loads the full token log for session_id ordered by pos ASC.
//*out_tokens is malloc'd — caller must free() it.
//returns 0 on success (out_n may be 0), -1 on failure
//------------------------------------------------------------
int qml_sqlite_get_tokens(const char * session_id, qml_token_rec_t ** out_tokens, size_t * out_n);

//------------------------------------------------------------
//Func : qml_sqlite_last_pos
//returns 0 if session_id found (*out_last_pos filled, -1 if this session
//has never had any tokens recorded -- matches qml_session's own
//last_pos DEFAULT), 1 if session_id doesn't exist at all, -1 on failure.
//------------------------------------------------------------
int qml_sqlite_last_pos(const char * session_id, int32_t * out_last_pos);

//------------------------------------------------------------
//Func : qml_sqlite_append_tokens
//Cross-engine/incremental counterpart to qml_sqlite_put_tokens:
//instead of the caller supplying pos for each token (typically 0..N-1,
//a full resend/overwrite of everything known about this session -- the
//convention every engine's own within-engine save path already uses and
//should keep using), this looks up the session's CURRENT last_pos
//(qml_sqlite_last_pos) and appends token_ids starting at
//last_pos+1. Lets a session handed off from a different engine (or a
//second engine's own genuinely-incremental turn) continue the position
//sequence instead of colliding with/overwriting rows 0..N-1 that a
//different engine already wrote for this session_id.
//Does NOT touch/require an existing qml_session row beyond reading
//last_pos from it (0 if absent) -- caller is still responsible for its
//own qml_session upsert.
//returns 0 on success, -1 on failure.
//------------------------------------------------------------
int qml_sqlite_append_tokens(const char * session_id, const int32_t * token_ids, size_t n_tokens, int32_t seq_id);

//------------------------------------------------------------
//Func : qml_sqlite_free_tokens
//frees the array returned by qml_sqlite_get_tokens. Exists so ctypes
//callers (hooks/vllm/qml_store_bindings.py) never need to reach into
//libc's free() directly -- everything crosses this API's own boundary.
//safe to call with tokens == NULL.
//------------------------------------------------------------
void qml_sqlite_free_tokens(qml_token_rec_t * tokens);

//------------------------------------------------------------
//qml_control_cmd_t
//mirrors one row of qml_control_cmd (see docs/udf-control-plane-proposal.md).
//status is not included -- callers of qml_sqlite_control_claim
//only ever see rows this function itself just transitioned to 'CLAIMED',
//so the value is implied, not read back.
//------------------------------------------------------------
typedef struct
{
    int64_t cmd_id;
    char    target_engine[32];
    char    cmd_type[32];
    char    session_id[256];
    int64_t created_at;
} qml_control_cmd_t;

//------------------------------------------------------------
//Func : qml_sqlite_control_put
//inserts one PENDING command row, plus one qml_control_cmd_blocks row per
//entry in block_hashes (pass NULL/0 for cmd_types with no list payload,
//e.g. RESTORE/MIGRATE/BACKUP_NOW/MONITOR_PING/COMPACT_NOW -- only PREWARM
//is expected to use this). Writes the new row's cmd_id to *out_cmd_id.
//returns 0 on success, -1 on failure
//------------------------------------------------------------
int qml_sqlite_control_put(
    const char * target_engine,
    int32_t      target_pid,
    const char * cmd_type,
    const char * session_id,
    const char * const * block_hashes,
    size_t       n_block_hashes,
    int64_t    * out_cmd_id);

//------------------------------------------------------------
//Func : qml_sqlite_control_claim
//atomically claims (PENDING -> CLAIMED) the oldest pending command
//targeting target_pid, if any. The SELECT-then-UPDATE is wrapped in one
//BEGIN IMMEDIATE transaction so two threads/processes racing this call
//never both claim the same row (this is the actual mechanism behind
//docs/udf-control-plane-proposal.md's blocker #1 resolution -- callers
//are expected to invoke this from a polling thread, e.g. an extended
//hooks/llama.cpp/qml_llama_hook.cpp's persister_main()).
//returns 0 if a row was claimed (out_cmd filled), 1 if none pending,
//-1 on failure
//------------------------------------------------------------
int qml_sqlite_control_claim(int32_t target_pid, qml_control_cmd_t * out_cmd);

//------------------------------------------------------------
//Func : qml_sqlite_control_set_status
//status should be one of 'DONE' | 'FAILED' (or 'PENDING'/'CLAIMED', though
//callers normally only ever move a claimed row forward to a terminal state)
//returns 0 on success, -1 on failure
//------------------------------------------------------------
int qml_sqlite_control_set_status(int64_t cmd_id, const char * status);

//------------------------------------------------------------
//Func : qml_sqlite_control_blocks
//loads the block_hash list for cmd_id (PREWARM's list payload).
//*out_hashes is malloc'd (array of fixed-size buffers) — caller must free().
//returns 0 on success (out_n may be 0), -1 on failure
//------------------------------------------------------------
typedef struct
{
    char block_hash[80]; // sha256 hex = 64 chars + NUL, rounded up
} qml_control_cmd_block_t;

int qml_sqlite_control_blocks(int64_t cmd_id, qml_control_cmd_block_t ** out_hashes, size_t * out_n);

//------------------------------------------------------------
//Func : qml_sqlite_compact
//runs a WAL checkpoint (TRUNCATE, so the -wal file shrinks back down)
//followed by VACUUM. Only touches the SQLite metadata db -- LMDB
//compaction (mdb_env_copy with MDB_CP_COMPACT) is a separate, not yet
//implemented concern (see docs/udf-control-plane-proposal.md's
//COMPACT_NOW row). Blocks the caller for the duration (VACUUM rewrites
//the whole file) -- callers on a latency-sensitive thread should not
//call this directly from that thread.
//returns 0 on success, -1 on failure
//------------------------------------------------------------
int qml_sqlite_compact(void);

//------------------------------------------------------------
//Func : qml_sqlite_lock
//split-brain guard for the UDF control plane (docs/udf-control-plane-
//proposal.md's blocker #2). session_id must already exist as a row (via
//qml_sqlite_put_session/qml_sqlite_put_vllm_session) -- a session
//unknown to the base table can't be locked.
//returns 0 if acquired (either previously FREE, or already owned by this
//exact pid -- re-acquiring is a no-op success), 1 if held by a DIFFERENT
//pid (conflict -- caller must not proceed), -1 on failure (e.g. unknown
//session_id).
//------------------------------------------------------------
int qml_sqlite_lock(const char * session_id, int32_t pid);

//------------------------------------------------------------
//Func : qml_sqlite_unlock
//only releases if currently owned by this exact pid.
//returns 0 on success (released, or was already FREE/owned by someone
//else -- releasing something you don't own is simply a no-op, not an
//error), -1 on failure.
//------------------------------------------------------------
int qml_sqlite_unlock(const char * session_id, int32_t pid);

//------------------------------------------------------------
//Func : qml_sqlite_lock_state
//returns 0 if found (out_owner_pid/out_locked filled; out_locked is 1 if
//lock_state=='LOCKED', 0 if 'FREE'), 1 if session_id not found, -1 on
//failure.
//------------------------------------------------------------
int qml_sqlite_lock_state(const char * session_id, int32_t * out_owner_pid, int * out_locked);

//===========================================================================//
// [FTS-RECALL] BEGIN — archive full-text search primitives.
// Feature-flagged, additive-only block (engine backlog item 8: automatic
// re-injection of archived turns). To remove the feature entirely, delete
// every block between "[FTS-RECALL] BEGIN/END" markers across:
//   storage/sqlite/qml_sqlite_store.{h,c}
//   include/qml_llama_hook.h, hooks/llama.cpp/qml_llama_hook.cpp
//   third_party/llama.cpp/tools/server/server-context.cpp
// Nothing outside these blocks depends on them; runtime default is OFF
// (env QML_ARCHIVE_RECALL=1 enables).
//===========================================================================//

//Func : qml_sqlite_fts_status
// in: session_id — session address
//out: 2 = session has no archived rows (nothing to recall);
//     1 = FTS index rows exist for this session (ready to search);
//     0 = archive rows exist but not yet indexed (caller should index);
//    -1 = error or SQLite built without FTS5 (feature silently unavailable)
int qml_sqlite_fts_status(const char * session_id);

//Func : qml_sqlite_archive_fetch
// in: session_id — session address
//out: 0 on success; *out_pos/*out_tok are malloc'd arrays of *out_n archive
//     rows (original positions + token ids, ordered by orig_pos); caller
//     frees both. -1 on error / no rows.
int qml_sqlite_archive_fetch(const char * session_id,
                             int32_t ** out_pos, int32_t ** out_tok, size_t * out_n);

//Func : qml_sqlite_fts_add
// in: session_id — session address
// in: turn_pos / turn_end — original-position range [turn_pos, turn_end) of one turn
// in: content — detokenized plain text of that turn (FTS-indexed column)
//out: 0 on success, -1 on error
int qml_sqlite_fts_add(const char * session_id,
                                  int64_t turn_pos, int64_t turn_end,
                                  const char * content);

// [FTS-BULK-FIX 2026-08-14] When inserting many documents, wrap them with begin/end.
// Without it a WAL commit happens for every single document — the first index build of 19,446 documents
// took 77.7 s (NFS storage, about 4ms per document), and that cost is added directly to the first question
// after reconnecting. To revert, delete these two declarations, the call sites and the implementation.
//Func : qml_sqlite_trim_head_get / _set
//Per-session stored value of the head (system prefix) exemption decision (-1 = not decided). Decided once at
//the first trim and stored; later trims reuse it — fixes the defect where, after renumbering, a front turn
//was mistaken for system and locked in as a permanent exemption (2026-09-01).
long qml_sqlite_trim_head_get(const char * session_id);
int  qml_sqlite_trim_head_set(const char * session_id, long head_len);

//Func : qml_sqlite_trim_apply
//window-trim record surgery (TAWT storage half; policy in core/qml_window.c):
//archives log rows [head_len, cut), renumbers the rest from 0, updates the
//session counters — one IMMEDIATE transaction with in-transaction boundary
//re-validation (both boundary rows must still end with eot_token).
//returns tokens archived; 0 = boundaries no longer valid (no-op); -1 = error
long qml_sqlite_trim_apply(const char * session_id, size_t head_len,
                           size_t cut, int32_t eot_token);

//Func : qml_sqlite_fts_drop
//drops a session's FTS index rows so the next recall lazily rebuilds them
//over the grown archive (needed after a trim; the lazy indexer runs once)
//returns 0 on success, -1 on error
int qml_sqlite_fts_drop(const char * session_id);

//Func : qml_sqlite_fts_begin
//out: 0 on success, -1 on error. Nested calls are harmless (returns 0 if already open).
int qml_sqlite_fts_begin(void);
//Func : qml_sqlite_fts_end
// in: commit — nonzero means COMMIT, 0 means ROLLBACK
//out: 0 on success, -1 on error
int qml_sqlite_fts_end(int commit);

//Func : qml_sqlite_fts_search
// in: session_id — session address (search is scoped to this session ONLY —
//     identity first, content second; never a cross-session content lookup)
// in: match_query — FTS5 MATCH expression (caller sanitizes)
// in: k — max hits
// in: out_score — [RIG-GATE] optional array of size k receiving each hit's BM25
//     rank value as FTS5 returns it (negative; smaller = better match). Pass
//     NULL to skip. Used by the RIG decision stage to derive a similarity term.
//out: number of hits written to out_turn_pos/out_turn_end (arrays of size k),
//     ordered by FTS rank; -1 on error
int qml_sqlite_fts_search(const char * session_id, const char * match_query,
                                  int k, int64_t * out_turn_pos, int64_t * out_turn_end,
                                  double * out_score);

//Func : qml_sqlite_cond_search
// [COND-SUBSET 2026-08-31] Finds reference material by condition-bit decision. Unlike lexical and
// embedding search, it decides by **containment**, not by rank:
// if all condition bits of a stored item are contained in the query (case) bits, it is a candidate.
// Unit test (tests/unit_rule_search.py, 63 decisions): lexical search top-1 46~58.7%,
// embedding 41.3%, this method 100%. Narrowing first with BM25 drops it to 88.9%, so
// no pre-filter is used and everything is scanned (43 byte operation).
// in: session_id — search scope (the session where the reference material is loaded)
// in: case_code / n_code — condition bits of the case (the deployment decides the position scheme)
// in: k — maximum number of candidates. Passing only the top-1 for recall is the confirmed rule
//     (adding candidates makes the model fail to pick and lowers accuracy).
//out: number of candidates. Items with more (more specific) conditions first; ties keep the load order
//     — mixing another key into tie ordering drops top-1 to 90.5% (measured).
//Func : qml_sqlite_cond_put
// [COND-SUBSET 2026-08-31] Records the condition bits of one reference item. At load
// time the deployment decides the position scheme (that deployment's `field=value` list) and passes a bit
// vector with only the condition positions that item requires set to 1. cond_n is the number of conditions —
// used only for search order (more specific items first).
// in: n_cond — number of conditions (0 excludes the item from condition search)
//out: 0 success, -1 error
int qml_sqlite_cond_put(const char * session_id, int64_t turn_pos,
                                int64_t turn_end, const unsigned char * cond_code,
                                int n_code, int n_cond);

// out_sid: which session each candidate came from (it is an owner-scope search, so it can differ from
//     the decision session). The caller must read tokens from this session.
//     Omitted if NULL.
int qml_sqlite_cond_search(const char * session_id,
                                   const unsigned char * case_code, int n_code,
                                   int k, int64_t * out_turn_pos, int64_t * out_turn_end,
                                   char out_sid[][256]);

// [HYBRID 2026-08-27] archive turn embedding vectors (the storage half of semantic search)
//Func : qml_sqlite_vec_add
// in: session_id / turn_pos / turn_end — coordinates of the matching archive turn
// in: vec / dim — float32 embedding (stored as a BLOB)
//out: 0 on success, -1 on error
int qml_sqlite_vec_add(const char * session_id, int64_t turn_pos,
                               int64_t turn_end, const float * vec, size_t dim);
//Func : qml_sqlite_vec_missing
// in: session_id / max_n — up to max_n turns that are in the FTS index but have no vector
//out: count (>=0); coordinates in the out_pos/out_end arrays (size max_n). -1 on error
int qml_sqlite_vec_missing(const char * session_id, int64_t max_n,
                                   int64_t * out_pos, int64_t * out_end);
//Func : qml_sqlite_fts_search_text
// The same MATCH search as fts_search, but returns the content text directly —
// for paths that inject the text itself, such as context summaries (tag='fact' rows).
// out_txt: array of size k, each element a UTF-8 body the caller must free()
//out: number of returned rows (>=0), -1 on error
// out_pos: NULL allowed — turn_pos of each row (coordinate for deduplication)
int qml_sqlite_fts_search_text(const char * session_id,
                                      const char * match_query,
                                      int k, char ** out_txt,
                                      int64_t * out_pos);
//Func : qml_sqlite_vec_missing_text
// Same as vec_missing but also returns the body (content) — the FTS5 auxiliary columns are
// UNINDEXED, so a column equality lookup is a full scan; if only coordinates are returned and each body
// is read again per document with qml_sqlite_fts_text, the scan repeats once per document
// (2026-09-02 measured: 1.3s/call × 485 calls on a 1.48-million-row store = 10.5 minutes per session).
// This function extracts coordinates + body together in a single pass of the same scan.
// out_txt: array of size max_n — each element a UTF-8 body the caller must free()
//out: count (>=0). -1 on error
int qml_sqlite_vec_missing_text(const char * session_id, int64_t max_n,
                                       int64_t * out_pos, int64_t * out_end,
                                       char ** out_txt);
//------------------------------------------------------------
//Func : qml_sqlite_entity_add / _lookup   [ENTITY-INDEX 2026-08-29]
// Indexes identifiers extracted from archive turns by deterministic rules (pattern id, etc.), and
// when the query has the same identifier, recalls by condition match without rank competition.
//  in: entity — identifier string (case-sensitive, verbatim)
//  in: max_hits / out_pos / out_end — recall limit and result arrays
// out: add = 0/-1, lookup = number of recalled rows (in latest turn_pos order) or -1
//------------------------------------------------------------
int qml_sqlite_entity_add(const char * session_id, const char * entity,
                                  int64_t turn_pos, int64_t turn_end);
//  peer version: also queries other sessions of the same owner ("owner::" prefix) —
//  sessions of a different owner are not crossed (identity isolation is kept). out_sid receives the session id
//  of each result (because tokens must be read from that session's archive).
int qml_sqlite_entity_find_peers(const char * owner_prefix,
                                          const char * entity, int max_hits,
                                          char out_sid[][128], int64_t * out_pos,
                                          int64_t * out_end);
int qml_sqlite_entity_find(const char * session_id, const char * entity,
                                     int max_hits, int64_t * out_pos,
                                     int64_t * out_end);

//Func : qml_sqlite_vec_search
// in: session_id / qvec / dim — query embedding. k — top k (<=16)
//out: count; out_pos/out_end/out_score (cosine, larger is more similar). Session-scoped
//     brute-force cosine — measured tens of ms at 19,446 documents (TEST-LEDGER A-2 item 22)
int qml_sqlite_vec_search(const char * session_id, const float * qvec,
                                  size_t dim, int k, int64_t * out_pos,
                                  int64_t * out_end, double * out_score);

//Func : qml_sqlite_get_tail
// in: session_id — session address
// in: max_n — read at most this many of the NEWEST log tokens
//out: 0 on success; *out_ids = malloc'd token ids in position order (caller
//     frees), *out_n <= max_n. -1 on error. Used by the RIG in-context
//     duplicate guard (scan window over the live log tail).
int qml_sqlite_get_tail(const char * session_id, size_t max_n,
                                  int32_t ** out_ids, size_t * out_n);

//Func : qml_sqlite_fts_text
// in: session_id — session address (same identity-first scoping as the search)
// in: turn_pos — the hit's turn_pos as returned by qml_sqlite_fts_search
//out: malloc'd copy of the indexed content text for that row (caller frees),
//     or NULL when the row is missing / on error. Used by the RIG absolute
//     match check (qml_rig_match_count) after the relative gate passes.
char * qml_sqlite_fts_text(const char * session_id, int64_t turn_pos);

//Func : qml_sqlite_archive_range
// in: session_id — session address
// in: pos_start / pos_end — original-position range [pos_start, pos_end)
//out: 0 on success; *out_tok = malloc'd token ids (*out_n of them) in position
//     order; caller frees. -1 on error / empty range.
int qml_sqlite_archive_range(const char * session_id,
                             int64_t pos_start, int64_t pos_end,
                             int32_t ** out_tok, size_t * out_n);
//===========================================================================//
// [FTS-RECALL] END
//===========================================================================//

#ifdef __cplusplus
}
#endif

#endif // QML_SQLITE_STORE_H
