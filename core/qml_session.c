// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
// core/qml_session.c
//
// Implements the qml_router.h session bridge API declared for the
// llama.cpp LD_PRELOAD hook (see hooks/llama.cpp/qml_llama_preload.cpp).
// Engine-agnostic: only raw byte buffers / token arrays cross this file's
// boundary, nothing here knows about llama_context or ggml_tensor.

#include "qml_router.h"
#include "qml_format.h"
#include "qml_sqlite_store.h"
#include "qml_lmdb_store.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

//------------------------------------------------------------
//qml_llama_session
//live per-context tracking handle, created in qml_session_begin and
//destroyed in qml_session_end
//------------------------------------------------------------
struct qml_llama_session
{
    char    user_id[128];
    char    session_id[256];
    int32_t seq_id;
    char    model_hash[65]; // 64 hex chars (SHA256) + NUL -- see core/qml_model_hash.c

    int32_t restore_offset;      // n_tokens carried over from a prior session (0 if fresh)

    void * restore_buf;          // malloc'd blob copy handed out by qml_session_restore;
                                 // freed in qml_session_restore_done (or qml_session_end)

    qml_token_rec_t * log_buf;   // accumulated (pos, seq_id, token_id) rows this run
    size_t            log_len;
    size_t            log_cap;
};

//------------------------------------------------------------
//slot geometry of the live context, set by the engine hook (see
//qml_session_set_geometry). 0 means "unknown" -- every check below then
//behaves exactly as it did before this existed.
//------------------------------------------------------------
static uint32_t g_n_stream  = 0;
static uint32_t g_n_ctx_seq = 0;

// Records the live context's slot geometry so restore paths can refuse mismatched blobs.
//  in: n_stream — number of parallel KV streams of the live context (0 = unknown)
//  in: n_ctx_seq — per-sequence context length of the live context (0 = unknown)
// out: none
// side effects: sets the file-global g_n_stream / g_n_ctx_seq used by geometry checks
void qml_session_set_geometry(uint32_t n_stream, uint32_t n_ctx_seq)
{
    g_n_stream  = n_stream;
    g_n_ctx_seq = n_ctx_seq;
}

//------------------------------------------------------------
//Func : geometry_rejects
//true when the stored blob was written under a different n_stream than the
//live context uses. The engine would throw "n_stream mismatch" deep inside
//state_read; refusing here saves the whole blob read. Unknown on either
//side (0) means "cannot tell" -- try, exactly as before.
//  in: meta — stored session metadata whose recorded n_stream is compared to the live one
// out: 1 = refuse the blob (n_stream mismatch); 0 = proceed (match, or either side unknown)
//------------------------------------------------------------
static int geometry_rejects(const qml_llama_session_meta_t* meta)
{
    return (g_n_stream != 0 && meta->n_stream != 0 &&
            (uint32_t) meta->n_stream != g_n_stream);
}

//------------------------------------------------------------
//Func : qml_store_init
// Opens both storage tiers (SQLite metadata/token log + LMDB blob store).
//  in: sqlite_path — filesystem path of the SQLite database file
//  in: lmdb_path — directory path of the LMDB environment
// out: 0 on success; -1 if either store fails to open
// side effects: opens global store handles; reads QML_LMDB_MAP_SIZE_MB from the environment
//------------------------------------------------------------
int qml_store_init(const char* sqlite_path, const char* lmdb_path)
{
    if (qml_sqlite_open(sqlite_path) != 0)
    {
        return -1;
    }

    // 4GiB of reserved address space; LMDB only backs pages it actually
    // touches, so this costs no physical memory upfront
    // LMDB map size. This used to be a hard 4 GiB, which silently caps the whole
    // store: an 8K-token session on Llama-3.1-8B with f16 KV is ~1 GiB of blob
    // (128 KB/token measured), so a handful of long sessions would hit
    // MDB_MAP_FULL. The map is sparse -- it reserves address space, not disk --
    // so a large default costs nothing until blobs are actually written.
    // Override with QML_LMDB_MAP_SIZE_MB when the address space or the
    // filesystem needs a smaller reservation.
    size_t      map_size_bytes = (size_t) 64 * 1024 * 1024 * 1024; // 64 GiB
    const char* map_env        = getenv("QML_LMDB_MAP_SIZE_MB");
    if (map_env != NULL)
    {
        const long long mb = atoll(map_env);
        if (mb > 0)
        {
            map_size_bytes = (size_t) mb * 1024 * 1024;
        }
    }

    if (qml_lmdb_open(lmdb_path, map_size_bytes) != 0)
    {
        return -1;
    }

    return 0;
}

//------------------------------------------------------------
//Func : qml_store_close
// Closes both storage tiers opened by qml_store_init.
//  in: none
// out: none
// side effects: closes the global SQLite and LMDB handles
//------------------------------------------------------------
void qml_store_close()
{
    qml_sqlite_close();
    qml_lmdb_close();
}

//------------------------------------------------------------
//Func : qml_session_begin
// Creates a live tracking handle and, if a usable prior snapshot exists, hands out its blob.
//  in: user_id — owning user identifier (copied into the handle)
//  in: session_id — session key used for the SQLite row and the LMDB blob
//  in: seq_id — engine sequence id the tracked tokens belong to
//  in: model_hash — fingerprint of the loaded model ("" skips the model check)
//  in: out_state_data — receives the stored state blob, or NULL when starting fresh
//  in: out_state_size — receives the blob size in bytes (0 when fresh)
//  in: out_last_pos — receives the stored last token position (-1 when fresh)
// out: newly allocated handle (never NULL); caller must pass it to qml_session_end.
//      *out_state_data is owned by the handle -- freed by qml_session_restore_done/_end,
//      never by the caller. Any mismatch (model, geometry, missing blob) starts fresh.
//------------------------------------------------------------
qml_llama_session_t* qml_session_begin(
    const char* user_id,
    const char* session_id,
    int32_t     seq_id,
    const char* model_hash,
    const void** out_state_data,
    size_t*      out_state_size,
    int32_t*     out_last_pos)
{
    qml_llama_session_t* sess = (qml_llama_session_t*) calloc(1, sizeof(qml_llama_session_t));

    strncpy(sess->user_id,    user_id,    sizeof(sess->user_id) - 1);
    strncpy(sess->session_id, session_id, sizeof(sess->session_id) - 1);
    strncpy(sess->model_hash, model_hash, sizeof(sess->model_hash) - 1);

    sess->seq_id = seq_id;

    sess->log_cap = 1024;
    sess->log_len = 0;
    sess->log_buf = (qml_token_rec_t*) malloc(sess->log_cap * sizeof(qml_token_rec_t));

    *out_state_data = NULL;
    *out_state_size = 0;
    *out_last_pos   = -1;

    qml_llama_session_meta_t meta;
    if (qml_sqlite_get_session(session_id, &meta) != 0)
    {
        return sess; // not found (or lookup error) -- start fresh
    }

    if (model_hash[0] != '\0' && strcmp(model_hash, meta.model_hash) != 0)
    {
        // stored state belongs to a different model -- restoring its KV
        // tensors here would be reading past the wrong shapes/layer count
        return sess;
    }

    if (geometry_rejects(&meta))
    {
        return sess; // different slot geometry -- start fresh (token log path)
    }

    void*  data = NULL;
    size_t size = 0;

    if (qml_lmdb_get_blob(meta.lmdb_key, &data, &size) != 0)
    {
        return sess; // metadata row exists but blob missing/corrupt -- start fresh
    }

    sess->restore_buf    = data;
    sess->restore_offset = (int32_t) meta.base.n_tokens;

    *out_state_data = data;
    *out_state_size = size;
    *out_last_pos   = meta.base.last_pos;

    return sess;
}

//------------------------------------------------------------
//Func : qml_session_restore_done
// Releases the restore blob handed out by qml_session_begin once the engine has consumed it.
//  in: sess — session handle (NULL-safe; no-op when there is no pending restore buffer)
// out: none
//------------------------------------------------------------
void qml_session_restore_done(qml_llama_session_t* sess)
{
    if (sess == NULL || sess->restore_buf == NULL)
    {
        return;
    }

    free(sess->restore_buf);
    sess->restore_buf = NULL;
}

//------------------------------------------------------------
//Func : qml_session_track_decode
// Records one decode batch into the session's token log, shifting positions past the restore.
//  in: sess — session handle (NULL-safe no-op)
//  in: token_ids — token ids of the batch, n_tokens entries
//  in: pos — token positions; MODIFIED IN PLACE: each gets restore_offset added
//  in: seq_ids — sequence id per token, n_tokens entries
//  in: n_tokens — number of tokens in the batch
// out: none (log buffer grows by doubling as needed)
//------------------------------------------------------------
void qml_session_track_decode(
    qml_llama_session_t* sess,
    const int32_t* token_ids,
    int32_t*       pos,
    const int32_t* seq_ids,
    size_t         n_tokens)
{
    if (sess == NULL)
    {
        return;
    }

    for (size_t i = 0; i < n_tokens; ++i)
    {
        pos[i] += sess->restore_offset;

        if (sess->log_len == sess->log_cap)
        {
            sess->log_cap *= 2;
            sess->log_buf  = (qml_token_rec_t*) realloc(sess->log_buf, sess->log_cap * sizeof(qml_token_rec_t));
        }

        sess->log_buf[sess->log_len].pos      = pos[i];
        sess->log_buf[sess->log_len].seq_id   = seq_ids[i];
        sess->log_buf[sess->log_len].token_id = token_ids[i];
        ++sess->log_len;
    }
}

//------------------------------------------------------------
//Func : qml_session_has_new
// Tells whether any tokens were tracked since the session began (i.e. a save is worthwhile).
//  in: sess — session handle (NULL-safe)
// out: 1 if at least one token was tracked; 0 otherwise (incl. NULL handle)
//------------------------------------------------------------
int qml_session_has_new(qml_llama_session_t* sess)
{
    return (sess != NULL && sess->log_len > 0) ? 1 : 0;
}

//------------------------------------------------------------
//Func : qml_session_end
// Persists the session (only if tokens were tracked) and destroys the handle.
//  in: sess — session handle to finalize and free (NULL-safe)
//  in: state_data — engine state blob to store, or NULL to skip persisting
//  in: state_size — blob size in bytes (0 skips persisting)
// out: none; sess and its buffers are freed unconditionally
// side effects: writes the LMDB blob and upserts the SQLite session/token-log rows
//------------------------------------------------------------
void qml_session_end(qml_llama_session_t* sess, const void* state_data, size_t state_size)
{
    if (sess == NULL)
    {
        return;
    }

    // some hosts (this llama-cli build included) construct a throwaway
    // "probe" llama_context that is freed again without ever calling
    // llama_decode (see qml_session_track_decode). such a context has
    // nothing new to report -- persisting here would overwrite a real,
    // previously-saved session with an empty/near-empty snapshot. only
    // persist once at least one token has actually been tracked.
    if (sess->log_len > 0 && state_data != NULL && state_size > 0)
    {
        // lmdb_key == session_id keeps the sqlite <-> lmdb mapping 1:1
        qml_lmdb_put_blob(sess->session_id, state_data, state_size);

        qml_llama_session_meta_t meta;
        memset(&meta, 0, sizeof(meta));

        strncpy(meta.base.user_id,    sess->user_id,    sizeof(meta.base.user_id) - 1);
        strncpy(meta.base.session_id, sess->session_id, sizeof(meta.base.session_id) - 1);
        strncpy(meta.model_hash,      sess->model_hash, sizeof(meta.model_hash) - 1);
        strncpy(meta.lmdb_key,        sess->session_id, sizeof(meta.lmdb_key) - 1);

        meta.base.engine_type = QML_ENGINE_LLAMA_CPP;
        meta.base.seq_id      = sess->seq_id;
        meta.state_size       = (int64_t) state_size;
        meta.base.last_token  = sess->log_buf[sess->log_len - 1].token_id;
        meta.base.last_pos    = sess->log_buf[sess->log_len - 1].pos;
        meta.base.n_tokens    = (int64_t) sess->restore_offset + (int64_t) sess->log_len;

        qml_sqlite_put_session(sess->session_id, &meta);
        qml_sqlite_put_tokens(sess->session_id, sess->log_buf, sess->log_len);
    }

    free(sess->restore_buf);
    free(sess->log_buf);
    free(sess);
}

//------------------------------------------------------------
//Func : qml_session_load
// Loads a stored snapshot: state blob plus the matching token-id array, consistency-checked.
//  in: session_id — session key to look up
//  in: model_hash — live model fingerprint ("" skips the model check)
//  in: out_state_data — receives a malloc'd copy of the state blob
//  in: out_state_size — receives the blob size in bytes
//  in: out_token_ids — receives a malloc'd token-id array
//  in: out_n_tokens — receives the token count
// out: 0 found (caller frees via qml_session_load_done); 1 nothing usable
//      (missing, model/geometry mismatch, or blob/log token-count disagreement); -1 error
//------------------------------------------------------------
int qml_session_load(
    const char*  session_id,
    const char*  model_hash,
    void**       out_state_data,
    size_t*      out_state_size,
    int32_t**    out_token_ids,
    size_t*      out_n_tokens)
{
    *out_state_data = NULL;
    *out_state_size = 0;
    *out_token_ids  = NULL;
    *out_n_tokens   = 0;

    qml_llama_session_meta_t meta;
    if (qml_sqlite_get_session(session_id, &meta) != 0)
    {
        return 1; // not found
    }

    if (model_hash[0] != '\0' && strcmp(model_hash, meta.model_hash) != 0)
    {
        // stored state belongs to a different model -- restoring its KV
        // tensors here would be reading past the wrong shapes/layer count
        return 1;
    }

    if (geometry_rejects(&meta))
    {
        return 1; // different slot geometry -- caller recomputes from the token log
    }

    void*  data = NULL;
    size_t size = 0;

    if (qml_lmdb_get_blob(meta.lmdb_key, &data, &size) != 0)
    {
        return 1; // metadata row exists but blob missing/corrupt -- treat as not found
    }

    qml_token_rec_t* recs   = NULL;
    size_t           n_recs = 0;

    if (qml_sqlite_get_tokens(session_id, &recs, &n_recs) != 0)
    {
        free(data);
        return -1;
    }

    int32_t* tok_ids = (int32_t*) malloc(n_recs * sizeof(int32_t));
    for (size_t i = 0; i < n_recs; ++i)
    {
        tok_ids[i] = recs[i].token_id;
    }
    free(recs);

    // the token log and the KV blob MUST describe the same token count --
    // meta.base.n_tokens is written by the same save that produced this blob.
    // A disagreement means the pair is inconsistent (this is exactly what the
    // 2026-07-30 corruption looked like: meta 31 vs 58 rows in the log), and
    // handing the caller more "resident" tokens than the blob actually holds
    // makes the engine attend to stale KV. Refuse instead: report not-found so
    // the caller starts fresh (correct, just slower) rather than silently
    // continuing from a broken state.
    if ((int64_t) n_recs != meta.base.n_tokens)
    {
        free(tok_ids);
        free(data);
        return 1;
    }

    *out_state_data = data;
    *out_state_size = size;
    *out_token_ids  = tok_ids;
    *out_n_tokens   = n_recs;

    return 0;
}

//------------------------------------------------------------
//Func : qml_session_load_tokens
//
//Token list + consistency checks WITHOUT reading the state blob. The
//llama-server request path peeks the token array in launch_slot_with_task()
//before deciding to restore; going through qml_session_load there
//assembled a full copy of the state blob (4.4 GiB at 128K tokens) just to
//throw it away -- measured as a whole second of the warm restore. This
//performs the same model_hash and token-log/meta pairing checks, and
//verifies the blob EXISTS (header read only), but never copies it.
//returns 0 if found, 1 if nothing usable, -1 on error.
//  in: session_id — session key to look up
//  in: model_hash — live model fingerprint ("" skips the check; mismatch is a hard miss)
//  in: out_token_ids — receives a malloc'd token-id array (caller frees)
//  in: out_n_tokens — receives the token count
// out: 0 found, 1 nothing usable, -1 error (select failure or allocation failure)
//------------------------------------------------------------
int qml_session_load_tokens(
    const char* session_id,
    const char* model_hash,
    int32_t**   out_token_ids,
    size_t*     out_n_tokens)
{
    *out_token_ids = NULL;
    *out_n_tokens  = 0;

    /* A session created by ANOTHER engine (vLLM connector, SGLang backend,
     * task #37) has a base qml_session row and a token log but NO llama
     * snapshot -- no qml_llama_session_ext row and no llama-format blob.
     * The token log is the engine-portable carrier, so any soft miss on the
     * llama-snapshot side falls through to it: the caller prepends the
     * tokens and rebuilds by prefill (found live 2026-08-04: an SGLang-made
     * session was answered with NO history because this function required
     * the llama ext row). A model_hash MISMATCH on an existing llama row
     * still refuses outright -- different weights mean the blob and log
     * belong to another model, and that is a hard error, not a miss. */
    int have_snapshot = 0;

    qml_llama_session_meta_t meta;
    if (qml_sqlite_get_session(session_id, &meta) == 0)
    {
        if (model_hash[0] != '\0' && strcmp(model_hash, meta.model_hash) != 0)
        {
            fprintf(stderr, "[qml][peek] '%s': model_hash mismatch (stored='%.12s')\n",
                    session_id, meta.model_hash);
            return 1;
        }
        size_t blob_size = 0;
        have_snapshot = (qml_lmdb_blob_stat(meta.lmdb_key, &blob_size) == 0 &&
                         blob_size != 0);
    }

    qml_token_rec_t* recs   = NULL;
    size_t           n_recs = 0;

    if (qml_sqlite_get_tokens(session_id, &recs, &n_recs) != 0)
    {
        fprintf(stderr, "[qml][peek] '%s': select_tokens ERROR\n", session_id);
        return -1;
    }

    if (n_recs == 0)
    {
        fprintf(stderr, "[qml][peek] '%s': token log EMPTY (have_snapshot=%d)\n",
                session_id, have_snapshot);
        free(recs);
        return 1; // no token log either -- genuinely nothing usable
    }

    if (have_snapshot && (int64_t) n_recs != meta.base.n_tokens)
    {
        /* llama blob exists but disagrees with the log (e.g. another engine
         * extended the session since). The log is newer; the restore path's
         * own coverage check (qml_restore_covers) discards the stale blob,
         * so returning the log tokens here is safe and self-healing. */
        fprintf(stderr,
                "[qml] session '%s': token log has %zu tokens but the llama snapshot records %lld -- using the log\n",
                session_id, n_recs, (long long) meta.base.n_tokens);
    }

    int32_t* tok_ids = (int32_t*) malloc(n_recs * sizeof(int32_t));
    if (tok_ids == NULL)
    {
        free(recs);
        return -1;
    }

    for (size_t i = 0; i < n_recs; ++i)
    {
        tok_ids[i] = recs[i].token_id;
    }
    free(recs);

    *out_token_ids = tok_ids;
    *out_n_tokens  = n_recs;

    return 0;
}

//------------------------------------------------------------
//Func : qml_session_prefetch
//
//Warms the host page cache for a session's stored KV blob WITHOUT touching
//any live KV state and WITHOUT occupying an engine slot (task #58). PREWARM
//is the expensive tier: it loads the session into one of the engine's
//n_parallel physical slots, so a mispredicted PREWARM evicts a real user's
//session (measured 350-550ms to undo). This tier only faults the LMDB pages
//in, so mispredicting costs page cache -- nothing else -- which makes it
//safe to fire for SEVERAL likely-next sessions at once.
//
//The read is a plain byte-sum over the mmap'd blob: the sum itself is
//discarded, but computing it is what forces every page to be resident, and
//volatile keeps the compiler from optimising the loop away.
//
//returns 0 if the blob was found and touched, 1 if there is nothing stored
//for this session, -1 on error.
//  in: session_id — session whose stored blob should be paged into RAM
//  in: out_bytes — optional (may be NULL); receives the touched blob size in bytes
// out: 0 blob found and paged in; 1 nothing usable (missing row/blob, or lookup
//      failure -- the current body never returns -1)
// side effects: faults the blob's pages into the OS page cache
//------------------------------------------------------------
int qml_session_prefetch(const char* session_id, size_t* out_bytes)
{
    if (out_bytes != NULL)
    {
        *out_bytes = 0;
    }

    qml_llama_session_meta_t meta;
    if (qml_sqlite_get_session(session_id, &meta) != 0)
    {
        return 1;
    }

    void*  data = NULL;
    size_t size = 0;

    // chunk-assembling the blob reads every stored page, which is exactly
    // the warming effect this tier wants; the copy is freed immediately.
    if (qml_lmdb_get_blob(meta.lmdb_key, &data, &size) != 0)
    {
        return 1;
    }

    free(data);

    if (out_bytes != NULL)
    {
        *out_bytes = size;
    }

    return 0;
}

//------------------------------------------------------------
//Func : qml_session_load_blob
//
//Same as qml_session_load but WITHOUT the token-log SELECT. Exists
//because the llama-server restore path was reading the token log TWICE per
//restore: once in launch_slot_with_task() (to prepend the history) and again
//here (to rebuild slot.prompt.tokens). On an 8K-token session that is 7,953
//rows read twice, and it showed up in the measurement: restoring the same
//280 MiB q4_0 blob took 250ms through this path versus 87ms through stock
//llama.cpp's plain file restore (/slots API), with the difference living in
//the metadata queries and the control round-trip -- not in mmap vs fread.
//
//The caller already holds the token array from its earlier peek, so it passes
//the count it expects; *out_n_tokens_meta is the count recorded by the save
//that produced this blob, and a mismatch means the caller's tokens and this
//blob do not describe the same state (another writer moved in between). The
//caller must then fall back to the full load rather than pairing them.
//
//returns 0 if found, 1 if nothing usable, -1 on error.
//  in: session_id — session key to look up
//  in: model_hash — live model fingerprint ("" skips the check)
//  in: out_state_data — receives a malloc'd copy of the state blob (caller frees)
//  in: out_state_size — receives the blob size in bytes
//  in: out_n_tokens_meta — receives the token count recorded by the save that wrote the blob
// out: 0 found; 1 nothing usable (missing, model or geometry mismatch, blob read failure)
//------------------------------------------------------------
int qml_session_load_blob(
    const char*  session_id,
    const char*  model_hash,
    void**       out_state_data,
    size_t*      out_state_size,
    int64_t*     out_n_tokens_meta)
{
    *out_state_data    = NULL;
    *out_state_size    = 0;
    *out_n_tokens_meta = 0;

    qml_llama_session_meta_t meta;
    if (qml_sqlite_get_session(session_id, &meta) != 0)
    {
        return 1;
    }

    if (model_hash[0] != '\0' && strcmp(model_hash, meta.model_hash) != 0)
    {
        return 1;
    }

    // the engine would throw "n_stream mismatch" after we finished reading --
    // refuse now and let the caller fall back to the token log (see
    // qml_session_set_geometry)
    if (geometry_rejects(&meta))
    {
        return 1;
    }

    void*  data = NULL;
    size_t size = 0;

    if (qml_lmdb_get_blob(meta.lmdb_key, &data, &size) != 0)
    {
        return 1;
    }

    *out_state_data    = data;
    *out_state_size    = size;
    *out_n_tokens_meta = meta.base.n_tokens;

    return 0;
}

//------------------------------------------------------------
//Func : qml_session_load_any
//Same as qml_session_load_blob but WITHOUT the slot-geometry
//guard, and it reports the geometry the blob was written under. For the
//one caller that can do something about a mismatch: the llama hook, which
//knows the engine's state format and can re-frame the blob's stream
//header instead of throwing the bytes away (see reframe_state_blob).
//returns 0 if found, 1 if nothing usable, -1 on error.
//  in: session_id — session key to look up
//  in: model_hash — live model fingerprint ("" skips the check)
//  in: out_state_data — receives a malloc'd copy of the state blob (caller frees)
//  in: out_state_size — receives the blob size in bytes
//  in: out_n_tokens_meta — receives the token count recorded with the blob
//  in: out_n_stream — receives the n_stream the blob was written under (0 = unknown)
// out: 0 found; 1 nothing usable (missing row, model mismatch, blob read failure)
// side effects: logs a per-load sqlite/lmdb timing line to stderr
//------------------------------------------------------------
int qml_session_load_any(
    const char*  session_id,
    const char*  model_hash,
    void**       out_state_data,
    size_t*      out_state_size,
    int64_t*     out_n_tokens_meta,
    uint32_t*    out_n_stream)
{
    *out_state_data    = NULL;
    *out_state_size    = 0;
    *out_n_tokens_meta = 0;
    *out_n_stream      = 0;

    struct timespec ts0, ts1, ts2;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    qml_llama_session_meta_t meta;
    if (qml_sqlite_get_session(session_id, &meta) != 0)
    {
        return 1;
    }

    if (model_hash[0] != '\0' && strcmp(model_hash, meta.model_hash) != 0)
    {
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    void*  data = NULL;
    size_t size = 0;

    if (qml_lmdb_get_blob(meta.lmdb_key, &data, &size) != 0)
    {
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    /* one line per load: metadata lookup vs blob read, the two halves of a
     * fetch -- needed to profile the CPU-side restore floor */
    fprintf(stderr, "[qml][timing] load key='%s' size=%zu sqlite_ms=%.1f lmdb_ms=%.1f\n",
            meta.lmdb_key, size,
            (ts1.tv_sec - ts0.tv_sec) * 1e3 + (ts1.tv_nsec - ts0.tv_nsec) / 1e6,
            (ts2.tv_sec - ts1.tv_sec) * 1e3 + (ts2.tv_nsec - ts1.tv_nsec) / 1e6);

    *out_state_data    = data;
    *out_state_size    = size;
    *out_n_tokens_meta = meta.base.n_tokens;
    *out_n_stream      = (uint32_t) meta.n_stream;

    return 0;
}

//------------------------------------------------------------
//Func : qml_session_load_done
// Frees the buffers returned by the snapshot-load family of calls.
//  in: state_data — blob pointer to free (NULL-safe)
//  in: token_ids — token-id array to free (NULL-safe)
// out: none
//------------------------------------------------------------
void qml_session_load_done(void* state_data, int32_t* token_ids)
{
    free(state_data);
    free(token_ids);
}

//------------------------------------------------------------
//Func : qml_session_save
// Full-snapshot save: state blob, session metadata, and the complete token log as one unit.
//  in: session_id — session key (also used as the LMDB blob key)
//  in: user_id — owning user identifier
//  in: model_hash — fingerprint of the model the blob belongs to
//  in: seq_id — engine sequence id recorded for every token row
//  in: state_data — engine state blob (must be non-NULL)
//  in: state_size — blob size in bytes (must be > 0)
//  in: token_ids — the session's full token history, n_tokens entries
//  in: n_tokens — token count (0 clears any previously stored log)
// out: 0 on success; -1 on any storage failure (a failed token ARCHIVE does not fail the save)
// side effects: writes the LMDB blob, upserts the SQLite session/ext rows, replaces the
//               token log, and archives rows a context shift is about to discard
//------------------------------------------------------------
int qml_session_save(
    const char*    session_id,
    const char*    user_id,
    const char*    model_hash,
    int32_t        seq_id,
    const void*    state_data,
    size_t         state_size,
    const int32_t* token_ids,
    size_t         n_tokens)
{
    if (state_data == NULL || state_size == 0)
    {
        return -1;
    }

    if (qml_lmdb_put_blob(session_id, state_data, state_size) != 0)
    {
        // [BLOB-FALLBACK 2026-10-02] The token log is the record; the blob only
        // saves recomputation. Returning -1 here used to drop the whole save
        // without a log line, so a turn (or a q+ injection) vanished whenever
        // LMDB could not take the blob -- seen on the H200 with the 64 GiB map
        // 99.7% used (in one load run, 7 copy destinations lost their copy).
        // Save the session row and token log anyway, exactly like the capture
        // throttle's tokens-only save: the blob already stored (if any) stays
        // and restores as a PREFIX (qml_restore_covers), the tail recomputes;
        // with no blob the session replays from the token log.
        fprintf(stderr, "[qml][save] blob write failed session=%s size=%zu tokens=%zu"
                " -- saving the session row and token log without the new blob\n",
                session_id, state_size, n_tokens);
        return qml_session_save_tokens(session_id, user_id, model_hash,
                                       seq_id, token_ids, n_tokens);
    }

    qml_llama_session_meta_t meta;
    memset(&meta, 0, sizeof(meta));

    strncpy(meta.base.user_id,    user_id,    sizeof(meta.base.user_id) - 1);
    strncpy(meta.base.session_id, session_id, sizeof(meta.base.session_id) - 1);
    strncpy(meta.model_hash,      model_hash, sizeof(meta.model_hash) - 1);
    strncpy(meta.lmdb_key,        session_id, sizeof(meta.lmdb_key) - 1);

    meta.base.engine_type = QML_ENGINE_LLAMA_CPP;
    meta.base.seq_id      = seq_id;
    meta.state_size       = (int64_t) state_size;
    meta.n_stream         = (int64_t) g_n_stream;
    meta.n_ctx_seq        = (int64_t) g_n_ctx_seq;
    meta.base.n_tokens    = (int64_t) n_tokens;
    meta.base.last_token  = (n_tokens > 0) ? token_ids[n_tokens - 1] : -1;
    meta.base.last_pos    = (int32_t) n_tokens - 1;

    // Session row, archive of the rows a context shift discards (llama-server
    // rewrites its window as n_keep + tail, see qml_sqlite_archive_rewrite) and
    // the full token log are saved in ONE transaction. They used to be three,
    // and a failed token-log write left a session row whose n_tokens pointed at
    // an empty log (2026-10-02, H200). An archive failure still does not fail
    // the save. n_tokens == 0 clears any stored log, otherwise the next load
    // would pair an empty blob with stale rows.
    return (qml_sqlite_save_session(session_id, &meta, NULL, token_ids, n_tokens,
                                    seq_id, 1) == 0) ? 0 : -1;
}

//------------------------------------------------------------
//Func : qml_session_save_tokens
//capture-throttle save (2026-08-11): persist the token log and session row
//but leave the blob/ext untouched. The blob goes prefix-stale by fewer than
//the throttle delta; the restore path applies it as a prefix and recomputes
//only the tail. Data safety is identical -- the token log remains the source
//of truth.
//  in: session_id — session key
//  in: user_id — owning user identifier
//  in: model_hash — unused (kept for signature parity with the full save)
//  in: seq_id — engine sequence id recorded for every token row
//  in: token_ids — the session's full token history, n_tokens entries
//  in: n_tokens — token count (0 leaves the stored log untouched)
// out: 0 on success; -1 on any SQLite failure (archive failure alone does not fail the save)
// side effects: upserts the SQLite base row and replaces the token log; no LMDB write
//------------------------------------------------------------
int qml_session_save_tokens(
    const char*    session_id,
    const char*    user_id,
    const char*    model_hash,
    int32_t        seq_id,
    const int32_t* token_ids,
    size_t         n_tokens)
{
    (void) model_hash;

    qml_session_meta_t base;
    memset(&base, 0, sizeof(base));

    strncpy(base.user_id,    user_id,    sizeof(base.user_id) - 1);
    strncpy(base.session_id, session_id, sizeof(base.session_id) - 1);

    base.engine_type = QML_ENGINE_LLAMA_CPP;
    base.seq_id      = seq_id;
    base.n_tokens    = (int64_t) n_tokens;
    base.last_token  = (n_tokens > 0) ? token_ids[n_tokens - 1] : -1;
    base.last_pos    = (int32_t) n_tokens - 1;

    // one transaction, as in qml_session_save; n_tokens == 0 leaves the
    // stored log untouched (the previous behaviour of this path)
    return (qml_sqlite_save_session(session_id, NULL, &base, token_ids, n_tokens,
                                    seq_id, 0) == 0) ? 0 : -1;
}

//------------------------------------------------------------
//Func : qml_control_put
// Enqueues a control-plane command (thin wrapper over qml_sqlite_control_put).
//  in: target_engine — engine name the command is addressed to
//  in: target_pid — pid of the target engine process
//  in: cmd_type — command verb (e.g. RESTORE / PREWARM)
//  in: session_id — session the command refers to (NULL stored as "")
//  in: block_hashes — optional KV block hash list, n_block_hashes entries
//  in: n_block_hashes — number of block hashes
//  in: out_cmd_id — optional (may be NULL); receives the new command's rowid
// out: 0 on success; -1 on error
// side effects: inserts command + block rows into the SQLite control tables
//------------------------------------------------------------
int qml_control_put(
    const char*        target_engine,
    int32_t            target_pid,
    const char*        cmd_type,
    const char*        session_id,
    const char* const* block_hashes,
    size_t             n_block_hashes,
    int64_t*           out_cmd_id)
{
    return qml_sqlite_control_put(
        target_engine, target_pid, cmd_type, session_id, block_hashes, n_block_hashes, out_cmd_id);
}

//------------------------------------------------------------
//Func : qml_control_claim
// Atomically claims the oldest PENDING command addressed to this pid (SQLite wrapper).
//  in: target_pid — pid whose queue is polled
//  in: out_cmd — receives the claimed command's fields
// out: 0 claimed; 1 nothing pending; -1 on error
// side effects: marks the claimed row CLAIMED in the control table
//------------------------------------------------------------
int qml_control_claim(int32_t target_pid, qml_control_cmd_t* out_cmd)
{
    return qml_sqlite_control_claim(target_pid, out_cmd);
}

//------------------------------------------------------------
//Func : qml_control_set_status
// Updates a claimed command's status (e.g. DONE / FAILED); SQLite wrapper.
//  in: cmd_id — rowid returned when the command was inserted/claimed
//  in: status — new status string
// out: 0 on success; -1 on error
// side effects: updates the command row in the SQLite control table
//------------------------------------------------------------
int qml_control_set_status(int64_t cmd_id, const char* status)
{
    return qml_sqlite_control_set_status(cmd_id, status);
}

//------------------------------------------------------------
//Func : qml_control_blocks
// Fetches the KV block hash list attached to a command (SQLite wrapper).
//  in: cmd_id — command rowid
//  in: out_hashes — receives a malloc'd array of block-hash records (caller frees)
//  in: out_n — receives the number of records
// out: 0 on success (possibly 0 records); -1 on error
//------------------------------------------------------------
int qml_control_blocks(int64_t cmd_id, qml_control_cmd_block_t** out_hashes, size_t* out_n)
{
    return qml_sqlite_control_blocks(cmd_id, out_hashes, out_n);
}

//------------------------------------------------------------
//Func : qml_store_compact
// Compacts the SQLite store (WAL checkpoint + VACUUM); wrapper over qml_sqlite_compact.
//  in: none
// out: 0 on success; -1 on error
// side effects: truncates the WAL and rewrites the database file
//------------------------------------------------------------
int qml_store_compact(void)
{
    return qml_sqlite_compact();
}

//------------------------------------------------------------
//Func : qml_store_drop_kv
// Drops the KV blocks only this session stored: records first (one SQLite
// transaction), then the LMDB values (one LMDB transaction). A block whose
// record is gone is never matched again, so a crash between the two steps
// leaves unreferenced values for SWEEP_BLOBS, never a record without data.
//  in: owner — session id
// out: LMDB values deleted (>= 0); -1 on error
//------------------------------------------------------------
long qml_store_drop_kv(const char * owner)
{
    char ** keys = NULL;
    size_t  n    = 0;
    if (qml_sqlite_owner_take(owner, &keys, &n) != 0)
    {
        return -1;
    }
    long deleted = 0;
    if (n > 0)
    {
        deleted = qml_lmdb_delete_many((const char * const *) keys, n);
    }
    qml_sqlite_free_keys(keys, n);
    return deleted;
}

//------------------------------------------------------------
//Func : qml_session_lock
// Takes the cross-process ownership lock on a session (SQLite wrapper; steals from dead pids).
//  in: session_id — session to lock
//  in: pid — pid requesting ownership
// out: 0 acquired (or already ours); 1 held by a different live pid; -1 on error
// side effects: updates owner_pid/lock_state on the session row
//------------------------------------------------------------
int qml_session_lock(const char* session_id, int32_t pid)
{
    return qml_sqlite_lock(session_id, pid);
}

//------------------------------------------------------------
//Func : qml_session_unlock
// Releases the session lock if this pid holds it (SQLite wrapper; no-op otherwise).
//  in: session_id — session to unlock
//  in: pid — pid that believes it owns the lock
// out: 0 on success (including no-op when not the owner); -1 on error
// side effects: clears lock_state on the session row when owned by pid
//------------------------------------------------------------
int qml_session_unlock(const char* session_id, int32_t pid)
{
    return qml_sqlite_unlock(session_id, pid);
}

//------------------------------------------------------------
//Func : qml_session_lock_state
// Reads a session's current lock state without changing it (SQLite wrapper).
//  in: session_id — session to inspect
//  in: out_owner_pid — receives the recorded owner pid
//  in: out_locked — receives 1 if lock_state is LOCKED, else 0
// out: 0 found; 1 no such session; -1 on error
//------------------------------------------------------------
int qml_session_lock_state(const char* session_id, int32_t* out_owner_pid, int* out_locked)
{
    return qml_sqlite_lock_state(session_id, out_owner_pid, out_locked);
}

//------------------------------------------------------------
//Func : qml_session_last_pos
// Reads the last stored token position of a session (SQLite wrapper).
//  in: session_id — session to inspect
//  in: out_last_pos — receives the stored last_pos
// out: 0 found; 1 no such session; -1 on error
//------------------------------------------------------------
int qml_session_last_pos(const char* session_id, int32_t* out_last_pos)
{
    return qml_sqlite_last_pos(session_id, out_last_pos);
}

//------------------------------------------------------------
//Func : qml_session_append
// Appends tokens to a session's log, continuing positions after the stored last_pos.
//  in: session_id — session whose log grows
//  in: token_ids — token ids to append, n_tokens entries
//  in: n_tokens — number of tokens (0 is a successful no-op)
//  in: seq_id — sequence id recorded for every appended row
// out: 0 on success; -1 on error
// side effects: inserts rows into the SQLite token log
//------------------------------------------------------------
int qml_session_append(const char* session_id, const int32_t* token_ids, size_t n_tokens, int32_t seq_id)
{
    return qml_sqlite_append_tokens(session_id, token_ids, n_tokens, seq_id);
}
