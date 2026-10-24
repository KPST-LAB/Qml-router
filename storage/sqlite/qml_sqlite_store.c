// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
// storage/sqlite/qml_sqlite_store.c

#include "qml_sqlite_store.h"
#include "qml_sqlite_schema.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static sqlite3         * g_db   = NULL;
static pthread_mutex_t   g_lock = PTHREAD_MUTEX_INITIALIZER;

// [BUSY-TIMEOUT 2026-10-02] How long a write waits for another connection's
// lock before SQLite returns SQLITE_BUSY. WAL mode still allows one writer at a
// time: the gateway, a second engine and the q+ copy path write the same
// qml.db, and with no busy timeout a BEGIN IMMEDIATE that met their lock failed
// at once (H200 service: 4 of 7 token-log saves lost). With 5000 ms the same
// service stored 10 of 10 saves. A constant, not an env switch.
#define QML_SQLITE_BUSY_TIMEOUT_MS 5000

//------------------------------------------------------------
//Func : store_log_fail
// Writes one line about a failed SQLite step to stderr (the engine log).
//  in: where — function or step name
//  in: session_id — session concerned (NULL allowed)
// out: none
// side effects: one "[qml][sqlite] ..." line; call it BEFORE ROLLBACK, which
//               replaces the connection's error message
//------------------------------------------------------------
static void store_log_fail(const char * where, const char * session_id)
{
    const int rc = (g_db != NULL) ? sqlite3_extended_errcode(g_db) : SQLITE_ERROR;
    fprintf(stderr, "[qml][sqlite] %s failed session=%s rc=%d (%s) errmsg=\"%s\"\n",
            where, session_id ? session_id : "-", rc, sqlite3_errstr(rc),
            (g_db != NULL) ? sqlite3_errmsg(g_db) : "no connection");
    fflush(stderr);
}

//------------------------------------------------------------
//Func : commit_or_rollback
// COMMITs the open transaction; when COMMIT fails, logs it and rolls back.
// Why: a failed COMMIT (for example SQLITE_BUSY) leaves the transaction OPEN.
// Returning -1 without ROLLBACK kept the connection inside that transaction,
// and every later BEGIN on it failed as well.
//  in: where — caller name for the log line
//  in: session_id — session concerned (NULL allowed)
// out: 0 committed; -1 rolled back
// side effects: caller must hold g_lock with a transaction open
//------------------------------------------------------------
static int commit_or_rollback(const char * where, const char * session_id)
{
    if (sqlite3_exec(g_db, "COMMIT;", NULL, NULL, NULL) == SQLITE_OK)
    {
        return 0;
    }
    store_log_fail(where, session_id);
    sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
    return -1;
}

//------------------------------------------------------------
//Func : qml_sqlite_open
// Opens the process-wide SQLite handle and creates the full schema (idempotent).
//  in: db_path — filesystem path of the database file
// out: 0 on success (also when already open); -1 on open or DDL failure
// side effects: sets the global handle, enables WAL, creates/updates all tables and indexes
//------------------------------------------------------------
int qml_sqlite_open(const char * db_path)
{
    pthread_mutex_lock(&g_lock);

    if (g_db != NULL)
    {
        pthread_mutex_unlock(&g_lock);
        return 0;
    }

    int rc = sqlite3_open(db_path, &g_db);
    if (rc != SQLITE_OK)
    {
        fprintf(stderr, "[qml][sqlite] open %s failed rc=%d (%s)\n",
                db_path, rc, sqlite3_errstr(rc));
        sqlite3_close(g_db);   // sqlite3_open returns a handle even on failure
        g_db = NULL;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_busy_timeout(g_db, QML_SQLITE_BUSY_TIMEOUT_MS);

    sqlite3_exec(g_db, "PRAGMA journal_mode=WAL;",   NULL, NULL, NULL);
    sqlite3_exec(g_db, "PRAGMA synchronous=NORMAL;", NULL, NULL, NULL);
    // NOTE: cache_size / mmap_size / temp_store tuning was tried here and
    // REJECTED after an A/B measurement (2026-07-30). Workload: 24 sessions x
    // 8000 tokens = 192k rows, 3 read passes.
    //   with tuning : save median 23.60ms / total 685.8ms, read median 3.03ms / total 245.8ms
    //   without     : save median 24.83ms / total 668.7ms, read median 3.13ms / total 229.4ms
    // Medians favoured the tuning while totals favoured its absence -- i.e.
    // noise, not an effect. The token log is only a few MB, so it already sits
    // in the OS page cache and a bigger SQLite page cache has nothing to do.
    // Left out deliberately: an unmeasured "optimisation" is exactly what this
    // project rejects. Revisit only with a workload where it is measurable.

    char * err = NULL;

    rc = sqlite3_exec(g_db, QML_DDL_CREATE_SESSION, NULL, NULL, &err);
    // [TRIM-HEAD-STICKY 2026-09-01] Per-session storage column for the head (system prefix)
    // exemption decision. It must be created when the DB is opened -- at first it was placed in the vec lazy
    // initialization block, so trims that ran before the first search ran without the column, fell back to
    // the old behavior (re-decide on every trim) and locked the first session's rule turn into the head
    // (measured: finance trim_head_len=121 = length of rule 1). -1 = not yet decided.
    sqlite3_exec(g_db, "ALTER TABLE qml_session ADD COLUMN trim_head_len"
                 " INTEGER NOT NULL DEFAULT -1",
                 NULL, NULL, NULL);   // ignore the error if it already exists (convention)

    if (rc != SQLITE_OK)
    {
        sqlite3_free(err);
        sqlite3_close(g_db);
        g_db = NULL;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    rc = sqlite3_exec(g_db, QML_DDL_CREATE_LLAMA_EXT, NULL, NULL, &err);
    if (rc != SQLITE_OK)
    {
        sqlite3_free(err);
        sqlite3_close(g_db);
        g_db = NULL;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    rc = sqlite3_exec(g_db, QML_DDL_CREATE_VLLM_EXT, NULL, NULL, &err);
    if (rc != SQLITE_OK)
    {
        sqlite3_free(err);
        sqlite3_close(g_db);
        g_db = NULL;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    rc = sqlite3_exec(g_db, QML_DDL_CREATE_TOKEN_LOG, NULL, NULL, &err);
    if (rc != SQLITE_OK)
    {
        sqlite3_free(err);
        sqlite3_close(g_db);
        g_db = NULL;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    rc = sqlite3_exec(g_db, QML_DDL_DROP_TOKEN_LOG_IDX, NULL, NULL, &err);
    if (rc != SQLITE_OK)
    {
        sqlite3_free(err);
        sqlite3_close(g_db);
        g_db = NULL;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    static const char * const control_cmd_ddl[] = {
        QML_DDL_CREATE_CONTROL_CMD,
        QML_DDL_CREATE_CONTROL_CMD_IDX,
        QML_DDL_CREATE_CONTROL_CMD_BLOCKS,
        QML_DDL_CREATE_CONTROL_CMD_BLOCKS_IDX,
        QML_DDL_CREATE_KV_BLOCK,
        QML_DDL_CREATE_KV_BLOCK_IDX,
        QML_DDL_CREATE_KV_OWNER,
        QML_DDL_CREATE_KV_OWNER_IDX,
        QML_DDL_CREATE_TOKEN_ARCHIVE,
    };
    for (size_t i = 0; i < sizeof(control_cmd_ddl) / sizeof(control_cmd_ddl[0]); ++i)
    {
        rc = sqlite3_exec(g_db, control_cmd_ddl[i], NULL, NULL, &err);
        if (rc != SQLITE_OK)
        {
            sqlite3_free(err);
            sqlite3_close(g_db);
            g_db = NULL;
            pthread_mutex_unlock(&g_lock);
            return -1;
        }
    }

    pthread_mutex_unlock(&g_lock);
    return 0;
}

//------------------------------------------------------------
//Func : qml_sqlite_close
// Closes the process-wide SQLite handle (safe to call when already closed).
//  in: none
// out: none
// side effects: closes and clears the global handle
//------------------------------------------------------------
void qml_sqlite_close(void)
{
    pthread_mutex_lock(&g_lock);

    if (g_db != NULL)
    {
        sqlite3_close(g_db);
        g_db = NULL;
    }

    pthread_mutex_unlock(&g_lock);
}

//------------------------------------------------------------
//Func : upsert_session_base (internal)
//writes only the qml_session base row. caller must already hold g_lock
//and have an open transaction. returns the raw sqlite3_step result
//(SQLITE_DONE on success) so the caller decides whether to roll back.
//shared by qml_sqlite_put_session (llama) and
//qml_sqlite_put_vllm_session so the base-row logic exists exactly once.
//  in: session_id — session key of the row to upsert
//  in: base — engine-agnostic session fields to write
// out: SQLITE_DONE on success; any other sqlite3 result code on failure
//------------------------------------------------------------
static int upsert_session_base(const char * session_id, const qml_session_meta_t * base)
{
    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_UPSERT_SESSION, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        return rc;
    }

    const int64_t now = (int64_t) time(NULL);

    sqlite3_bind_text (stmt, 1, session_id,        -1, SQLITE_TRANSIENT);
    sqlite3_bind_text (stmt, 2, base->user_id,      -1, SQLITE_TRANSIENT);
    sqlite3_bind_int  (stmt, 3, base->engine_type);
    sqlite3_bind_int  (stmt, 4, base->seq_id);
    sqlite3_bind_int  (stmt, 5, base->last_token);
    sqlite3_bind_int  (stmt, 6, base->last_pos);
    sqlite3_bind_int64(stmt, 7, base->n_tokens);
    sqlite3_bind_int64(stmt, 8, now);
    sqlite3_bind_int64(stmt, 9, now);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return rc;
}


//------------------------------------------------------------
//Func : qml_sqlite_copy_session
// Duplicates a session's base row and token log under a new id, in one transaction.
//  in: src_session_id — existing session to copy (must exist)
//  in: dst_session_id — new session id (must not exist yet)
//  in: new_user_id — owner for the copy; NULL or "" keeps the source's user_id
// out: 0 copied; 1 precondition failed (src missing or dst already exists); -1 on error
// side effects: inserts qml_session and qml_token_log rows; ext rows/blobs are NOT copied
//------------------------------------------------------------
int qml_sqlite_copy_session(const char * src_session_id, const char * dst_session_id,
                            const char * new_user_id)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    char * err = NULL;
    if (sqlite3_exec(g_db, "BEGIN IMMEDIATE;", NULL, NULL, &err) != SQLITE_OK)
    {
        sqlite3_free(err);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    int result = -1;
    sqlite3_stmt * stmt = NULL;

    // dst must not exist / src must exist -- both checked inside the txn
    if (sqlite3_prepare_v2(g_db,
            "SELECT (SELECT COUNT(*) FROM qml_session WHERE session_id = ?1), "
            "       (SELECT COUNT(*) FROM qml_session WHERE session_id = ?2);",
            -1, &stmt, NULL) == SQLITE_OK)
    {
        sqlite3_bind_text(stmt, 1, src_session_id, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, dst_session_id, -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW)
        {
            result = (sqlite3_column_int(stmt, 0) == 1 &&
                      sqlite3_column_int(stmt, 1) == 0) ? 0 : 1;
        }
        sqlite3_finalize(stmt);
        stmt = NULL;
    }

    if (result == 0 && sqlite3_prepare_v2(g_db,
            "INSERT INTO qml_session (session_id, user_id, engine_type, seq_id, "
            "  last_token, last_pos, n_tokens, owner_pid, lock_state, created_at, updated_at) "
            "SELECT ?2, CASE WHEN ?3 = '' THEN user_id ELSE ?3 END, engine_type, seq_id, "
            "  last_token, last_pos, n_tokens, "
            "  0, 'FREE', strftime('%s','now'), strftime('%s','now') "
            "FROM qml_session WHERE session_id = ?1;",
            -1, &stmt, NULL) == SQLITE_OK)
    {
        sqlite3_bind_text(stmt, 1, src_session_id, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, dst_session_id, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, (new_user_id != NULL) ? new_user_id : "", -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) != SQLITE_DONE)
        {
            result = -1;
        }
        sqlite3_finalize(stmt);
        stmt = NULL;
    }
    else if (result == 0)
    {
        result = -1;
    }

    if (result == 0 && sqlite3_prepare_v2(g_db,
            "INSERT INTO qml_token_log (session_id, pos, seq_id, token_id) "
            "SELECT ?2, pos, seq_id, token_id FROM qml_token_log WHERE session_id = ?1;",
            -1, &stmt, NULL) == SQLITE_OK)
    {
        sqlite3_bind_text(stmt, 1, src_session_id, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, dst_session_id, -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) != SQLITE_DONE)
        {
            result = -1;
        }
        sqlite3_finalize(stmt);
        stmt = NULL;
    }
    else if (result == 0)
    {
        result = -1;
    }

    if (result == 0)
    {
        if (commit_or_rollback(__func__, dst_session_id) != 0)
        {
            result = -1;
        }
    }
    else
    {
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
    }
    pthread_mutex_unlock(&g_lock);
    return result;
}

//------------------------------------------------------------
//Func : qml_sqlite_delete_llama_ext
// Deletes only the llama extension row, leaving the base session and token log intact.
//  in: session_id — session whose qml_llama_session_ext row is removed
// out: 0 on success (also when no row existed); -1 on error
//------------------------------------------------------------
int qml_sqlite_delete_llama_ext(const char * session_id)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_DELETE_LLAMA_EXT, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);
    return (rc == SQLITE_DONE) ? 0 : -1;
}

//------------------------------------------------------------
//Func : upsert_llama_ext
// Writes the qml_llama_session_ext row; no transaction control, no locking.
//  in: session_id — session key
//  in: meta — llama-specific fields (model_hash, lmdb_key, sizes, geometry)
// out: SQLITE_DONE on success; any other sqlite3 result code on failure
//------------------------------------------------------------
static int upsert_llama_ext(const char * session_id, const qml_llama_session_meta_t * meta)
{
    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_UPSERT_LLAMA_EXT, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        return rc;
    }

    sqlite3_bind_text (stmt, 1, session_id,        -1, SQLITE_TRANSIENT);
    sqlite3_bind_text (stmt, 2, meta->model_hash,   -1, SQLITE_TRANSIENT);
    sqlite3_bind_text (stmt, 3, meta->lmdb_key,     -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 4, meta->state_size);
    sqlite3_bind_int64(stmt, 5, meta->n_stream);
    sqlite3_bind_int64(stmt, 6, meta->n_ctx_seq);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return rc;
}

//------------------------------------------------------------
//Func : qml_sqlite_uses_blob
// Tells whether any session still refers to an LMDB blob key.
//  in: key — LMDB blob key (for llama sessions this is the session id)
// out: 1 when a qml_session row has that id or a qml_llama_session_ext row has
//      that lmdb_key; 0 when neither; 1 on any error (callers delete only on 0)
//------------------------------------------------------------
int qml_sqlite_uses_blob(const char * key)
{
    if (g_db == NULL || key == NULL)
    {
        return 1;
    }
    pthread_mutex_lock(&g_lock);
    sqlite3_stmt * st = NULL;
    int            ref = 1;
    if (sqlite3_prepare_v2(g_db,
            "SELECT EXISTS(SELECT 1 FROM qml_session WHERE session_id = ?1)"
            "    OR EXISTS(SELECT 1 FROM qml_llama_session_ext WHERE lmdb_key = ?1);",
            -1, &st, NULL) == SQLITE_OK)
    {
        sqlite3_bind_text(st, 1, key, -1, SQLITE_TRANSIENT);
        if (sqlite3_step(st) == SQLITE_ROW)
        {
            ref = sqlite3_column_int(st, 0) ? 1 : 0;
        }
        sqlite3_finalize(st);
    }
    pthread_mutex_unlock(&g_lock);
    return ref;
}

//------------------------------------------------------------
//Func : qml_sqlite_put_session
//------------------------------------------------------------
//Updates only the session base row -- does not touch ext (blob record). Used when a capture-saving
//save (tokens-only) must keep the previous blob metadata.
//  in: session_id — session key of the row to upsert
//  in: base — engine-agnostic session fields to write
// out: 0 on success; -1 on error (extension rows are left untouched either way)
int qml_sqlite_put_session_base(const char * session_id, const qml_session_meta_t * base)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    const int rc = upsert_session_base(session_id, base);

    pthread_mutex_unlock(&g_lock);
    return (rc == SQLITE_DONE) ? 0 : -1;
}

// Upserts a llama session: base row plus qml_llama_session_ext row in one transaction.
//  in: session_id — session key of the row to upsert
//  in: meta — base fields plus llama-specific fields (model_hash, lmdb_key, sizes, geometry)
// out: 0 on success; -1 on error (transaction rolled back, nothing partially written)
int qml_sqlite_put_session(const char * session_id, const qml_llama_session_meta_t * meta)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    char * err = NULL;
    int rc = sqlite3_exec(g_db, "BEGIN IMMEDIATE;", NULL, NULL, &err);
    if (rc != SQLITE_OK)
    {
        sqlite3_free(err);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    rc = upsert_session_base(session_id, &meta->base);
    if (rc != SQLITE_DONE)
    {
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    // extension row (qml_llama_session_ext): llama-specific fields only
    rc = upsert_llama_ext(session_id, meta);
    if (rc != SQLITE_DONE)
    {
        store_log_fail(__func__, session_id);
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    if (commit_or_rollback(__func__, NULL) != 0)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    pthread_mutex_unlock(&g_lock);

    return 0;
}

//------------------------------------------------------------
//Func : qml_sqlite_put_vllm_session
//mirrors qml_sqlite_put_session, writing qml_vllm_session_ext instead
//of qml_llama_session_ext as the extension row.
//  in: session_id — session key of the row to upsert
//  in: meta — base fields plus vLLM-specific fields (model_hash, lmdb_prefix, block layout)
// out: 0 on success; -1 on error (transaction rolled back)
//------------------------------------------------------------
int qml_sqlite_put_vllm_session(const char * session_id, const qml_vllm_session_meta_t * meta)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    char * err = NULL;
    int rc = sqlite3_exec(g_db, "BEGIN IMMEDIATE;", NULL, NULL, &err);
    if (rc != SQLITE_OK)
    {
        sqlite3_free(err);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    rc = upsert_session_base(session_id, &meta->base);
    if (rc != SQLITE_DONE)
    {
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_stmt * stmt = NULL;
    rc = sqlite3_prepare_v2(g_db, QML_SQL_UPSERT_VLLM_EXT, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_text (stmt, 1, session_id,          -1, SQLITE_TRANSIENT);
    sqlite3_bind_text (stmt, 2, meta->model_hash,     -1, SQLITE_TRANSIENT);
    sqlite3_bind_text (stmt, 3, meta->lmdb_prefix,    -1, SQLITE_TRANSIENT);
    sqlite3_bind_int  (stmt, 4, meta->block_size);
    sqlite3_bind_int  (stmt, 5, meta->num_blocks);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE)
    {
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    if (commit_or_rollback(__func__, NULL) != 0)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    pthread_mutex_unlock(&g_lock);

    return 0;
}

//------------------------------------------------------------
//Func : qml_sqlite_get_session
// Loads a llama session's metadata (base row joined with its llama ext row).
//  in: session_id — session key to look up
//  in: out_meta — receives the full metadata; zeroed then filled on hit
// out: 0 found; 1 not found; -1 on error
//------------------------------------------------------------
int qml_sqlite_get_session(const char * session_id, qml_llama_session_meta_t * out_meta)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_SELECT_LLAMA_SESSION, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);

    if (rc == SQLITE_ROW)
    {
        memset(out_meta, 0, sizeof(*out_meta));

        // column order per QML_SQL_SELECT_LLAMA_SESSION: user_id, engine_type,
        // seq_id, last_token, last_pos, n_tokens, model_hash, lmdb_key, state_size,
        // n_stream, n_ctx_seq
        const unsigned char * user_id    = sqlite3_column_text(stmt, 0);
        const unsigned char * model_hash = sqlite3_column_text(stmt, 6);
        const unsigned char * lmdb_key   = sqlite3_column_text(stmt, 7);

        strncpy(out_meta->base.user_id,    (const char *) user_id,    sizeof(out_meta->base.user_id) - 1);
        strncpy(out_meta->base.session_id, session_id,                sizeof(out_meta->base.session_id) - 1);
        strncpy(out_meta->model_hash,      (const char *) model_hash, sizeof(out_meta->model_hash) - 1);
        strncpy(out_meta->lmdb_key,        (const char *) lmdb_key,   sizeof(out_meta->lmdb_key) - 1);

        out_meta->base.engine_type = sqlite3_column_int  (stmt, 1);
        out_meta->base.seq_id      = sqlite3_column_int  (stmt, 2);
        out_meta->base.last_token  = sqlite3_column_int  (stmt, 3);
        out_meta->base.last_pos    = sqlite3_column_int  (stmt, 4);
        out_meta->base.n_tokens    = sqlite3_column_int64(stmt, 5);
        out_meta->state_size       = sqlite3_column_int64(stmt, 8);
        out_meta->n_stream         = sqlite3_column_int64(stmt, 9);
        out_meta->n_ctx_seq        = sqlite3_column_int64(stmt, 10);

        sqlite3_finalize(stmt);
        pthread_mutex_unlock(&g_lock);
        return 0;
    }

    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);

    return (rc == SQLITE_DONE) ? 1 : -1;
}

//------------------------------------------------------------
//Func : qml_sqlite_recent
// Lists the most recently updated session ids for one engine type.
//  in: engine_type — QML_ENGINE_* value to filter on
//  in: limit — maximum number of ids to return (must be > 0)
//  in: out_ids — receives a malloc'd array of strdup'd ids
//  in: out_n — receives the number of ids returned
// out: 0 on success (possibly 0 ids); -1 on error.
//      Caller frees the result with qml_sqlite_recent_free.
//------------------------------------------------------------
int qml_sqlite_recent(int engine_type, size_t limit, char *** out_ids, size_t * out_n)
{
    if (g_db == NULL || out_ids == NULL || out_n == NULL || limit == 0)
    {
        return -1;
    }

    *out_ids = NULL;
    *out_n   = 0;

    pthread_mutex_lock(&g_lock);

    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db,
        "SELECT session_id FROM qml_session WHERE engine_type = ? "
        "ORDER BY updated_at DESC LIMIT ?;", -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_int  (stmt, 1, engine_type);
    sqlite3_bind_int64(stmt, 2, (sqlite3_int64) limit);

    char ** ids = (char **) calloc(limit, sizeof(char *));
    size_t  n   = 0;

    while (n < limit && sqlite3_step(stmt) == SQLITE_ROW)
    {
        const unsigned char * sid = sqlite3_column_text(stmt, 0);
        if (sid != NULL)
        {
            ids[n++] = strdup((const char *) sid);
        }
    }

    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);

    *out_ids = ids;
    *out_n   = n;

    return 0;
}

//------------------------------------------------------------
//Func : qml_sqlite_recent_free
// Frees the id array returned by qml_sqlite_recent.
//  in: ids — array to free (NULL-safe)
//  in: n — number of entries in the array
// out: none
//------------------------------------------------------------
void qml_sqlite_recent_free(char ** ids, size_t n)
{
    if (ids == NULL)
    {
        return;
    }

    for (size_t i = 0; i < n; ++i)
    {
        free(ids[i]);
    }

    free(ids);
}

//------------------------------------------------------------
//Func : qml_sqlite_get_vllm_session
// Loads a vLLM session's metadata (base row joined with its vLLM ext row).
//  in: session_id — session key to look up
//  in: out_meta — receives the full metadata; zeroed then filled on hit
// out: 0 found; 1 not found; -1 on error
//------------------------------------------------------------
int qml_sqlite_get_vllm_session(const char * session_id, qml_vllm_session_meta_t * out_meta)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_SELECT_VLLM_SESSION, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);

    if (rc == SQLITE_ROW)
    {
        memset(out_meta, 0, sizeof(*out_meta));

        // column order per QML_SQL_SELECT_VLLM_SESSION: user_id, engine_type,
        // seq_id, last_token, last_pos, n_tokens, model_hash, lmdb_prefix,
        // block_size, num_blocks
        const unsigned char * user_id     = sqlite3_column_text(stmt, 0);
        const unsigned char * model_hash  = sqlite3_column_text(stmt, 6);
        const unsigned char * lmdb_prefix = sqlite3_column_text(stmt, 7);

        strncpy(out_meta->base.user_id,    (const char *) user_id,     sizeof(out_meta->base.user_id) - 1);
        strncpy(out_meta->base.session_id, session_id,                 sizeof(out_meta->base.session_id) - 1);
        strncpy(out_meta->model_hash,      (const char *) model_hash,  sizeof(out_meta->model_hash) - 1);
        strncpy(out_meta->lmdb_prefix,     (const char *) lmdb_prefix, sizeof(out_meta->lmdb_prefix) - 1);

        out_meta->base.engine_type = sqlite3_column_int(stmt, 1);
        out_meta->base.seq_id      = sqlite3_column_int(stmt, 2);
        out_meta->base.last_token  = sqlite3_column_int(stmt, 3);
        out_meta->base.last_pos    = sqlite3_column_int(stmt, 4);
        out_meta->base.n_tokens    = sqlite3_column_int64(stmt, 5);
        out_meta->block_size       = sqlite3_column_int(stmt, 8);
        out_meta->num_blocks       = sqlite3_column_int(stmt, 9);

        sqlite3_finalize(stmt);
        pthread_mutex_unlock(&g_lock);
        return 0;
    }

    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);

    return (rc == SQLITE_DONE) ? 1 : -1;
}

//------------------------------------------------------------
//Func : qml_sqlite_delete_session
// Removes every SQLite trace of a session: base row, both ext rows, and the token log.
//  in: session_id — session to delete
// out: 0 on success (also when nothing existed); -1 on error.
//      NOTE: the LMDB blob is not touched here -- the caller deletes it separately.
//------------------------------------------------------------
int qml_sqlite_delete_session(const char * session_id)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_DELETE_SESSION, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    rc = sqlite3_prepare_v2(g_db, QML_SQL_DELETE_LLAMA_EXT, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    rc = sqlite3_prepare_v2(g_db, QML_SQL_DELETE_VLLM_EXT, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    rc = sqlite3_prepare_v2(g_db, QML_SQL_DELETE_TOKENS, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    pthread_mutex_unlock(&g_lock);

    return (rc == SQLITE_DONE) ? 0 : -1;
}

//------------------------------------------------------------
//Func : qml_sqlite_block_touch
// Upserts one KV block's recency row (size + last-used timestamp) for LRU accounting.
//  in: block_key — cache key of the KV block (must be non-NULL)
//  in: n_bytes — stored size of the block in bytes
// out: 0 on success; -1 on error
// side effects: writes the qml_kv_block row with the current wall-clock time in µs
//------------------------------------------------------------
int qml_sqlite_block_touch(const char * block_key, int64_t n_bytes)
{
    if (g_db == NULL || block_key == NULL)
    {
        return -1;
    }

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    const int64_t now_us = (int64_t) ts.tv_sec * 1000000 + ts.tv_nsec / 1000;

    pthread_mutex_lock(&g_lock);

    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_KV_BLOCK_TOUCH, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_text (stmt, 1, block_key, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, n_bytes);
    sqlite3_bind_int64(stmt, 3, now_us);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);

    return (rc == SQLITE_DONE) ? 0 : -1;
}

//------------------------------------------------------------
//Func : qml_sqlite_block_touch_many
// Batch version of qml_sqlite_block_touch: touches many keys inside one transaction.
//  in: block_keys — array of cache keys (NULL entries are skipped)
//  in: n_keys — number of keys (0 is a successful no-op)
//  in: n_bytes — per-block size recorded for every key in the batch
// out: 0 on success; -1 on error (whole batch rolled back)
// side effects: upserts qml_kv_block rows sharing one timestamp
//------------------------------------------------------------
int qml_sqlite_block_touch_many(const char * const * block_keys, size_t n_keys, int64_t n_bytes)
{
    if (g_db == NULL || block_keys == NULL)
    {
        return -1;
    }

    if (n_keys == 0)
    {
        return 0;
    }

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    const int64_t now_us = (int64_t) ts.tv_sec * 1000000 + ts.tv_nsec / 1000;

    pthread_mutex_lock(&g_lock);

    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_KV_BLOCK_TOUCH, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    //one explicit transaction for the whole batch: without it SQLite
    //commits per statement, which is what made recency bookkeeping cost
    //more than the KV read it was accounting for
    sqlite3_exec(g_db, "BEGIN", NULL, NULL, NULL);

    int failed = 0;

    for (size_t i = 0; i < n_keys; ++i)
    {
        if (block_keys[i] == NULL)
        {
            continue;
        }

        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
        sqlite3_bind_text (stmt, 1, block_keys[i], -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 2, n_bytes);
        sqlite3_bind_int64(stmt, 3, now_us);

        if (sqlite3_step(stmt) != SQLITE_DONE)
        {
            failed = 1;
            break;
        }
    }

    sqlite3_finalize(stmt);
    if (failed)
    {
        sqlite3_exec(g_db, "ROLLBACK", NULL, NULL, NULL);
    }
    else if (commit_or_rollback(__func__, NULL) != 0)
    {
        failed = 1;
    }
    pthread_mutex_unlock(&g_lock);

    return failed ? -1 : 0;
}

//------------------------------------------------------------
//Func : qml_sqlite_owner_add
// Records that owner stored these blocks (one transaction).
//  in: owner — session id that stored the blocks
//  in: keys / n — block keys as written to LMDB
// out: 0 on success; -1 on error
//------------------------------------------------------------
int qml_sqlite_owner_add(const char * owner, const char * const * keys, size_t n)
{
    if (g_db == NULL || owner == NULL || owner[0] == '\0' || keys == NULL)
    {
        return -1;
    }
    if (n == 0)
    {
        return 0;
    }
    pthread_mutex_lock(&g_lock);
    sqlite3_stmt * stmt = NULL;
    if (sqlite3_prepare_v2(g_db, QML_SQL_KV_OWNER_ADD, -1, &stmt, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    if (sqlite3_exec(g_db, "BEGIN IMMEDIATE;", NULL, NULL, NULL) != SQLITE_OK)
    {
        store_log_fail("owner_add BEGIN", owner);
        sqlite3_finalize(stmt);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    int failed = 0;
    for (size_t i = 0; i < n && !failed; ++i)
    {
        if (keys[i] == NULL)
        {
            continue;
        }
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
        sqlite3_bind_text(stmt, 1, owner, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, keys[i], -1, SQLITE_TRANSIENT);
        failed = (sqlite3_step(stmt) != SQLITE_DONE);
    }
    sqlite3_finalize(stmt);
    if (failed)
    {
        store_log_fail("owner_add", owner);
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
    }
    else if (commit_or_rollback(__func__, owner) != 0)
    {
        failed = 1;
    }
    pthread_mutex_unlock(&g_lock);
    return failed ? -1 : 0;
}

//------------------------------------------------------------
//Func : qml_sqlite_owner_take
// Removes owner's block records and returns the blocks no other owner
// recorded, so the caller can delete their LMDB values. The accounting rows
// (qml_kv_block) of those blocks go in the same transaction.
//  in: owner — session id
// out: 0 on success with *out_keys (malloc'd array of malloc'd strings,
//      free with qml_sqlite_free_keys) and *out_n set; -1 on error
//------------------------------------------------------------
int qml_sqlite_owner_take(const char * owner, char *** out_keys, size_t * out_n)
{
    if (g_db == NULL || owner == NULL || out_keys == NULL || out_n == NULL)
    {
        return -1;
    }
    *out_keys = NULL;
    *out_n    = 0;
    pthread_mutex_lock(&g_lock);
    if (sqlite3_exec(g_db, "BEGIN IMMEDIATE;", NULL, NULL, NULL) != SQLITE_OK)
    {
        store_log_fail("owner_take BEGIN", owner);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    char ** keys = NULL;
    size_t  n = 0, cap = 0;
    int     failed = 0;
    sqlite3_stmt * sel = NULL;
    sqlite3_stmt * del = NULL;
    if (sqlite3_prepare_v2(g_db, QML_SQL_KV_OWNER_KEYS, -1, &sel, NULL) != SQLITE_OK ||
        sqlite3_prepare_v2(g_db, QML_SQL_KV_BLOCK_FORGET, -1, &del, NULL) != SQLITE_OK)
    {
        failed = 1;
    }
    if (!failed)
    {
        sqlite3_bind_text(sel, 1, owner, -1, SQLITE_TRANSIENT);
        int rc;
        while ((rc = sqlite3_step(sel)) == SQLITE_ROW)
        {
            const char * k = (const char *) sqlite3_column_text(sel, 0);
            if (k == NULL)
            {
                continue;
            }
            if (n == cap)
            {
                cap = cap ? cap * 2 : 256;
                char ** grown = (char **) realloc(keys, cap * sizeof(char *));
                if (grown == NULL)
                {
                    failed = 1;
                    break;
                }
                keys = grown;
            }
            keys[n] = strdup(k);
            if (keys[n] == NULL)
            {
                failed = 1;
                break;
            }
            n++;
        }
        if (!failed && rc != SQLITE_DONE)
        {
            failed = 1;
        }
    }
    for (size_t i = 0; i < n && !failed; ++i)
    {
        sqlite3_reset(del);
        sqlite3_bind_text(del, 1, keys[i], -1, SQLITE_TRANSIENT);
        failed = (sqlite3_step(del) != SQLITE_DONE);
    }
    sqlite3_finalize(sel);
    sqlite3_finalize(del);
    if (!failed)
    {
        sqlite3_stmt * drop = NULL;
        failed = (sqlite3_prepare_v2(g_db, QML_SQL_KV_OWNER_DROP, -1, &drop, NULL) != SQLITE_OK);
        if (!failed)
        {
            sqlite3_bind_text(drop, 1, owner, -1, SQLITE_TRANSIENT);
            failed = (sqlite3_step(drop) != SQLITE_DONE);
        }
        sqlite3_finalize(drop);
    }
    if (failed)
    {
        store_log_fail("owner_take", owner);
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
    }
    else if (commit_or_rollback(__func__, owner) != 0)
    {
        failed = 1;
    }
    pthread_mutex_unlock(&g_lock);
    if (failed)
    {
        qml_sqlite_free_keys(keys, n);
        return -1;
    }
    *out_keys = keys;
    *out_n    = n;
    return 0;
}

//------------------------------------------------------------
//Func : qml_sqlite_free_keys
// Frees a key list returned by qml_sqlite_owner_take.
//------------------------------------------------------------
void qml_sqlite_free_keys(char ** keys, size_t n)
{
    if (keys == NULL)
    {
        return;
    }
    for (size_t i = 0; i < n; ++i)
    {
        free(keys[i]);
    }
    free(keys);
}

//------------------------------------------------------------
//Func : qml_sqlite_block_total_bytes
// Sums the recorded sizes of all tracked KV blocks.
//  in: out_total — receives the total in bytes (0 when the table is empty)
// out: 0 on success; -1 on error
//------------------------------------------------------------
int qml_sqlite_block_total_bytes(int64_t * out_total)
{
    if (g_db == NULL || out_total == NULL)
    {
        return -1;
    }

    *out_total = 0;

    pthread_mutex_lock(&g_lock);

    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_KV_BLOCK_TOTAL_BYTES, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW)
    {
        *out_total = sqlite3_column_int64(stmt, 0);
    }
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);

    return (rc == SQLITE_ROW) ? 0 : -1;
}

//------------------------------------------------------------
//Func : qml_sqlite_block_evict_lru
//
//Returns the LRU block keys whose removal brings the total under
//max_bytes. This function deliberately does NOT touch LMDB itself: the
//caller owns the blob store handle and must delete the returned keys there
//(qml_lmdb_delete) -- keeping the SQLite layer free of blob-store
//dependencies, the same separation the rest of this file follows. The rows
//ARE removed from qml_kv_block here, so a caller that crashes mid-delete
//leaks LMDB bytes rather than leaving phantom accounting; that direction is
//the safe one, since an untracked block is only wasted space while a
//tracked-but-missing block would be reported as a cache hit.
//  in: max_bytes — target ceiling for total tracked bytes (<= 0 means nothing to do)
//  in: out_keys — receives a malloc'd array of strdup'd evicted keys
//  in: out_n — receives the number of evicted keys (bounded at 4096 per call)
// out: 0 on success (possibly 0 keys); -1 on error.
//      Caller must delete the keys from LMDB, then free them with
//      qml_sqlite_block_evict_free.
// side effects: deletes the selected rows from qml_kv_block
//------------------------------------------------------------
int qml_sqlite_block_evict_lru(int64_t max_bytes, char *** out_keys, size_t * out_n)
{
    if (g_db == NULL || out_keys == NULL || out_n == NULL)
    {
        return -1;
    }

    *out_keys = NULL;
    *out_n    = 0;

    int64_t total = 0;
    if (qml_sqlite_block_total_bytes(&total) != 0)
    {
        return -1;
    }

    if (max_bytes <= 0 || total <= max_bytes)
    {
        return 0; // nothing to do
    }

    const size_t CAP = 4096; // bounded per call -- eviction is idempotent, callers repeat
    char **      keys = (char **) malloc(CAP * sizeof(char *));
    if (keys == NULL)
    {
        return -1;
    }
    size_t n = 0;

    pthread_mutex_lock(&g_lock);

    sqlite3_stmt * sel = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_KV_BLOCK_SELECT_LRU, -1, &sel, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        free(keys);
        return -1;
    }
    sqlite3_bind_int(sel, 1, (int) CAP);

    int64_t freed = 0;
    while (n < CAP && sqlite3_step(sel) == SQLITE_ROW && (total - freed) > max_bytes)
    {
        const unsigned char * k = sqlite3_column_text(sel, 0);
        const int64_t         b = sqlite3_column_int64(sel, 1);
        if (k == NULL)
        {
            continue;
        }
        keys[n] = strdup((const char *) k);
        if (keys[n] == NULL)
        {
            break;
        }
        freed += b;
        ++n;
    }
    sqlite3_finalize(sel);

    sqlite3_stmt * del = NULL;
    if (n > 0 && sqlite3_prepare_v2(g_db, QML_SQL_KV_BLOCK_DELETE, -1, &del, NULL) == SQLITE_OK)
    {
        sqlite3_exec(g_db, "BEGIN IMMEDIATE;", NULL, NULL, NULL);
        for (size_t i = 0; i < n; ++i)
        {
            sqlite3_reset(del);
            sqlite3_bind_text(del, 1, keys[i], -1, SQLITE_TRANSIENT);
            sqlite3_step(del);
        }
        (void) commit_or_rollback(__func__, NULL);
        sqlite3_finalize(del);
    }

    pthread_mutex_unlock(&g_lock);

    *out_keys = keys;
    *out_n    = n;
    return 0;
}

//------------------------------------------------------------
//Func : qml_sqlite_block_evict_free
// Frees the key array returned by qml_sqlite_block_evict_lru.
//  in: keys — array to free (NULL-safe)
//  in: n — number of entries in the array
// out: none
//------------------------------------------------------------
void qml_sqlite_block_evict_free(char ** keys, size_t n)
{
    if (keys == NULL)
    {
        return;
    }
    for (size_t i = 0; i < n; ++i)
    {
        free(keys[i]);
    }
    free(keys);
}

//------------------------------------------------------------
//Func : qml_sqlite_archive_rewrite
//
//Called just before a full-snapshot save replaces qml_token_log. If the stored
//log is a PREFIX of the incoming window, the session simply grew and there is
//nothing to preserve. Otherwise the engine rewrote its window -- a context
//shift discarded a middle range -- and the rows that are about to disappear are
//copied into qml_token_archive first.
//
//The prefix test compares token ids, not lengths: a shorter new window is the
//obvious truncation case, but a LONGER window can also be a rewrite (shift plus
//new generation), and length alone cannot tell those apart.
//
//Returns the number of tokens archived (0 = pure growth), -1 on failure. A
//failure here does NOT abort the save: losing the archive is bad, but refusing
//to persist the live session would be worse.
//  in: session_id — session whose stored log is compared to the incoming window
//  in: new_tokens — the incoming full token window, n_new entries (may be NULL)
//  in: n_new — length of the incoming window
// out: number of tokens archived (0 = stored log is a prefix, nothing lost); -1 on failure
// side effects: appends the old log's token ids to qml_token_archive in one transaction
//------------------------------------------------------------
int qml_sqlite_archive_rewrite(const char * session_id, const int32_t * new_tokens, size_t n_new)
{
    if (g_db == NULL || session_id == NULL)
    {
        return -1;
    }

    qml_token_rec_t * old_recs = NULL;
    size_t            n_old    = 0;

    if (qml_sqlite_get_tokens(session_id, &old_recs, &n_old) != 0)
    {
        return -1;
    }

    if (n_old == 0)
    {
        free(old_recs);
        return 0; // nothing stored yet
    }

    int is_prefix = (n_old <= n_new);
    if (is_prefix)
    {
        for (size_t i = 0; i < n_old; ++i)
        {
            if (new_tokens == NULL || old_recs[i].token_id != new_tokens[i])
            {
                is_prefix = 0;
                break;
            }
        }
    }

    if (is_prefix)
    {
        free(old_recs);
        return 0; // session only grew -- the live log already holds this history
    }

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    const int64_t now_us = (int64_t) ts.tv_sec * 1000000 + ts.tv_nsec / 1000;

    pthread_mutex_lock(&g_lock);

    int64_t        next_seq = 0;
    sqlite3_stmt * stmt     = NULL;

    if (sqlite3_prepare_v2(g_db, QML_SQL_ARCHIVE_MAX_SEQ, -1, &stmt, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        free(old_recs);
        return -1;
    }
    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) == SQLITE_ROW)
    {
        next_seq = sqlite3_column_int64(stmt, 0) + 1;
    }
    sqlite3_finalize(stmt);

    if (sqlite3_exec(g_db, "BEGIN IMMEDIATE;", NULL, NULL, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        free(old_recs);
        return -1;
    }

    if (sqlite3_prepare_v2(g_db, QML_SQL_ARCHIVE_INSERT, -1, &stmt, NULL) != SQLITE_OK)
    {
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
        pthread_mutex_unlock(&g_lock);
        free(old_recs);
        return -1;
    }

    size_t archived = 0;
    for (size_t i = 0; i < n_old; ++i)
    {
        sqlite3_reset(stmt);
        sqlite3_bind_text (stmt, 1, session_id, -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 2, next_seq + (int64_t) i);
        sqlite3_bind_int  (stmt, 3, old_recs[i].token_id);
        sqlite3_bind_int64(stmt, 4, now_us);
        if (sqlite3_step(stmt) != SQLITE_DONE)
        {
            sqlite3_finalize(stmt);
            sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
            pthread_mutex_unlock(&g_lock);
            free(old_recs);
            return -1;
        }
        ++archived;
    }
    sqlite3_finalize(stmt);
    if (commit_or_rollback(__func__, session_id) != 0)
    {
        pthread_mutex_unlock(&g_lock);
        free(old_recs);
        return -1;
    }
    pthread_mutex_unlock(&g_lock);

    free(old_recs);
    return (int) archived;
}

//------------------------------------------------------------
//Func : replace_tokens_locked
// DELETE-then-INSERT of a session's token log; no transaction control, no locking.
//  in: session_id — session whose log is replaced
//  in: tokens — the new complete log, n_tokens records (ignored when n_tokens is 0)
//  in: n_tokens — new log length (0 leaves the log empty)
// out: 0 on success; -1 on the first failing statement (caller rolls back)
//------------------------------------------------------------
static int replace_tokens_locked(const char * session_id, const qml_token_rec_t * tokens, size_t n_tokens)
{
    sqlite3_stmt * stmt = NULL;
    if (sqlite3_prepare_v2(g_db, QML_SQL_DELETE_TOKENS, -1, &stmt, NULL) != SQLITE_OK)
    {
        return -1;
    }
    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE)
    {
        return -1;
    }
    if (n_tokens == 0)
    {
        return 0;
    }

    if (sqlite3_prepare_v2(g_db, QML_SQL_INSERT_TOKEN, -1, &stmt, NULL) != SQLITE_OK)
    {
        return -1;
    }
    for (size_t i = 0; i < n_tokens; ++i)
    {
        sqlite3_reset(stmt);
        sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);
        sqlite3_bind_int (stmt, 2, tokens[i].pos);
        sqlite3_bind_int (stmt, 3, tokens[i].seq_id);
        sqlite3_bind_int (stmt, 4, tokens[i].token_id);
        if (sqlite3_step(stmt) != SQLITE_DONE)
        {
            sqlite3_finalize(stmt);
            return -1;
        }
    }
    sqlite3_finalize(stmt);
    return 0;
}

//------------------------------------------------------------
//Func : qml_sqlite_replace_tokens
//DELETE-then-INSERT in ONE transaction, so the stored log is exactly the
//n_tokens rows passed in -- never a mix of new rows and a stale tail from
//an earlier, LONGER save. qml_sqlite_put_tokens alone cannot do
//this: INSERT OR REPLACE only overwrites positions 0..n_tokens-1 and
//silently leaves rows at pos >= n_tokens behind. That produced a live
//data-corruption bug (found 2026-07-30): a session whose meta said 31
//tokens had 58 rows in qml_token_log (31 fresh + 27 stale), so restore
//handed llama-server 58 "resident" tokens for a KV blob that only held
//31 -- positions 31..57 attended to stale KV and the model produced
//semantically wrong continuations. Use this for any FULL-snapshot save;
//qml_sqlite_append_tokens remains the incremental path.
//  in: session_id — session whose log is replaced
//  in: tokens — the new complete log, n_tokens records (ignored when n_tokens is 0)
//  in: n_tokens — new log length (0 leaves the log empty)
// out: 0 on success; -1 on error (whole transaction rolled back, old log kept)
// side effects: deletes then re-inserts the session's qml_token_log rows atomically
//------------------------------------------------------------
int qml_sqlite_replace_tokens(const char * session_id, const qml_token_rec_t * tokens, size_t n_tokens)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    if (sqlite3_exec(g_db, "BEGIN IMMEDIATE;", NULL, NULL, NULL) != SQLITE_OK)
    {
        store_log_fail("replace_tokens BEGIN", session_id);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    if (replace_tokens_locked(session_id, tokens, n_tokens) != 0)
    {
        store_log_fail("replace_tokens", session_id);
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    const int rc = commit_or_rollback("replace_tokens", session_id);
    pthread_mutex_unlock(&g_lock);
    return rc;
}

//------------------------------------------------------------
//Func : archive_rewritten_locked
// Same rule as qml_sqlite_archive_rewrite, run INSIDE the caller's
// transaction: when the stored log is not a prefix of the incoming window, the
// stored rows are appended to qml_token_archive. The writes sit under a
// SAVEPOINT, so an archive failure undoes only the archive rows and the save
// continues (losing the archive is bad, refusing to persist the live session
// is worse -- the existing policy).
//  in: session_id — session whose stored log is compared to the incoming window
//  in: new_tokens, n_new — the incoming full token window
//  in: now_us — archived_at value
// out: number of tokens archived (0 = pure growth); -1 on failure (logged)
// side effects: caller holds g_lock with a transaction open
//------------------------------------------------------------
static int archive_rewritten_locked(const char * session_id, const int32_t * new_tokens,
                                    size_t n_new, int64_t now_us)
{
    sqlite3_stmt * stmt = NULL;
    if (sqlite3_prepare_v2(g_db, QML_SQL_SELECT_TOKENS_ORDERED, -1, &stmt, NULL) != SQLITE_OK)
    {
        store_log_fail("archive_rewritten select", session_id);
        return -1;
    }
    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);

    size_t    cap = 1024, n_old = 0;
    int32_t * old = (int32_t *) malloc(cap * sizeof(int32_t));
    int       rc  = SQLITE_DONE;
    while (old != NULL && (rc = sqlite3_step(stmt)) == SQLITE_ROW)
    {
        if (n_old == cap)
        {
            cap *= 2;
            int32_t * grown = (int32_t *) realloc(old, cap * sizeof(int32_t));
            if (grown == NULL)
            {
                free(old);
                old = NULL;
                break;
            }
            old = grown;
        }
        old[n_old++] = sqlite3_column_int(stmt, 2);
    }
    sqlite3_finalize(stmt);
    if (old == NULL || rc != SQLITE_DONE)
    {
        store_log_fail("archive_rewritten select", session_id);
        free(old);
        return -1;
    }

    int is_prefix = (n_old <= n_new);
    for (size_t i = 0; is_prefix && i < n_old; ++i)
    {
        is_prefix = (new_tokens != NULL && old[i] == new_tokens[i]);
    }
    if (n_old == 0 || is_prefix)
    {
        free(old);
        return 0;
    }

    if (sqlite3_exec(g_db, "SAVEPOINT qml_archive;", NULL, NULL, NULL) != SQLITE_OK)
    {
        store_log_fail("archive_rewritten savepoint", session_id);
        free(old);
        return -1;
    }

    int64_t next_seq = 0;
    int     ok       = (sqlite3_prepare_v2(g_db, QML_SQL_ARCHIVE_MAX_SEQ, -1, &stmt, NULL) == SQLITE_OK);
    if (ok)
    {
        sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW)
        {
            next_seq = sqlite3_column_int64(stmt, 0) + 1;
        }
        sqlite3_finalize(stmt);
    }
    ok = ok && (sqlite3_prepare_v2(g_db, QML_SQL_ARCHIVE_INSERT, -1, &stmt, NULL) == SQLITE_OK);
    for (size_t i = 0; ok && i < n_old; ++i)
    {
        sqlite3_reset(stmt);
        sqlite3_bind_text (stmt, 1, session_id, -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 2, next_seq + (int64_t) i);
        sqlite3_bind_int  (stmt, 3, old[i]);
        sqlite3_bind_int64(stmt, 4, now_us);
        ok = (sqlite3_step(stmt) == SQLITE_DONE);
    }
    if (stmt != NULL)
    {
        sqlite3_finalize(stmt);
    }
    free(old);

    if (!ok)
    {
        store_log_fail("archive_rewritten insert", session_id);
        sqlite3_exec(g_db, "ROLLBACK TO qml_archive;", NULL, NULL, NULL);
        sqlite3_exec(g_db, "RELEASE qml_archive;", NULL, NULL, NULL);
        return -1;
    }
    sqlite3_exec(g_db, "RELEASE qml_archive;", NULL, NULL, NULL);
    return (int) n_old;
}

//------------------------------------------------------------
//Func : qml_sqlite_save_session
// Saves the session row (plus the llama ext row), the archive of rows a context
// shift is about to discard, and the full token log in ONE transaction.
// Why (2026-10-02, H200 service): these were three transactions. When the
// token-log one failed (SQLITE_BUSY: another connection held the write lock),
// the committed session row said n_tokens > 0 while the token log was empty,
// and the engine then refused the session ("stored history ... could not be
// read"). In one transaction a failure keeps the previous row and log together.
//  in: session_id — session key
//  in: meta — llama row with ext fields; NULL writes the base row only
//  in: base — base row used when meta is NULL (ignored otherwise)
//  in: token_ids, n_tokens — the session's full token window
//  in: seq_id — seq id stored on every token row
//  in: clear_when_empty — when n_tokens is 0: 1 clears the stored log, 0 leaves it
// out: 0 saved; -1 nothing written (rolled back, reason on stderr)
// side effects: archive rows go in under a SAVEPOINT; their failure does not fail the save
//------------------------------------------------------------
int qml_sqlite_save_session(const char * session_id,
                            const qml_llama_session_meta_t * meta,
                            const qml_session_meta_t * base,
                            const int32_t * token_ids, size_t n_tokens,
                            int32_t seq_id, int clear_when_empty)
{
    if (g_db == NULL || session_id == NULL || (meta == NULL && base == NULL))
    {
        return -1;
    }

    qml_token_rec_t * recs = NULL;
    if (n_tokens > 0)
    {
        recs = (qml_token_rec_t *) malloc(n_tokens * sizeof(qml_token_rec_t));
        if (recs == NULL)
        {
            return -1;
        }
        for (size_t i = 0; i < n_tokens; ++i)
        {
            recs[i].pos      = (int32_t) i;
            recs[i].seq_id   = seq_id;
            recs[i].token_id = token_ids[i];
        }
    }

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    const int64_t now_us = (int64_t) ts.tv_sec * 1000000 + ts.tv_nsec / 1000;

    pthread_mutex_lock(&g_lock);

    if (sqlite3_exec(g_db, "BEGIN IMMEDIATE;", NULL, NULL, NULL) != SQLITE_OK)
    {
        store_log_fail("save_session BEGIN", session_id);
        pthread_mutex_unlock(&g_lock);
        free(recs);
        return -1;
    }

    const char * step = "save_session session row";
    int ok = (upsert_session_base(session_id, (meta != NULL) ? &meta->base : base) == SQLITE_DONE);
    if (ok && meta != NULL)
    {
        step = "save_session ext row";
        ok = (upsert_llama_ext(session_id, meta) == SQLITE_DONE);
    }
    if (ok)
    {
        (void) archive_rewritten_locked(session_id, token_ids, n_tokens, now_us);
    }
    if (ok && (n_tokens > 0 || clear_when_empty))
    {
        step = "save_session token log";
        ok = (replace_tokens_locked(session_id, recs, n_tokens) == 0);
    }

    int rc = -1;
    if (!ok)
    {
        store_log_fail(step, session_id);
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
    }
    else
    {
        rc = commit_or_rollback("save_session COMMIT", session_id);
    }

    pthread_mutex_unlock(&g_lock);
    free(recs);
    return rc;
}

//------------------------------------------------------------
//Func : qml_sqlite_put_tokens
// Inserts token records as-is inside one transaction (incremental path; no delete first).
//  in: session_id — session the rows belong to
//  in: tokens — records carrying pos/seq_id/token_id, n_tokens entries
//  in: n_tokens — number of records (0 is a successful no-op)
// out: 0 on success; -1 on error (whole transaction rolled back)
// side effects: inserts qml_token_log rows (INSERT OR REPLACE per the schema statement)
//------------------------------------------------------------
int qml_sqlite_put_tokens(const char * session_id, const qml_token_rec_t * tokens, size_t n_tokens)
{
    if (g_db == NULL)
    {
        return -1;
    }

    if (n_tokens == 0)
    {
        return 0;
    }

    pthread_mutex_lock(&g_lock);

    char * err = NULL;
    int rc = sqlite3_exec(g_db, "BEGIN IMMEDIATE;", NULL, NULL, &err);
    if (rc != SQLITE_OK)
    {
        sqlite3_free(err);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_stmt * stmt = NULL;
    rc = sqlite3_prepare_v2(g_db, QML_SQL_INSERT_TOKEN, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    for (size_t i = 0; i < n_tokens; ++i)
    {
        sqlite3_reset(stmt);
        sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);
        sqlite3_bind_int (stmt, 2, tokens[i].pos);
        sqlite3_bind_int (stmt, 3, tokens[i].seq_id);
        sqlite3_bind_int (stmt, 4, tokens[i].token_id);

        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE)
        {
            sqlite3_finalize(stmt);
            sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
            pthread_mutex_unlock(&g_lock);
            return -1;
        }
    }

    sqlite3_finalize(stmt);

    if (commit_or_rollback(__func__, NULL) != 0)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    pthread_mutex_unlock(&g_lock);
    return 0;
}

//------------------------------------------------------------
//Func : qml_sqlite_get_tokens
// Reads a session's full token log, ordered by position.
//  in: session_id — session to read
//  in: out_tokens — receives a malloc'd record array (free via qml_sqlite_free_tokens)
//  in: out_n — receives the record count (0 for an unknown session, still returns 0)
// out: 0 on success; -1 on error
//------------------------------------------------------------
int qml_sqlite_get_tokens(const char * session_id, qml_token_rec_t ** out_tokens, size_t * out_n)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_SELECT_TOKENS_ORDERED, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);

    size_t capacity = 1024;
    size_t count    = 0;
    qml_token_rec_t * buf = (qml_token_rec_t *) malloc(capacity * sizeof(qml_token_rec_t));

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW)
    {
        if (count == capacity)
        {
            capacity *= 2;
            buf = (qml_token_rec_t *) realloc(buf, capacity * sizeof(qml_token_rec_t));
        }

        buf[count].pos      = sqlite3_column_int(stmt, 0);
        buf[count].seq_id   = sqlite3_column_int(stmt, 1);
        buf[count].token_id = sqlite3_column_int(stmt, 2);
        ++count;
    }

    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);

    if (rc != SQLITE_DONE)
    {
        free(buf);
        return -1;
    }

    *out_tokens = buf;
    *out_n      = count;

    return 0;
}

//------------------------------------------------------------
//Func : qml_sqlite_free_tokens
// Frees the record array returned by qml_sqlite_get_tokens.
//  in: tokens — array to free (NULL-safe)
// out: none
//------------------------------------------------------------
void qml_sqlite_free_tokens(qml_token_rec_t * tokens)
{
    free(tokens);
}

//------------------------------------------------------------
//Func : qml_sqlite_last_pos
// Reads the last_pos column of a session's base row.
//  in: session_id — session to look up
//  in: out_last_pos — receives the stored last_pos on hit
// out: 0 found; 1 no such session; -1 on error
//------------------------------------------------------------
int qml_sqlite_last_pos(const char * session_id, int32_t * out_last_pos)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_SELECT_LAST_POS, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);

    if (rc == SQLITE_ROW)
    {
        *out_last_pos = sqlite3_column_int(stmt, 0);
        sqlite3_finalize(stmt);
        pthread_mutex_unlock(&g_lock);
        return 0;
    }

    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);

    return (rc == SQLITE_DONE) ? 1 : -1;
}

//------------------------------------------------------------
//Func : qml_sqlite_append_tokens
// Appends tokens after the session's stored last_pos (or from pos 0 for a new session).
//  in: session_id — session whose log grows
//  in: token_ids — token ids to append, n_tokens entries
//  in: n_tokens — number of tokens (0 is a successful no-op)
//  in: seq_id — sequence id recorded for every appended row
// out: 0 on success; -1 on error (lookup, allocation, or insert failure)
// side effects: inserts qml_token_log rows in one transaction
//------------------------------------------------------------
int qml_sqlite_append_tokens(const char * session_id, const int32_t * token_ids, size_t n_tokens, int32_t seq_id)
{
    if (n_tokens == 0)
    {
        return 0;
    }

    int32_t last_pos = -1;
    const int get_rc = qml_sqlite_last_pos(session_id, &last_pos);
    if (get_rc == -1)
    {
        return -1;
    }
    if (get_rc == 1)
    {
        last_pos = -1; // unknown session -- start fresh, same as the schema default
    }

    qml_token_rec_t * recs = (qml_token_rec_t *) malloc(n_tokens * sizeof(qml_token_rec_t));
    if (recs == NULL)
    {
        return -1;
    }

    for (size_t i = 0; i < n_tokens; ++i)
    {
        recs[i].pos      = last_pos + 1 + (int32_t) i;
        recs[i].seq_id   = seq_id;
        recs[i].token_id = token_ids[i];
    }

    const int rc = qml_sqlite_put_tokens(session_id, recs, n_tokens);
    free(recs);

    return rc;
}

//------------------------------------------------------------
//Func : qml_sqlite_control_put
// Enqueues a control-plane command plus its optional block-hash list, in one transaction.
//  in: target_engine — engine name the command is addressed to
//  in: target_pid — pid of the target engine process
//  in: cmd_type — command verb (e.g. RESTORE / PREWARM)
//  in: session_id — session the command refers to (NULL is stored as "")
//  in: block_hashes — optional KV block hashes, n_block_hashes entries
//  in: n_block_hashes — number of block hashes (0 inserts no block rows)
//  in: out_cmd_id — optional (may be NULL); receives the new command's rowid
// out: 0 on success; -1 on error (whole transaction rolled back)
// side effects: inserts into the control-command and control-command-blocks tables
//------------------------------------------------------------
int qml_sqlite_control_put(
    const char * target_engine,
    int32_t      target_pid,
    const char * cmd_type,
    const char * session_id,
    const char * const * block_hashes,
    size_t       n_block_hashes,
    int64_t    * out_cmd_id)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    char * err = NULL;
    int rc = sqlite3_exec(g_db, "BEGIN IMMEDIATE;", NULL, NULL, &err);
    if (rc != SQLITE_OK)
    {
        sqlite3_free(err);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_stmt * stmt = NULL;
    rc = sqlite3_prepare_v2(g_db, QML_SQL_INSERT_CONTROL_CMD, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_text (stmt, 1, target_engine,        -1, SQLITE_TRANSIENT);
    sqlite3_bind_int  (stmt, 2, target_pid);
    sqlite3_bind_text (stmt, 3, cmd_type,              -1, SQLITE_TRANSIENT);
    sqlite3_bind_text (stmt, 4, session_id ? session_id : "", -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 5, (int64_t) time(NULL));

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE)
    {
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    const int64_t cmd_id = sqlite3_last_insert_rowid(g_db);

    for (size_t i = 0; i < n_block_hashes; ++i)
    {
        stmt = NULL;
        rc = sqlite3_prepare_v2(g_db, QML_SQL_INSERT_CONTROL_CMD_BLOCK, -1, &stmt, NULL);
        if (rc != SQLITE_OK)
        {
            sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
            pthread_mutex_unlock(&g_lock);
            return -1;
        }

        sqlite3_bind_int64(stmt, 1, cmd_id);
        sqlite3_bind_text (stmt, 2, block_hashes[i], -1, SQLITE_TRANSIENT);

        rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        if (rc != SQLITE_DONE)
        {
            sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
            pthread_mutex_unlock(&g_lock);
            return -1;
        }
    }

    if (commit_or_rollback(__func__, NULL) != 0)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    pthread_mutex_unlock(&g_lock);

    if (out_cmd_id != NULL)
    {
        *out_cmd_id = cmd_id;
    }

    return 0;
}

//------------------------------------------------------------
//Func : qml_sqlite_control_claim
//SELECT + UPDATE wrapped in one BEGIN IMMEDIATE transaction -- IMMEDIATE
//takes the write lock up front (rather than at first write), so a second
//caller racing this same call blocks at BEGIN IMMEDIATE until the first
//one COMMITs, instead of both reading the same PENDING row before either
//writes. That's what makes the claim atomic, not the UPDATE alone.
//  in: target_pid — pid whose pending queue is polled
//  in: out_cmd — receives the claimed command's fields (zeroed then filled)
// out: 0 claimed; 1 nothing pending for this pid; -1 on error
// side effects: marks the claimed row CLAIMED inside the same transaction
//------------------------------------------------------------
int qml_sqlite_control_claim(int32_t target_pid, qml_control_cmd_t * out_cmd)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    char * err = NULL;
    int rc = sqlite3_exec(g_db, "BEGIN IMMEDIATE;", NULL, NULL, &err);
    if (rc != SQLITE_OK)
    {
        sqlite3_free(err);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_stmt * stmt = NULL;
    rc = sqlite3_prepare_v2(g_db, QML_SQL_CLAIM_NEXT_CONTROL_CMD, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_int(stmt, 1, target_pid);
    rc = sqlite3_step(stmt);

    if (rc == SQLITE_DONE)
    {
        // nothing pending for this target_pid
        sqlite3_finalize(stmt);
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
        pthread_mutex_unlock(&g_lock);
        return 1;
    }

    if (rc != SQLITE_ROW)
    {
        sqlite3_finalize(stmt);
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    memset(out_cmd, 0, sizeof(*out_cmd));

    // column order per QML_SQL_CLAIM_NEXT_CONTROL_CMD:
    // cmd_id, target_engine, cmd_type, session_id, created_at
    out_cmd->cmd_id = sqlite3_column_int64(stmt, 0);

    const unsigned char * target_engine = sqlite3_column_text(stmt, 1);
    const unsigned char * cmd_type      = sqlite3_column_text(stmt, 2);
    const unsigned char * session_id    = sqlite3_column_text(stmt, 3);

    strncpy(out_cmd->target_engine, (const char *) target_engine, sizeof(out_cmd->target_engine) - 1);
    strncpy(out_cmd->cmd_type,      (const char *) cmd_type,      sizeof(out_cmd->cmd_type) - 1);
    strncpy(out_cmd->session_id,    (const char *) session_id,    sizeof(out_cmd->session_id) - 1);
    out_cmd->created_at = sqlite3_column_int64(stmt, 4);

    sqlite3_finalize(stmt);

    stmt = NULL;
    rc = sqlite3_prepare_v2(g_db, QML_SQL_MARK_CONTROL_CMD_CLAIMED, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_int64(stmt, 1, out_cmd->cmd_id);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE)
    {
        sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    if (commit_or_rollback(__func__, NULL) != 0)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    pthread_mutex_unlock(&g_lock);
    return 0;
}

//------------------------------------------------------------
//Func : qml_sqlite_control_set_status
// Sets a command row's status string (e.g. DONE / FAILED).
//  in: cmd_id — rowid of the command to update
//  in: status — new status value
// out: 0 on success (also when no row matched); -1 on error
//------------------------------------------------------------
int qml_sqlite_control_set_status(int64_t cmd_id, const char * status)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_UPDATE_CONTROL_CMD_STATUS, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_text (stmt, 1, status, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, cmd_id);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    pthread_mutex_unlock(&g_lock);

    return (rc == SQLITE_DONE) ? 0 : -1;
}

//------------------------------------------------------------
//Func : qml_sqlite_control_blocks
// Reads the block-hash list attached to a command.
//  in: cmd_id — rowid of the command
//  in: out_hashes — receives a malloc'd array of block-hash records (caller frees)
//  in: out_n — receives the record count (0 when the command has no blocks)
// out: 0 on success; -1 on error
//------------------------------------------------------------
int qml_sqlite_control_blocks(int64_t cmd_id, qml_control_cmd_block_t ** out_hashes, size_t * out_n)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_SELECT_CONTROL_CMD_BLOCKS, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_int64(stmt, 1, cmd_id);

    size_t capacity = 16;
    size_t count    = 0;
    qml_control_cmd_block_t * buf = (qml_control_cmd_block_t *) malloc(capacity * sizeof(qml_control_cmd_block_t));

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW)
    {
        if (count == capacity)
        {
            capacity *= 2;
            buf = (qml_control_cmd_block_t *) realloc(buf, capacity * sizeof(qml_control_cmd_block_t));
        }

        const unsigned char * block_hash = sqlite3_column_text(stmt, 0);
        memset(buf[count].block_hash, 0, sizeof(buf[count].block_hash));
        strncpy(buf[count].block_hash, (const char *) block_hash, sizeof(buf[count].block_hash) - 1);
        ++count;
    }

    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);

    if (rc != SQLITE_DONE)
    {
        free(buf);
        return -1;
    }

    *out_hashes = buf;
    *out_n      = count;

    return 0;
}

//------------------------------------------------------------
//Func : qml_sqlite_compact
// Reclaims file space: WAL checkpoint (TRUNCATE) followed by VACUUM.
//  in: none
// out: 0 on success; -1 on error
// side effects: truncates the WAL file and rewrites the database file
//------------------------------------------------------------
int qml_sqlite_compact(void)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    char * err = NULL;
    int rc = sqlite3_exec(g_db, "PRAGMA wal_checkpoint(TRUNCATE);", NULL, NULL, &err);
    if (rc != SQLITE_OK)
    {
        sqlite3_free(err);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    rc = sqlite3_exec(g_db, "VACUUM;", NULL, NULL, &err);
    if (rc != SQLITE_OK)
    {
        sqlite3_free(err);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    pthread_mutex_unlock(&g_lock);
    return 0;
}

//------------------------------------------------------------
//Func : session_get_lock_locked (internal, caller must hold g_lock)
// Reads a session row's lock fields; shared by get_lock and acquire_lock.
//  in: session_id — session to inspect
//  in: out_owner_pid — receives the recorded owner pid
//  in: out_locked — receives 1 if lock_state is LOCKED, else 0
// out: 0 found; 1 no such session; -1 on error
//------------------------------------------------------------
static int session_get_lock_locked(const char * session_id, int32_t * out_owner_pid, int * out_locked)
{
    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_SESSION_GET_LOCK, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        return -1;
    }

    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);

    if (rc == SQLITE_ROW)
    {
        *out_owner_pid = sqlite3_column_int(stmt, 0);
        *out_locked    = (strcmp((const char *) sqlite3_column_text(stmt, 1), "LOCKED") == 0) ? 1 : 0;
        sqlite3_finalize(stmt);
        return 0;
    }

    sqlite3_finalize(stmt);
    return (rc == SQLITE_DONE) ? 1 : -1;
}

//------------------------------------------------------------
//Func : qml_sqlite_lock
// Takes the session's ownership lock; steals it when the recorded owner pid is dead.
//  in: session_id — session to lock (must have a base row)
//  in: pid — pid requesting ownership
// out: 0 acquired (fresh, re-entrant, or stolen from a dead owner);
//      1 held by a different LIVE pid; -1 on error or unknown session
// side effects: updates owner_pid/lock_state; logs a takeover from a dead pid to stderr
//------------------------------------------------------------
int qml_sqlite_lock(const char * session_id, int32_t pid)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_SESSION_ACQUIRE_LOCK, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_int (stmt, 1, pid);
    sqlite3_bind_text(stmt, 2, session_id, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int (stmt, 3, pid);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    if (sqlite3_changes(g_db) == 1)
    {
        pthread_mutex_unlock(&g_lock);
        return 0; // acquired (or already ours)
    }

    // UPDATE affected nothing -- either session_id doesn't exist, or it's
    // genuinely held by a different pid. Distinguish via a follow-up read.
    int32_t owner_pid = 0;
    int     locked     = 0;
    const int get_rc = session_get_lock_locked(session_id, &owner_pid, &locked);

    if (get_rc != 0)
    {
        pthread_mutex_unlock(&g_lock);
        return -1; // session_id unknown, or lookup failed
    }

    // Crash consistency: a lock whose owner died (e.g. kill -9, OOM) must not
    // outlive that process, or every control-plane RESTORE/PREWARM for the
    // session fails forever (hit live 2026-08-04: a demo server was SIGKILLed
    // between scenarios and all later PREWARMs returned FAILED on its stale
    // lock). All qml processes share one embedded SQLite file, which is a
    // same-host design by construction, so kill(pid, 0) is a valid liveness
    // probe here: ESRCH means the recorded owner no longer exists. EPERM
    // means it exists but belongs to another user -- that is a LIVE owner,
    // not a stale one, and stays a conflict.
    if (locked && owner_pid != pid && owner_pid > 0 &&
        kill((pid_t) owner_pid, 0) != 0 && errno == ESRCH)
    {
        sqlite3_stmt * steal = NULL;
        rc = sqlite3_prepare_v2(g_db, QML_SQL_SESSION_STEAL_LOCK, -1, &steal, NULL);
        if (rc == SQLITE_OK)
        {
            sqlite3_bind_int (steal, 1, pid);
            sqlite3_bind_text(steal, 2, session_id, -1, SQLITE_TRANSIENT);
            sqlite3_bind_int (steal, 3, owner_pid);
            rc = sqlite3_step(steal);
            sqlite3_finalize(steal);
            if (rc == SQLITE_DONE && sqlite3_changes(g_db) == 1)
            {
                fprintf(stderr,
                        "[qml] session '%s': took over lock from dead pid %d (new owner %d)\n",
                        session_id, (int) owner_pid, (int) pid);
                pthread_mutex_unlock(&g_lock);
                return 0; // acquired from a dead owner
            }
        }
    }

    pthread_mutex_unlock(&g_lock);
    return 1; // held by a different, LIVE pid -- genuine conflict
}

//------------------------------------------------------------
//Func : qml_sqlite_unlock
// Releases the session lock when this pid owns it; a non-owner release is a silent no-op.
//  in: session_id — session to unlock
//  in: pid — pid that must match the recorded owner for the row to change
// out: 0 on success (statement ran, whether or not a row matched); -1 on error
// side effects: clears lock_state on the owned session row
//------------------------------------------------------------
int qml_sqlite_unlock(const char * session_id, int32_t pid)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);

    sqlite3_stmt * stmt = NULL;
    int rc = sqlite3_prepare_v2(g_db, QML_SQL_SESSION_RELEASE_LOCK, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int (stmt, 2, pid);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    pthread_mutex_unlock(&g_lock);

    return (rc == SQLITE_DONE) ? 0 : -1;
}

//------------------------------------------------------------
//Func : qml_sqlite_lock_state
// Public, mutex-guarded wrapper over session_get_lock_locked.
//  in: session_id — session to inspect
//  in: out_owner_pid — receives the recorded owner pid
//  in: out_locked — receives 1 if lock_state is LOCKED, else 0
// out: 0 found; 1 no such session; -1 on error
//------------------------------------------------------------
int qml_sqlite_lock_state(const char * session_id, int32_t * out_owner_pid, int * out_locked)
{
    if (g_db == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&g_lock);
    const int rc = session_get_lock_locked(session_id, out_owner_pid, out_locked);
    pthread_mutex_unlock(&g_lock);

    return rc;
}

//===========================================================================//
// [FTS-RECALL] BEGIN — archive full-text search primitives.
// Additive-only feature block; see the matching marker in qml_sqlite_store.h
// for the removal instructions. Default OFF at runtime (QML_ARCHIVE_RECALL).
//===========================================================================//

//Func : qml_fts_init (static helper)
// in: none (uses g_db under g_lock held by caller)
//out: 0 when both the archive table and the FTS5 virtual table exist,
//     -1 when FTS5 is unavailable in the linked SQLite
static int qml_fts_init(void)
{
    // The archive table is normally created by the gateway's window trim;
    // create it here too so an engine-only deployment can run the bench.
    sqlite3_exec(g_db,
        "CREATE TABLE IF NOT EXISTS qml_token_log_archive ("
        " session_id TEXT, orig_pos INTEGER, seq_id INTEGER,"
        " token_id INTEGER, archived_at INTEGER)",
        NULL, NULL, NULL);
    // [v1.1] composite index: without it every per-session archive lookup was a
    // full table scan (measured at 5.7M rows during the LongMemEval run)
    sqlite3_exec(g_db,
        "CREATE INDEX IF NOT EXISTS idx_qml_archive_sid"
        " ON qml_token_log_archive(session_id, orig_pos)",
        NULL, NULL, NULL);
    // [v1.1] schema migration: the v1 FTS table had no indexed session tag, so
    // MATCH ranked candidates across ALL sessions before the session filter.
    // If the old shape is present, drop it -- the lazy indexer rebuilds per
    // session on first recall.
    {
        sqlite3_stmt * chk = NULL;
        if (sqlite3_prepare_v2(g_db, "SELECT tag FROM qml_archive_fts LIMIT 0",
                               -1, &chk, NULL) != SQLITE_OK)
        {
            sqlite3_exec(g_db, "DROP TABLE IF EXISTS qml_archive_fts", NULL, NULL, NULL);
        }
        if (chk != NULL)
        {
            sqlite3_finalize(chk);
        }
    }
    char * err = NULL;
    int rc = sqlite3_exec(g_db,
        "CREATE VIRTUAL TABLE IF NOT EXISTS qml_archive_fts USING fts5("
        " content, tag, session_id UNINDEXED, turn_pos UNINDEXED, turn_end UNINDEXED)",
        NULL, NULL, &err);
    if (rc != SQLITE_OK)
    {
        // typical cause: SQLite built without FTS5 — feature unavailable
        if (err != NULL)
        {
            sqlite3_free(err);
        }
        return -1;
    }
    return 0;
}

// [FTS-SCAN-FIX] Forward declaration — the definition is further below on the search path (writes the
// FNV-1a 64-bit hex of session_id to out[17]). It is here so that the status check below uses the same tag.
static void qml_fts_tag(const char * session_id, char * out);

int qml_sqlite_fts_status(const char * session_id)
{
    if (g_db == NULL || session_id == NULL)
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    if (qml_fts_init() != 0)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_stmt * stmt = NULL;
    int have_archive = 0;
    int have_fts     = 0;
    if (sqlite3_prepare_v2(g_db,
            "SELECT 1 FROM qml_token_log_archive WHERE session_id=? LIMIT 1",
            -1, &stmt, NULL) == SQLITE_OK)
    {
        sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_STATIC);
        have_archive = (sqlite3_step(stmt) == SQLITE_ROW);
        sqlite3_finalize(stmt);
    }
    //=======================================================================//
    // [FTS-SCAN-FIX 2026-08-14] This existence check must filter after narrowing the range with MATCH.
    //
    // What was the problem: the previous statement was the following:
    //     SELECT 1 FROM qml_archive_fts WHERE session_id=? LIMIT 1
    // qml_archive_fts is an FTS5 virtual table, so a B-tree index cannot be built on a plain column such as
    // session_id, and SQLite scans the whole table. The cost is proportional not to the number of documents of
    // this session but to the number of rows of all sessions in the table.
    // Measured (2026-08-14, H200): 68.2ms at 241,773 rows, 904.2ms at 3,150,252 rows —
    // 13x the rows gives 13x the time (linear). This call runs once per recall, so
    // it was added directly to the per-question latency.
    //
    // What it was changed to: MATCH 'tag:<hex>' selects only this session's rows, and the session_id
    // equality is kept as is. tag is the FNV-1a 64-bit hex of session_id and is in the FTS
    // index, so the scan disappears (measured 9.9ms, 91x). The session_id equality
    // is kept as a correctness backstop against tag collisions; this is the same method the search path
    // (qml_sqlite_fts_search) already uses.
    //
    // Revert: replace the block below with the previous two lines (the SELECT statement in the comment above
    // + one bind_text). The behavior (return value) is the same; only the speed differs.
    //=======================================================================//
    char fts_tag[17];
    qml_fts_tag(session_id, fts_tag);
    char fts_probe[32];
    snprintf(fts_probe, sizeof(fts_probe), "tag:%s", fts_tag);
    if (sqlite3_prepare_v2(g_db,
            "SELECT 1 FROM qml_archive_fts"
            " WHERE qml_archive_fts MATCH ? AND session_id=? LIMIT 1",
            -1, &stmt, NULL) == SQLITE_OK)
    {
        sqlite3_bind_text(stmt, 1, fts_probe, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, session_id, -1, SQLITE_STATIC);
        have_fts = (sqlite3_step(stmt) == SQLITE_ROW);
        sqlite3_finalize(stmt);
    }
    pthread_mutex_unlock(&g_lock);
    if (!have_archive)
    {
        return 2;
    }
    return have_fts ? 1 : 0;
}

int qml_sqlite_archive_fetch(const char * session_id,
                             int32_t ** out_pos, int32_t ** out_tok, size_t * out_n)
{
    if (g_db == NULL || session_id == NULL || out_pos == NULL ||
        out_tok == NULL || out_n == NULL)
    {
        return -1;
    }
    *out_pos = NULL;
    *out_tok = NULL;
    *out_n   = 0;
    pthread_mutex_lock(&g_lock);
    sqlite3_stmt * stmt = NULL;
    if (sqlite3_prepare_v2(g_db,
            "SELECT orig_pos, token_id FROM qml_token_log_archive"
            " WHERE session_id=? ORDER BY orig_pos",
            -1, &stmt, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_STATIC);
    size_t cap = 4096;
    size_t n   = 0;
    int32_t * pos = (int32_t *) malloc(cap * sizeof(int32_t));
    int32_t * tok = (int32_t *) malloc(cap * sizeof(int32_t));
    while (pos != NULL && tok != NULL && sqlite3_step(stmt) == SQLITE_ROW)
    {
        if (n == cap)
        {
            cap *= 2;
            int32_t * np = (int32_t *) realloc(pos, cap * sizeof(int32_t));
            int32_t * nt = (int32_t *) realloc(tok, cap * sizeof(int32_t));
            if (np == NULL || nt == NULL)
            {
                free(np != NULL ? np : pos);
                free(nt != NULL ? nt : tok);
                pos = tok = NULL;
                break;
            }
            pos = np;
            tok = nt;
        }
        pos[n] = (int32_t) sqlite3_column_int64(stmt, 0);
        tok[n] = (int32_t) sqlite3_column_int(stmt, 1);
        n++;
    }
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);
    if (pos == NULL || tok == NULL || n == 0)
    {
        free(pos);
        free(tok);
        return -1;
    }
    *out_pos = pos;
    *out_tok = tok;
    *out_n   = n;
    return 0;
}

//Func : qml_fts_tag (static helper)
// in: session_id — session address
// in: out — 17-byte buffer for the tag
//out: writes a 16-hex-char token derived from the session id (FNV-1a) — a
//     single FTS5 token so 'tag:<hex>' scopes MATCH to one session at the
//     index level instead of rank-then-filter
static void qml_fts_tag(const char * session_id, char * out)
{
    unsigned long long h = 1469598103934665603ULL;
    for (const unsigned char * p = (const unsigned char *) session_id; *p; p++)
    {
        h ^= *p;
        h *= 1099511628211ULL;
    }
    snprintf(out, 17, "%016llx", h);
}

//===========================================================================//
// [FTS-BULK-FIX 2026-08-14] Build the index in one transaction + a reused statement.
//
// What was the problem: for every single document, qml_sqlite_fts_add ① opened and closed its own
// implicit transaction, causing a WAL commit (a network fsync on NFS storage), and
// ② prepared the INSERT statement anew each time. Measured (2026-08-14, H200):
// the first index build of 19,446 archive documents took **77.7 s** (about 4ms per document). This value
// does not depend on the model; it was the same 80.7 s on the 70B store as well. The first recall pays this cost
// in full, so the first question a user asks after reconnecting takes 78 s.
//
// What it was changed to: _batch_begin()/_batch_end() let the caller wrap everything in one
// transaction, and the INSERT statement is prepared only once as static and reused
// (reset on every call). A single call outside a batch behaves exactly as before
// (implicit transaction).
//
// Revert: delete the 3 places marked [FTS-BULK-FIX] — this block, the 2 declarations in the header,
// and the begin/end calls wrapping the lazy index loop in the hook — to get the previous behavior.
//===========================================================================//
static sqlite3_stmt * g_fts_ins_stmt = NULL;   // [FTS-BULK-FIX] reused statement
static int            g_fts_in_batch = 0;      // [FTS-BULK-FIX] whether a batch is in progress

int qml_sqlite_fts_begin(void)
{
    if (g_db == NULL)
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    if (g_fts_in_batch)
    {
        pthread_mutex_unlock(&g_lock);
        return 0;                   // already open (nested calls are harmless)
    }
    char * err = NULL;
    const int rc = sqlite3_exec(g_db, "BEGIN IMMEDIATE;", NULL, NULL, &err);
    if (rc == SQLITE_OK)
    {
        g_fts_in_batch = 1;
    }
    else if (err != NULL)
    {
        sqlite3_free(err);
    }
    pthread_mutex_unlock(&g_lock);
    return (rc == SQLITE_OK) ? 0 : -1;
}

int qml_sqlite_fts_end(int commit)
{
    if (g_db == NULL)
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    if (!g_fts_in_batch)
    {
        pthread_mutex_unlock(&g_lock);
        return 0;
    }
    int rc = 0;
    if (commit)
    {
        rc = commit_or_rollback(__func__, NULL);
    }
    else
    {
        rc = (sqlite3_exec(g_db, "ROLLBACK;", NULL, NULL, NULL) == SQLITE_OK) ? 0 : -1;
    }
    g_fts_in_batch = 0;
    pthread_mutex_unlock(&g_lock);
    return rc;
}

int qml_sqlite_fts_add(const char * session_id,
                                  int64_t turn_pos, int64_t turn_end,
                                  const char * content)
{
    if (g_db == NULL || session_id == NULL || content == NULL)
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    char tag[17];
    qml_fts_tag(session_id, tag);
    if (g_fts_ins_stmt == NULL)      // [FTS-BULK-FIX] prepare only the first time
    {
        if (sqlite3_prepare_v2(g_db,
                "INSERT INTO qml_archive_fts (content, tag, session_id, turn_pos, turn_end)"
                " VALUES (?,?,?,?,?)", -1, &g_fts_ins_stmt, NULL) != SQLITE_OK)
        {
            g_fts_ins_stmt = NULL;
            pthread_mutex_unlock(&g_lock);
            return -1;
        }
    }
    sqlite3_reset(g_fts_ins_stmt);
    sqlite3_clear_bindings(g_fts_ins_stmt);
    sqlite3_bind_text (g_fts_ins_stmt, 1, content, -1, SQLITE_STATIC);
    sqlite3_bind_text (g_fts_ins_stmt, 2, tag, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text (g_fts_ins_stmt, 3, session_id, -1, SQLITE_STATIC);
    sqlite3_bind_int64(g_fts_ins_stmt, 4, turn_pos);
    sqlite3_bind_int64(g_fts_ins_stmt, 5, turn_end);
    const int rc = sqlite3_step(g_fts_ins_stmt);
    sqlite3_reset(g_fts_ins_stmt);
    pthread_mutex_unlock(&g_lock);
    return (rc == SQLITE_DONE) ? 0 : -1;
}

//---------------------------------------------------------------------------
//Func : qml_sqlite_cond_put
//---------------------------------------------------------------------------
static int vec_table_ready(void);   // defined later -- uses the same table

int qml_sqlite_cond_put(const char * session_id, int64_t turn_pos,
                                int64_t turn_end, const unsigned char * cond_code,
                                int n_code, int n_cond)
{
    if (g_db == NULL || session_id == NULL || cond_code == NULL || n_code <= 0)
    {
        return -1;
    }
    // [DEFECT FIX 2026-09-01] The qml_archive_vec table was created only on the embedding
    // path -- in a deployment that uses only condition search without embeddings (QML_EMBED_MODEL
    // not set) the table did not exist and this INSERT failed silently (measured: cond_hits
    // 0, the whole test fell back to lexical search only). Condition search must work independently
    // of embeddings, so the table is prepared here as well.
    if (!vec_table_ready())
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    sqlite3_stmt * st = NULL;
    // If a vector row already exists, fill only the condition columns; otherwise create a row with only conditions
    // (a deployment that uses only condition search without embeddings must also work).
    const char * sql =
        "INSERT INTO qml_archive_vec (session_id, turn_pos, turn_end, vec,"
        " cond_code, cond_n) VALUES (?,?,?,x'',?,?)"
        " ON CONFLICT(session_id, turn_pos) DO UPDATE SET"
        " cond_code=excluded.cond_code, cond_n=excluded.cond_n";
    if (sqlite3_prepare_v2(g_db, sql, -1, &st, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_bind_text(st, 1, session_id, -1, SQLITE_STATIC);
    sqlite3_bind_int64(st, 2, turn_pos);
    sqlite3_bind_int64(st, 3, turn_end);
    sqlite3_bind_blob(st, 4, cond_code, n_code, SQLITE_STATIC);
    sqlite3_bind_int(st, 5, n_cond);
    const int rc = sqlite3_step(st);
    sqlite3_finalize(st);
    pthread_mutex_unlock(&g_lock);
    return (rc == SQLITE_DONE) ? 0 : -1;
}

//---------------------------------------------------------------------------
//Func : qml_sqlite_cond_search
//see header for the measured rationale (unit test: 100% top-1 versus 46~58.7%
//for lexical and 41.3% for embedding). Ordering is cond_n descending only --
//adding any other tie-break key drops top-1 to 90.5% (measured).
//---------------------------------------------------------------------------
int qml_sqlite_cond_search(const char * session_id,
                                   const unsigned char * case_code, int n_code,
                                   int k, int64_t * out_turn_pos, int64_t * out_turn_end,
                                   char out_sid[][256])
{
    if (g_db == NULL || session_id == NULL || case_code == NULL || n_code <= 0 ||
        k <= 0 || out_turn_pos == NULL || out_turn_end == NULL)
    {
        return -1;
    }
    if (!vec_table_ready())
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    sqlite3_stmt * st = NULL;
    // [SCOPE FIX 2026-08-31] Reference material is loaded into a session different from the decision session
    // (reference and conversation are separated), so searching only this session almost never triggers a decision.
    // The scope is **sessions whose id has this session's id as a prefix**
    // ('rule::procurement' -> 'rule::procurement%', which includes this session and
    //  'rule::procurement_ref').
    // At first the part before '::' was treated as the owner and the scope was 'rule::%', but then
    // the role distinction was lost and 6 procurement decisions took rules from the legal reference session
    // (measured). The same case gets a different decision per role, so that recall leads to wrong
    // answers. Caution: session names where one prefix contains another (e.g. 'a' and 'ab') are mixed in this
    // scope, so a deployment must not use such names.
    char owner[512];
    int use_owner = 0;
    {
        const size_t n = strlen(session_id);
        if (n > 0 && n < sizeof(owner) - 2)
        {
            memcpy(owner, session_id, n);
            owner[n]     = '%';
            owner[n + 1] = '\0';
            use_owner    = 1;
        }
    }
    const char * sql = use_owner
        ? "SELECT turn_pos, turn_end, session_id FROM qml_archive_vec"
          " WHERE session_id LIKE ? AND cond_code IS NOT NULL"
          "   AND qml_cond_subset(cond_code, ?)=1"
          " ORDER BY cond_n DESC, rowid ASC LIMIT ?"
        : "SELECT turn_pos, turn_end, session_id FROM qml_archive_vec"
          " WHERE session_id=? AND cond_code IS NOT NULL"
          "   AND qml_cond_subset(cond_code, ?)=1"
          " ORDER BY cond_n DESC, rowid ASC LIMIT ?";
    if (sqlite3_prepare_v2(g_db, sql, -1, &st, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_bind_text(st, 1, use_owner ? owner : session_id, -1, SQLITE_TRANSIENT);
    sqlite3_bind_blob(st, 2, case_code, n_code, SQLITE_STATIC);
    sqlite3_bind_int(st, 3, k);
    int n = 0;
    while (n < k && sqlite3_step(st) == SQLITE_ROW)
    {
        out_turn_pos[n] = sqlite3_column_int64(st, 0);
        out_turn_end[n] = sqlite3_column_int64(st, 1);
        if (out_sid != NULL)
        {
            const unsigned char * sd = sqlite3_column_text(st, 2);
            snprintf(out_sid[n], 256, "%s", sd != NULL ? (const char *) sd : "");
        }
        n++;
    }
    sqlite3_finalize(st);
    pthread_mutex_unlock(&g_lock);
    return n;
}

int qml_sqlite_fts_search_text(const char * session_id,
                                      const char * match_query,
                                      int k, char ** out_txt,
                                      int64_t * out_pos)
{
    if (g_db == NULL || session_id == NULL || match_query == NULL ||
        k <= 0 || out_txt == NULL)
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    sqlite3_stmt * stmt = NULL;
    // MATCH narrows the candidates with the inverted index, then filters by session_id equality —
    // the same form as the existing fts_search (not a full scan). It also returns content so that
    // the caller does not re-query per document (a scan each time).
    if (sqlite3_prepare_v2(g_db,
            "SELECT content, turn_pos FROM qml_archive_fts"
            " WHERE qml_archive_fts MATCH ? AND session_id=?"
            " ORDER BY rank LIMIT ?",
            -1, &stmt, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_bind_text (stmt, 1, match_query, -1, SQLITE_STATIC);
    sqlite3_bind_text (stmt, 2, session_id, -1, SQLITE_STATIC);
    sqlite3_bind_int  (stmt, 3, k);
    int n = 0;
    while (n < k && sqlite3_step(stmt) == SQLITE_ROW)
    {
        const unsigned char * txt = sqlite3_column_text(stmt, 0);
        char * dup = (txt != NULL) ? strdup((const char *) txt) : NULL;
        if (dup == NULL)
        {
            continue;
        }
        if (out_pos != NULL)
        {
            out_pos[n] = sqlite3_column_int64(stmt, 1);
        }
        out_txt[n++] = dup;
    }
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);
    return n;
}

int qml_sqlite_fts_search(const char * session_id, const char * match_query,
                                  int k, int64_t * out_turn_pos, int64_t * out_turn_end,
                                  double * out_score)
{
    if (g_db == NULL || session_id == NULL || match_query == NULL ||
        k <= 0 || out_turn_pos == NULL || out_turn_end == NULL)
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    // [v1.1] index-level session scoping: 'tag:<hex> AND (words)' restricts the
    // MATCH itself to this session's rows; the session_id equality stays as a
    // correctness backstop against tag collisions.
    char tag[17];
    qml_fts_tag(session_id, tag);
    char scoped[4096];
    snprintf(scoped, sizeof(scoped), "tag:%s AND (%s)", tag, match_query);
    sqlite3_stmt * stmt = NULL;
    if (sqlite3_prepare_v2(g_db,
            // [RIG-GATE 2026-08-14] Also reads rank (=BM25; FTS5 returns it with the sign flipped).
            // The RIG decision part uses it to compute the sim value.
            // To revert: remove rank from the SELECT list and delete the out_score handling.
            "SELECT turn_pos, turn_end, rank FROM qml_archive_fts"
            " WHERE qml_archive_fts MATCH ? AND session_id=?"
            " ORDER BY rank LIMIT ?",
            -1, &stmt, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_bind_text(stmt, 1, scoped, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, session_id, -1, SQLITE_STATIC);
    sqlite3_bind_int (stmt, 3, k);
    int n = 0;
    while (n < k && sqlite3_step(stmt) == SQLITE_ROW)
    {
        out_turn_pos[n] = sqlite3_column_int64(stmt, 0);
        out_turn_end[n] = sqlite3_column_int64(stmt, 1);
        if (out_score != NULL)          // [RIG-GATE] BM25 rank (negative, smaller is higher ranked)
        {
            out_score[n] = sqlite3_column_double(stmt, 2);
        }
        n++;
    }
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);
    return n;
}

// [HYBRID 2026-08-27] Storage and search of archive turn embedding vectors — the storage half of
// semantic (meaning-based) search. The table is created lazily. A vector is a float32 array BLOB
// (1024 dimensions for bge-m3 = 4KB/turn). Search is a session-scoped brute-force cosine —
// measured (TEST-LEDGER A-2 item 22): tens of ms per query even at 19,446 documents, so a separate
// vector DB process is unnecessary (keeps the embedded principle).
// [COND-SUBSET 2026-08-31] SQLite user-defined function
// qml_cond_subset(rule_code, case_code) — 1 if all condition bits of the rule are contained in the
// case bits, otherwise 0. That is, it decides (rule AND case) == rule.
//
// Why it is needed: in reference material whose conditions are structured as `field=value` (approval rules, policy
// gates), the correct material is defined as "the one whose conditions are all satisfied". Lexical
// search and embeddings convert partial matches into scores, so material that overlaps on only one condition
// can rank above material that overlaps on both.
// Measured in a unit test (tests/unit_rule_search.py, 63 decisions, 42~159 rules):
//   BM25 on full request text      top-1 46.0%  top-3 52.4%
//   BM25 on condition parameters   top-1 58.7%  top-3 88.9%
//   embedding only (bge-m3)        top-1 41.3%  top-3 66.7%
//   hybrid                         top-1 58.7%  top-3 90.5%
//   condition-bit decision (this)  top-1 100%   top-3 100%
// If BM25 first narrows to top-5 and then the bit decision runs, it drops to 88.9% -- lexical
// search drops the correct answer out of the candidates, so it is not used as a pre-filter. The bit
// decision is a 43 byte (339 positions) operation, so scanning everything costs almost nothing.
// Ordering uses only the number of conditions, descending (ties keep the load order). Mixing id into
// ties brings in string order and top-1 drops to 90.5% (measured).
// Recall passes **only the top-1** (confirmed by the author 2026-08-31): when several candidates are given,
// the model fails to pick among them and accuracy drops -- 1 rule given 69.6%,
// 3 rules 65.2%, 5 rules 65.2% (single-agent baseline measurement). This decision's top-1 is 100%,
// so there is no reason to add candidates either.
static void cond_subset_udf(sqlite3_context * ctx, int argc, sqlite3_value ** argv)
{
    if (argc != 2 ||
        sqlite3_value_type(argv[0]) != SQLITE_BLOB ||
        sqlite3_value_type(argv[1]) != SQLITE_BLOB)
    {
        sqlite3_result_int(ctx, 0);
        return;
    }
    const unsigned char * r = (const unsigned char *) sqlite3_value_blob(argv[0]);
    const unsigned char * c = (const unsigned char *) sqlite3_value_blob(argv[1]);
    const int nr = sqlite3_value_bytes(argv[0]);
    const int nc = sqlite3_value_bytes(argv[1]);
    if (nr != nc || nr <= 0)
    {
        sqlite3_result_int(ctx, 0);   // do not decide if the position schemes differ
        return;
    }
    for (int i = 0; i < nr; i++)
    {
        if ((r[i] & c[i]) != r[i])
        {
            sqlite3_result_int(ctx, 0);
            return;
        }
    }
    sqlite3_result_int(ctx, 1);
}

// [BINCODE 2026-08-29] SQLite user-defined function qml_hamming(a,b) — Hamming distance
// (number of differing bits) of two BLOBs. Registered so it can be used directly in SQL ORDER BY.
// popcount is done with the compiler builtin (__builtin_popcount).
static void hamming_udf(sqlite3_context * ctx, int argc, sqlite3_value ** argv)
{
    if (argc != 2 ||
        sqlite3_value_type(argv[0]) != SQLITE_BLOB ||
        sqlite3_value_type(argv[1]) != SQLITE_BLOB)
    {
        sqlite3_result_int(ctx, 1 << 20);   // not comparable — use the maximum distance
        return;
    }
    const unsigned char * a = (const unsigned char *) sqlite3_value_blob(argv[0]);
    const unsigned char * b = (const unsigned char *) sqlite3_value_blob(argv[1]);
    int na = sqlite3_value_bytes(argv[0]);
    const int nb = sqlite3_value_bytes(argv[1]);
    if (na != nb)
    {
        sqlite3_result_int(ctx, 1 << 20);   // length mismatch (different dimension) — excluded
        return;
    }
    int d = 0;
    for (int i = 0; i < na; i++)
    {
        d += __builtin_popcount((unsigned) (a[i] ^ b[i]));
    }
    sqlite3_result_int(ctx, d);
}

static int vec_table_ready(void)
{
    static int ready = 0;
    if (!ready)
    {
        if (sqlite3_exec(g_db,
                "CREATE TABLE IF NOT EXISTS qml_archive_vec ("
                " session_id TEXT NOT NULL, turn_pos INTEGER NOT NULL,"
                " turn_end INTEGER NOT NULL, vec BLOB NOT NULL,"
                " PRIMARY KEY (session_id, turn_pos))",
                NULL, NULL, NULL) != SQLITE_OK)
        {
            return 0;
        }
        // [BINCODE 2026-08-29] (b) binary sign column — compresses the sign of each vector dimension
        // into 1 bit (1024 dimensions -> 128 bytes, 1/32 of the original 4KB).
        // The goal is read cost, not accuracy: the bottleneck of similarity search is not computation
        // but reading all BLOBs (4.2 s at 500,000 rows measured, QML OS measurement).
        // A pre-filter narrows the candidates, then the original vectors decide precisely.
        // NULL for existing rows — in that case the pre-filter is skipped and the previous path is used.
        sqlite3_exec(g_db, "ALTER TABLE qml_archive_vec ADD COLUMN code BLOB",
                     NULL, NULL, NULL);   // ignore the error if it already exists (convention)
        // [COND-SUBSET 2026-08-31] condition bit column — a bit vector encoding the conditions of reference
        // material as `field=value` positions (339 positions = 43 byte for this data).
        // Unlike the code column it is used for an exact decision, not an approximation: if the stored bits are
        // all contained in the query bits, the row is a candidate. Material without conditions is left NULL and
        // excluded from condition search (the previous lexical and embedding paths are not affected).
        sqlite3_exec(g_db, "ALTER TABLE qml_archive_vec ADD COLUMN cond_code BLOB",
                     NULL, NULL, NULL);
        sqlite3_exec(g_db, "ALTER TABLE qml_archive_vec ADD COLUMN cond_n INTEGER",
                     NULL, NULL, NULL);
        sqlite3_create_function(g_db, "qml_cond_subset", 2,
                                SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                NULL, cond_subset_udf, NULL, NULL);
        sqlite3_create_function(g_db, "qml_hamming", 2, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                NULL, hamming_udf, NULL, NULL);
        ready = 1;
    }
    return 1;
}

// =====================================================================
// [ENTITY-INDEX 2026-08-29] (a) entity field index — stores identifiers extracted from documents by
// deterministic rules (pattern id, crew code, etc.) in a separate table, and when the query has the same
// identifier, narrows the candidates by condition match without rank competition.
// Reason: as similar documents accumulate, the one identifier needed to tell them apart gets buried in
// common words and pushed out of the ranking (large-scale recurrence recall 3/53, ledger row 34-2).
// A point lookup does not depend on table size (2.1ms measured at 3.15 million rows, ledger section A-4).
// =====================================================================
static int ent_table_ready(void)
{
    static int ready = 0;
    if (!ready)
    {
        if (sqlite3_exec(g_db,
                "CREATE TABLE IF NOT EXISTS qml_archive_entity ("
                " session_id TEXT NOT NULL, entity TEXT NOT NULL,"
                " turn_pos INTEGER NOT NULL, turn_end INTEGER NOT NULL,"
                " PRIMARY KEY (session_id, entity, turn_pos))",
                NULL, NULL, NULL) != SQLITE_OK)
        {
            return 0;
        }
        sqlite3_exec(g_db,
            "CREATE INDEX IF NOT EXISTS idx_qml_archive_entity_lookup"
            " ON qml_archive_entity(session_id, entity)", NULL, NULL, NULL);
        ready = 1;
    }
    return 1;
}

int qml_sqlite_entity_add(const char * session_id, const char * entity,
                                  int64_t turn_pos, int64_t turn_end)
{
    if (g_db == NULL || session_id == NULL || entity == NULL || entity[0] == 0)
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    if (!ent_table_ready())
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_stmt * stmt = NULL;
    if (sqlite3_prepare_v2(g_db,
            "INSERT OR REPLACE INTO qml_archive_entity"
            " (session_id, entity, turn_pos, turn_end) VALUES (?,?,?,?)",
            -1, &stmt, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_bind_text (stmt, 1, session_id, -1, SQLITE_STATIC);
    sqlite3_bind_text (stmt, 2, entity, -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 3, turn_pos);
    sqlite3_bind_int64(stmt, 4, turn_end);
    const int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);
    return (rc == SQLITE_DONE) ? 0 : -1;
}

// [ENTITY-PEER 2026-08-29] Also queries other sessions of the same owner.
// The session_id convention is "owner::name", so only sessions with the same owner prefix are
// targeted — sessions of a different owner are never crossed in any case (keeps the paper's identity
// isolation guarantee). Use: when the answer exists only in another agent's observations
// (e.g. analyst queries monitor's past observations) — the fix for cascading-failure 0/36.
//  in: owner_prefix — owner prefix that includes the separator at the end, such as "agent::"
// out: number of recalled rows (in latest turn_pos order) or -1. out_sid is the session id of each result
//      (returned together because the caller must read tokens from that session's archive).
int qml_sqlite_entity_find_peers(const char * owner_prefix,
                                          const char * entity, int max_hits,
                                          char out_sid[][128], int64_t * out_pos,
                                          int64_t * out_end)
{
    if (g_db == NULL || owner_prefix == NULL || entity == NULL ||
        out_sid == NULL || out_pos == NULL || out_end == NULL || max_hits <= 0)
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    if (!ent_table_ready())
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_stmt * stmt = NULL;
    if (sqlite3_prepare_v2(g_db,
            "SELECT session_id, turn_pos, turn_end FROM qml_archive_entity"
            " WHERE entity=? AND session_id LIKE ? ORDER BY turn_pos DESC LIMIT ?",
            -1, &stmt, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    char like[160];
    snprintf(like, sizeof(like), "%s%%", owner_prefix);
    sqlite3_bind_text(stmt, 1, entity, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, like, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int (stmt, 3, max_hits);
    int n = 0;
    while (n < max_hits && sqlite3_step(stmt) == SQLITE_ROW)
    {
        const char * sid = (const char *) sqlite3_column_text(stmt, 0);
        if (sid == NULL)
        {
            continue;
        }
        snprintf(out_sid[n], 128, "%s", sid);
        out_pos[n] = sqlite3_column_int64(stmt, 1);
        out_end[n] = sqlite3_column_int64(stmt, 2);
        n++;
    }
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);
    return n;
}

int qml_sqlite_entity_find(const char * session_id, const char * entity,
                                     int max_hits, int64_t * out_pos,
                                     int64_t * out_end)
{
    if (g_db == NULL || session_id == NULL || entity == NULL ||
        out_pos == NULL || out_end == NULL || max_hits <= 0)
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    if (!ent_table_ready())
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_stmt * stmt = NULL;
    if (sqlite3_prepare_v2(g_db,
            "SELECT turn_pos, turn_end FROM qml_archive_entity"
            " WHERE session_id=? AND entity=? ORDER BY turn_pos DESC LIMIT ?",
            -1, &stmt, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, entity, -1, SQLITE_STATIC);
    sqlite3_bind_int (stmt, 3, max_hits);
    int n = 0;
    while (n < max_hits && sqlite3_step(stmt) == SQLITE_ROW)
    {
        out_pos[n] = sqlite3_column_int64(stmt, 0);
        out_end[n] = sqlite3_column_int64(stmt, 1);
        n++;
    }
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);
    return n;
}

int qml_sqlite_vec_add(const char * session_id, int64_t turn_pos,
                               int64_t turn_end, const float * vec, size_t dim)
{
    if (g_db == NULL || session_id == NULL || vec == NULL || dim == 0)
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    if (!vec_table_ready())
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_stmt * stmt = NULL;
    if (sqlite3_prepare_v2(g_db,
            "INSERT OR REPLACE INTO qml_archive_vec"
            " (session_id, turn_pos, turn_end, vec, code) VALUES (?,?,?,?,?)",
            -1, &stmt, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    // [BINCODE] sign bit compression — dim bits into a byte array
    unsigned char code[256];
    const size_t n_code = (dim + 7) / 8 <= sizeof(code) ? (dim + 7) / 8 : sizeof(code);
    memset(code, 0, n_code);
    for (size_t i = 0; i < dim && (i / 8) < n_code; i++)
    {
        if (vec[i] > 0.0f)
        {
            code[i / 8] |= (unsigned char) (1u << (i % 8));
        }
    }
    sqlite3_bind_text (stmt, 1, session_id, -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 2, turn_pos);
    sqlite3_bind_int64(stmt, 3, turn_end);
    sqlite3_bind_blob (stmt, 4, vec, (int) (dim * sizeof(float)), SQLITE_STATIC);
    sqlite3_bind_blob (stmt, 5, code, (int) n_code, SQLITE_TRANSIENT);
    const int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);
    return (rc == SQLITE_DONE) ? 0 : -1;
}

int qml_sqlite_vec_missing(const char * session_id, int64_t max_n,
                                   int64_t * out_pos, int64_t * out_end)
{
    if (g_db == NULL || session_id == NULL || max_n <= 0 ||
        out_pos == NULL || out_end == NULL)
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    if (!vec_table_ready())
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_stmt * stmt = NULL;
    // turns that are in the FTS index but do not yet have a vector — targets of lazy embedding
    if (sqlite3_prepare_v2(g_db,
            "SELECT f.turn_pos, f.turn_end FROM qml_archive_fts f"
            " WHERE f.session_id=? AND NOT EXISTS"
            "  (SELECT 1 FROM qml_archive_vec v"
            "   WHERE v.session_id=f.session_id AND v.turn_pos=f.turn_pos)"
            " ORDER BY f.turn_pos LIMIT ?",
            -1, &stmt, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_bind_text (stmt, 1, session_id, -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 2, max_n);
    int n = 0;
    while (n < max_n && sqlite3_step(stmt) == SQLITE_ROW)
    {
        out_pos[n] = sqlite3_column_int64(stmt, 0);
        out_end[n] = sqlite3_column_int64(stmt, 1);
        n++;
    }
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);
    return n;
}

int qml_sqlite_vec_missing_text(const char * session_id, int64_t max_n,
                                       int64_t * out_pos, int64_t * out_end,
                                       char ** out_txt)
{
    if (g_db == NULL || session_id == NULL || max_n <= 0 ||
        out_pos == NULL || out_end == NULL || out_txt == NULL)
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    if (!vec_table_ready())
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_stmt * stmt = NULL;
    // The same scan as vec_missing, but reads content from the same row —
    // a form that removes per-document re-queries (each a full scan) (see the header comment)
    if (sqlite3_prepare_v2(g_db,
            "SELECT f.turn_pos, f.turn_end, f.content FROM qml_archive_fts f"
            " WHERE f.session_id=? AND NOT EXISTS"
            "  (SELECT 1 FROM qml_archive_vec v"
            "   WHERE v.session_id=f.session_id AND v.turn_pos=f.turn_pos)"
            " ORDER BY f.turn_pos LIMIT ?",
            -1, &stmt, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_bind_text (stmt, 1, session_id, -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 2, max_n);
    int n = 0;
    while (n < max_n && sqlite3_step(stmt) == SQLITE_ROW)
    {
        const unsigned char * txt = sqlite3_column_text(stmt, 2);
        char * dup = (txt != NULL) ? strdup((const char *) txt) : NULL;
        if (dup == NULL)
        {
            continue;   // skip rows without a body (there is nothing to embed)
        }
        out_pos[n] = sqlite3_column_int64(stmt, 0);
        out_end[n] = sqlite3_column_int64(stmt, 1);
        out_txt[n] = dup;
        n++;
    }
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);
    return n;
}

int qml_sqlite_vec_search(const char * session_id, const float * qvec,
                                  size_t dim, int k, int64_t * out_pos,
                                  int64_t * out_end, double * out_score)
{
    if (g_db == NULL || session_id == NULL || qvec == NULL || dim == 0 ||
        k <= 0 || out_pos == NULL || out_end == NULL)
    {
        return -1;
    }
    double qnorm = 0.0;
    for (size_t i = 0; i < dim; i++)
    {
        qnorm += (double) qvec[i] * (double) qvec[i];
    }
    qnorm = sqrt(qnorm);
    if (qnorm == 0.0)
    {
        return 0;
    }
    pthread_mutex_lock(&g_lock);
    if (!vec_table_ready())
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    // [BINCODE 2026-08-29] Pre-filter path: read only the code column and narrow the candidates by Hamming
    // distance (1/32 of the read volume), then read only those candidates' original vectors for a precise cosine decision.
    // env QML_VEC_BINCODE=0 returns to the previous path (read all vectors).
    // (C file — static initialization via a compute-once flag)
    static int use_code = -1;
    if (use_code < 0)
    {
        const char * e = getenv("QML_VEC_BINCODE");
        use_code = (e == NULL) || (atoi(e) != 0);
    }
    const char * sql_all =
        "SELECT turn_pos, turn_end, vec FROM qml_archive_vec WHERE session_id=?";
    // the number of candidates is 8x k (at least 32) — margin to absorb correct answers the sign bits missed
    char sql_code[512];
    snprintf(sql_code, sizeof(sql_code),
             "SELECT turn_pos, turn_end, vec FROM qml_archive_vec"
             " WHERE session_id=? AND code IS NOT NULL"
             " ORDER BY qml_hamming(code, ?) ASC LIMIT %d",
             (k * 8 > 32) ? k * 8 : 32);
    sqlite3_stmt * stmt = NULL;
    int used_code = 0;
    if (use_code)
    {
        unsigned char qcode[256];
        const size_t n_qc = (dim + 7) / 8 <= sizeof(qcode) ? (dim + 7) / 8 : sizeof(qcode);
        memset(qcode, 0, n_qc);
        for (size_t i = 0; i < dim && (i / 8) < n_qc; i++)
        {
            if (qvec[i] > 0.0f)
            {
                qcode[i / 8] |= (unsigned char) (1u << (i % 8));
            }
        }
        if (sqlite3_prepare_v2(g_db, sql_code, -1, &stmt, NULL) == SQLITE_OK)
        {
            sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_STATIC);
            sqlite3_bind_blob(stmt, 2, qcode, (int) n_qc, SQLITE_TRANSIENT);
            used_code = 1;
        }
        else
        {
            stmt = NULL;
        }
    }
    if (!used_code)
    {
        if (sqlite3_prepare_v2(g_db, sql_all, -1, &stmt, NULL) != SQLITE_OK)
        {
            pthread_mutex_unlock(&g_lock);
            return -1;
        }
        sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_STATIC);
    }
    // keep the top k with insertion sort (k is small — usually 3)
    int n_top = 0;
    double  top_s[16];
    int64_t top_p[16], top_e[16];
    if (k > 16)
    {
        k = 16;
    }
    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
        const size_t nb = (size_t) sqlite3_column_bytes(stmt, 2);
        if (nb != dim * sizeof(float))
        {
            continue;   // dimension mismatch (leftover row from a different embedding model) — skip
        }
        const float * v = (const float *) sqlite3_column_blob(stmt, 2);
        double dot = 0.0, vn = 0.0;
        for (size_t i = 0; i < dim; i++)
        {
            dot += (double) qvec[i] * (double) v[i];
            vn  += (double) v[i] * (double) v[i];
        }
        vn = sqrt(vn);
        const double cs = (vn > 0.0) ? dot / (qnorm * vn) : 0.0;
        int j = n_top;
        while (j > 0 && top_s[j - 1] < cs)
        {
            j--;
        }
        if (j < k)
        {
            const int last = (n_top < k) ? n_top : k - 1;
            for (int m = last; m > j; m--)
            {
                top_s[m] = top_s[m - 1];
                top_p[m] = top_p[m - 1];
                top_e[m] = top_e[m - 1];
            }
            top_s[j] = cs;
            top_p[j] = sqlite3_column_int64(stmt, 0);
            top_e[j] = sqlite3_column_int64(stmt, 1);
            if (n_top < k)
            {
                n_top++;
            }
        }
    }
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);
    for (int i = 0; i < n_top; i++)
    {
        out_pos[i] = top_p[i];
        out_end[i] = top_e[i];
        if (out_score != NULL)
        {
            out_score[i] = top_s[i];
        }
    }
    return n_top;
}

int qml_sqlite_get_tail(const char * session_id, size_t max_n,
                                  int32_t ** out_ids, size_t * out_n)
{
    if (g_db == NULL || session_id == NULL || max_n == 0 ||
        out_ids == NULL || out_n == NULL)
    {
        return -1;
    }
    *out_ids = NULL;
    *out_n   = 0;
    pthread_mutex_lock(&g_lock);
    sqlite3_stmt * stmt = NULL;
    // only the most recent — read with pos descending + LIMIT, then reverse into the caller's order (ascending)
    if (sqlite3_prepare_v2(g_db,
            "SELECT token_id FROM qml_token_log WHERE session_id=?"
            " ORDER BY pos DESC LIMIT ?", -1, &stmt, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_bind_text (stmt, 1, session_id, -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 2, (sqlite3_int64) max_n);
    int32_t * buf = (int32_t *) malloc(max_n * sizeof(int32_t));
    if (buf == NULL)
    {
        sqlite3_finalize(stmt);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    size_t n = 0;
    while (n < max_n && sqlite3_step(stmt) == SQLITE_ROW)
    {
        buf[n++] = (int32_t) sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);
    for (size_t i = 0; i < n / 2; i++)     // read with DESC, so reverse it
    {
        const int32_t t = buf[i];
        buf[i]         = buf[n - 1 - i];
        buf[n - 1 - i] = t;
    }
    *out_ids = buf;
    *out_n   = n;
    return 0;
}

char * qml_sqlite_fts_text(const char * session_id, int64_t turn_pos)
{
    if (g_db == NULL || session_id == NULL)
    {
        return NULL;
    }
    pthread_mutex_lock(&g_lock);
    sqlite3_stmt * stmt = NULL;
    if (sqlite3_prepare_v2(g_db,
            "SELECT content FROM qml_archive_fts"
            " WHERE session_id=? AND turn_pos=? LIMIT 1",
            -1, &stmt, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return NULL;
    }
    sqlite3_bind_text (stmt, 1, session_id, -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 2, turn_pos);
    char * out = NULL;
    if (sqlite3_step(stmt) == SQLITE_ROW)
    {
        const unsigned char * txt = sqlite3_column_text(stmt, 0);
        if (txt != NULL)
        {
            out = strdup((const char *) txt);
        }
    }
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);
    return out;
}

int qml_sqlite_archive_range(const char * session_id,
                             int64_t pos_start, int64_t pos_end,
                             int32_t ** out_tok, size_t * out_n)
{
    if (g_db == NULL || session_id == NULL || out_tok == NULL || out_n == NULL ||
        pos_end <= pos_start)
    {
        return -1;
    }
    *out_tok = NULL;
    *out_n   = 0;
    pthread_mutex_lock(&g_lock);
    sqlite3_stmt * stmt = NULL;
    if (sqlite3_prepare_v2(g_db,
            "SELECT token_id FROM qml_token_log_archive"
            " WHERE session_id=? AND orig_pos>=? AND orig_pos<? ORDER BY orig_pos",
            -1, &stmt, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_bind_text (stmt, 1, session_id, -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 2, pos_start);
    sqlite3_bind_int64(stmt, 3, pos_end);
    size_t cap = (size_t)(pos_end - pos_start);
    int32_t * tok = (int32_t *) malloc(cap * sizeof(int32_t));
    size_t n = 0;
    while (tok != NULL && n < cap && sqlite3_step(stmt) == SQLITE_ROW)
    {
        tok[n++] = (int32_t) sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);
    if (tok == NULL || n == 0)
    {
        free(tok);
        return -1;
    }
    *out_tok = tok;
    *out_n   = n;
    return 0;
}
//===========================================================================//
// [FTS-RECALL] END
//===========================================================================//

//===========================================================================//
// [TAWT-CORE 2026-08-23] window-trim storage half. The POLICY (where to cut)
// lives in core/qml_window.c; this function is the single-transaction record
// surgery it delegates to. Mirrors the earlier Python gateway implementation that
// carried every TAWT measurement, including
// its two recorded defect fixes: monotonic orig_pos across repeated trims
// (2026-08-13) and read+write inside ONE write transaction so a concurrent
// persister append can never fall between them (TAWT-RACE-FIX 2026-08-15;
// here the BEGIN IMMEDIATE plus in-transaction re-read plays that role).
//===========================================================================//

//------------------------------------------------------------
//Func : qml_sqlite_trim_apply
//moves qml_token_log rows [head_len, cut) of a session to
//qml_token_log_archive (orig_pos continues monotonically), renumbers the
//remaining rows from 0, and updates the qml_session counters.
//The caller computed head_len/cut from a snapshot of the log; both are
//re-validated INSIDE the transaction (the row at each boundary must still
//end with eot_token), so a log that changed shape in between makes this a
//safe no-op instead of a corruption.
//  in: session_id — session to shrink
//  in: head_len — rows [0, head_len) are the exempt system prefix (0 = none)
//  in: cut — first row of the kept tail; rows [head_len, cut) are archived
//  in: eot_token — end-of-turn token id used to re-validate both boundaries
// out: number of tokens archived; 0 = validation failed (no-op); -1 = error
// side effects: one IMMEDIATE transaction over qml_token_log /
//     qml_token_log_archive / qml_session
//------------------------------------------------------------
//------------------------------------------------------------
//Func : qml_sqlite_trim_head_get / _set
//Per-session stored value of the head (system prefix) exemption decision. -1 = not yet decided.
//The decision is made only once at the first trim, and later trims use the stored value as is
//(the exempt range always stays at the front of the log, so it stays valid regardless of renumbering).
//------------------------------------------------------------
long qml_sqlite_trim_head_get(const char * session_id)
{
    long v = -1;
    sqlite3_stmt * stmt = NULL;
    if (g_db == NULL || session_id == NULL) { return -1; }
    pthread_mutex_lock(&g_lock);
    if (sqlite3_prepare_v2(g_db,
            "SELECT trim_head_len FROM qml_session WHERE session_id=?",
            -1, &stmt, NULL) == SQLITE_OK)
    {
        sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW)
        {
            v = (long) sqlite3_column_int64(stmt, 0);
        }
    }
    if (stmt != NULL) { sqlite3_finalize(stmt); }
    pthread_mutex_unlock(&g_lock);
    return v;
}

int qml_sqlite_trim_head_set(const char * session_id, long head_len)
{
    int rc = -1;
    sqlite3_stmt * stmt = NULL;
    if (g_db == NULL || session_id == NULL) { return -1; }
    pthread_mutex_lock(&g_lock);
    if (sqlite3_prepare_v2(g_db,
            "UPDATE qml_session SET trim_head_len=? WHERE session_id=?",
            -1, &stmt, NULL) == SQLITE_OK)
    {
        sqlite3_bind_int64(stmt, 1, (sqlite3_int64) head_len);
        sqlite3_bind_text (stmt, 2, session_id, -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_DONE) { rc = 0; }
    }
    if (stmt != NULL) { sqlite3_finalize(stmt); }
    pthread_mutex_unlock(&g_lock);
    return rc;
}

long qml_sqlite_trim_apply(const char * session_id, size_t head_len,
                           size_t cut, int32_t eot_token)
{
    if (g_db == NULL || session_id == NULL || cut <= head_len)
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    // ensure the archive table exists; the call's FTS half may report -1 on
    // an SQLite without FTS5, which must NOT block a trim (archive is plain)
    (void) qml_fts_init();
    if (sqlite3_exec(g_db, "BEGIN IMMEDIATE", NULL, NULL, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    long           archived = -1;
    sqlite3_stmt * stmt     = NULL;
    qml_token_rec_t * rows  = NULL;
    size_t n = 0, cap = 0;
    if (sqlite3_prepare_v2(g_db,
            "SELECT pos, seq_id, token_id FROM qml_token_log"
            " WHERE session_id=? ORDER BY pos", -1, &stmt, NULL) != SQLITE_OK)
    {
        goto rollback;
    }
    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_STATIC);
    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
        if (n == cap)
        {
            cap = cap ? cap * 2 : 4096;
            qml_token_rec_t * nr = (qml_token_rec_t *) realloc(rows, cap * sizeof(*rows));
            if (nr == NULL)
            {
                sqlite3_finalize(stmt);
                goto rollback;
            }
            rows = nr;
        }
        rows[n].pos      = sqlite3_column_int(stmt, 0);
        rows[n].seq_id   = sqlite3_column_int(stmt, 1);
        rows[n].token_id = sqlite3_column_int(stmt, 2);
        n++;
    }
    sqlite3_finalize(stmt);
    stmt = NULL;
    // in-transaction re-validation: both boundaries must still be turn ends
    if (cut >= n ||
        rows[cut - 1].token_id != eot_token ||
        (head_len > 0 && rows[head_len - 1].token_id != eot_token))
    {
        archived = 0;   // log changed shape since the caller looked -- no-op
        goto rollback;
    }
    {
        int64_t base = 0;
        if (sqlite3_prepare_v2(g_db,
                "SELECT COALESCE(MAX(orig_pos), -1) + 1 FROM qml_token_log_archive"
                " WHERE session_id=?", -1, &stmt, NULL) != SQLITE_OK)
        {
            goto rollback;
        }
        sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW)
        {
            base = sqlite3_column_int64(stmt, 0);
        }
        sqlite3_finalize(stmt);
        stmt = NULL;
        if (sqlite3_prepare_v2(g_db,
                "INSERT INTO qml_token_log_archive VALUES"
                " (?,?,?,?,strftime('%s','now'))", -1, &stmt, NULL) != SQLITE_OK)
        {
            goto rollback;
        }
        for (size_t i = head_len; i < cut; i++)
        {
            sqlite3_reset(stmt);
            sqlite3_bind_text (stmt, 1, session_id, -1, SQLITE_STATIC);
            sqlite3_bind_int64(stmt, 2, base + (int64_t)(i - head_len));
            sqlite3_bind_int  (stmt, 3, rows[i].seq_id);
            sqlite3_bind_int  (stmt, 4, rows[i].token_id);
            if (sqlite3_step(stmt) != SQLITE_DONE)
            {
                goto rollback;
            }
        }
        sqlite3_finalize(stmt);
        stmt = NULL;
    }
    {
        sqlite3_stmt * del = NULL;
        if (sqlite3_prepare_v2(g_db,
                "DELETE FROM qml_token_log WHERE session_id=?",
                -1, &del, NULL) != SQLITE_OK)
        {
            goto rollback;
        }
        sqlite3_bind_text(del, 1, session_id, -1, SQLITE_STATIC);
        const int rc_del = sqlite3_step(del);
        sqlite3_finalize(del);
        if (rc_del != SQLITE_DONE)
        {
            goto rollback;
        }
    }
    {
        if (sqlite3_prepare_v2(g_db,
                "INSERT INTO qml_token_log (session_id, pos, seq_id, token_id)"
                " VALUES (?,?,?,?)", -1, &stmt, NULL) != SQLITE_OK)
        {
            goto rollback;
        }
        int32_t new_pos    = 0;
        int32_t last_token = 0;
        for (size_t i = 0; i < n; i++)
        {
            if (i >= head_len && i < cut)
            {
                continue;   // archived range
            }
            sqlite3_reset(stmt);
            sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_STATIC);
            sqlite3_bind_int (stmt, 2, new_pos);
            sqlite3_bind_int (stmt, 3, rows[i].seq_id);
            sqlite3_bind_int (stmt, 4, rows[i].token_id);
            if (sqlite3_step(stmt) != SQLITE_DONE)
            {
                goto rollback;
            }
            last_token = rows[i].token_id;
            new_pos++;
        }
        sqlite3_finalize(stmt);
        stmt = NULL;
        sqlite3_stmt * upd = NULL;
        if (sqlite3_prepare_v2(g_db,
                "UPDATE qml_session SET n_tokens=?, last_pos=?, last_token=?,"
                " updated_at=strftime('%s','now') WHERE session_id=?",
                -1, &upd, NULL) != SQLITE_OK)
        {
            goto rollback;
        }
        sqlite3_bind_int (upd, 1, new_pos);
        sqlite3_bind_int (upd, 2, new_pos - 1);
        sqlite3_bind_int (upd, 3, last_token);
        sqlite3_bind_text(upd, 4, session_id, -1, SQLITE_STATIC);
        const int rc_upd = sqlite3_step(upd);
        sqlite3_finalize(upd);
        if (rc_upd != SQLITE_DONE)
        {
            goto rollback;
        }
    }
    if (sqlite3_exec(g_db, "COMMIT", NULL, NULL, NULL) != SQLITE_OK)
    {
        goto rollback;
    }
    free(rows);
    pthread_mutex_unlock(&g_lock);
    return (long)(cut - head_len);

rollback:
    if (stmt != NULL)
    {
        sqlite3_finalize(stmt);
    }
    sqlite3_exec(g_db, "ROLLBACK", NULL, NULL, NULL);
    free(rows);
    pthread_mutex_unlock(&g_lock);
    return archived;
}

//------------------------------------------------------------
//Func : qml_sqlite_fts_drop
//drops a session's rows from the FTS index so the next recall lazily
//rebuilds it over the (now larger) archive. Called after a trim adds rows
//to an archive whose index was already built -- the lazy indexer builds
//once per session and would otherwise never see them (TEST-LEDGER B-2).
//  in: session_id — session whose FTS documents are removed
// out: 0 on success (including "nothing to delete"), -1 on error
//------------------------------------------------------------
int qml_sqlite_fts_drop(const char * session_id)
{
    if (g_db == NULL || session_id == NULL)
    {
        return -1;
    }
    pthread_mutex_lock(&g_lock);
    if (qml_fts_init() != 0)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;   // no FTS5 -> no index exists -> nothing to do, but say so
    }
    sqlite3_stmt * stmt = NULL;
    if (sqlite3_prepare_v2(g_db,
            "DELETE FROM qml_archive_fts WHERE session_id=?",
            -1, &stmt, NULL) != SQLITE_OK)
    {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    sqlite3_bind_text(stmt, 1, session_id, -1, SQLITE_STATIC);
    const int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&g_lock);
    return rc == SQLITE_DONE ? 0 : -1;
}
