// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
// storage/lmdb/qml_lmdb_store.c

#include "qml_lmdb_store.h"

#include <lmdb.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

struct qml_lmdb_txn
{
    MDB_txn * txn;
};

static MDB_env       * g_env  = NULL;
static MDB_dbi          g_dbi;
static pthread_mutex_t  g_lock = PTHREAD_MUTEX_INITIALIZER;

//------------------------------------------------------------
//Func : qml_lmdb_open
//------------------------------------------------------------
// open (or create) the process-wide LMDB environment; idempotent under g_lock
//  in: lmdb_path — directory for the LMDB data/lock files (mkdir'd best-effort)
//  in: map_size_bytes — LMDB env mapsize (maximum size of the memory-mapped region)
// out: 0 on success (including when already open), -1 on failure
// side effects: creates the directory and LMDB data/lock files on first open
int qml_lmdb_open(const char * lmdb_path, size_t map_size_bytes)
{
    pthread_mutex_lock(&g_lock);

    if (g_env != NULL)
    {
        pthread_mutex_unlock(&g_lock);
        return 0;
    }

    mkdir(lmdb_path, 0755); // best-effort; ignore EEXIST

    int rc = mdb_env_create(&g_env);
    if (rc != 0)
    {
        g_env = NULL;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    rc = mdb_env_set_mapsize(g_env, map_size_bytes);
    if (rc != 0)
    {
        mdb_env_close(g_env);
        g_env = NULL;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    // MDB_WRITEMAP: real mmap'd writable pages — reads hand back a
    // pointer directly into the mapping, no copy required
    rc = mdb_env_open(g_env, lmdb_path, MDB_WRITEMAP, 0664);
    if (rc != 0)
    {
        mdb_env_close(g_env);
        g_env = NULL;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    MDB_txn * txn = NULL;
    rc = mdb_txn_begin(g_env, NULL, 0, &txn);
    if (rc != 0)
    {
        mdb_env_close(g_env);
        g_env = NULL;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    rc = mdb_dbi_open(txn, NULL, 0, &g_dbi);
    if (rc != 0)
    {
        mdb_txn_abort(txn);
        mdb_env_close(g_env);
        g_env = NULL;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    mdb_txn_commit(txn);

    pthread_mutex_unlock(&g_lock);
    return 0;
}

//------------------------------------------------------------
//Func : qml_lmdb_close
//------------------------------------------------------------
// close the global LMDB environment if open (safe to call when already closed)
//  in: none
// out: none (void)
void qml_lmdb_close(void)
{
    pthread_mutex_lock(&g_lock);

    if (g_env != NULL)
    {
        mdb_env_close(g_env);
        g_env = NULL;
    }

    pthread_mutex_unlock(&g_lock);
}

//------------------------------------------------------------
//Func : qml_lmdb_put
//------------------------------------------------------------
// store/overwrite one value under key in a single write txn, committed before return
//  in: key — NUL-terminated key string
//  in: data — value bytes to store
//  in: size — value length in bytes
// out: 0 on success, -1 on failure (txn aborted)
// side effects: writes one record to the LMDB database (durable commit)
int qml_lmdb_put(const char * key, const void * data, size_t size)
{
    if (g_env == NULL)
    {
        return -1;
    }

    MDB_txn * txn = NULL;
    int rc = mdb_txn_begin(g_env, NULL, 0, &txn);
    if (rc != 0)
    {
        return -1;
    }

    MDB_val k;
    k.mv_size = strlen(key);
    k.mv_data = (void *) key;

    MDB_val v;
    v.mv_size = size;
    v.mv_data = (void *) data;

    rc = mdb_put(txn, g_dbi, &k, &v, 0);
    if (rc != 0)
    {
        mdb_txn_abort(txn);
        return -1;
    }

    rc = mdb_txn_commit(txn);
    return (rc == 0) ? 0 : -1;
}

//------------------------------------------------------------
//Func : qml_lmdb_get
//------------------------------------------------------------
// zero-copy read: return a pointer straight into the mmap'd LMDB page
//  in: key — NUL-terminated key string
//  in: out_data (out) — receives pointer into the LMDB mapping (valid until qml_lmdb_get_done)
//  in: out_size (out) — receives the stored blob length in bytes
//  in: out_txn (out) — receives the malloc'd read-txn wrapper the caller must release
// out: 0 on success (key found), 1 if key not found, -1 on failure
int qml_lmdb_get(const char * key, const void ** out_data, size_t * out_size, qml_lmdb_txn_t ** out_txn)
{
    if (g_env == NULL)
    {
        return -1;
    }

    MDB_txn * txn = NULL;
    int rc = mdb_txn_begin(g_env, NULL, MDB_RDONLY, &txn);
    if (rc != 0)
    {
        return -1;
    }

    MDB_val k;
    k.mv_size = strlen(key);
    k.mv_data = (void *) key;

    MDB_val v;
    rc = mdb_get(txn, g_dbi, &k, &v);

    if (rc == MDB_NOTFOUND)
    {
        mdb_txn_abort(txn);
        return 1;
    }

    if (rc != 0)
    {
        mdb_txn_abort(txn);
        return -1;
    }

    qml_lmdb_txn_t * wrapper = (qml_lmdb_txn_t *) malloc(sizeof(qml_lmdb_txn_t));
    wrapper->txn = txn;

    *out_data = v.mv_data;
    *out_size = v.mv_size;
    *out_txn  = wrapper;

    return 0;
}

//------------------------------------------------------------
//Func : qml_lmdb_get_done
//------------------------------------------------------------
// release the read txn opened by qml_lmdb_get; invalidates the data pointer
//  in: txn — txn wrapper returned by qml_lmdb_get (NULL is a no-op)
// out: none (void)
void qml_lmdb_get_done(qml_lmdb_txn_t * txn)
{
    if (txn == NULL)
    {
        return;
    }

    mdb_txn_abort(txn->txn);
    free(txn);
}

//------------------------------------------------------------
//chunked blob format
//
//A session state blob is stored as fixed 4 MiB chunks under keys
//"<key>#000000", "<key>#000001", ... with a 16-byte header under the base
//key. Fixed-size chunks are the whole point: LMDB (mdb_cursor_put) only
//overwrites an overflow value IN PLACE -- no copy-on-write, no freelist
//traffic -- when the env is MDB_WRITEMAP and the new value needs the same
//number of overflow pages as the old one. A whole-session blob changes size
//on every save (sessions grow every turn), so it re-allocates ~size/4096 pages
//per save and the file balloons (measured: +2.14 GB per 500-request churn
//round, 94% of it freelist). With 4 MiB chunks every save rewrites all
//chunks in place and only the LAST (partial) chunk ever re-allocates.
//
//Backward compatibility: the base key of a legacy store holds the whole
//blob (tens of MB). New-format base keys hold exactly sizeof(qml_chunk_hdr_t)
//bytes starting with the magic. qml_lmdb_get_blob() reads both formats;
//writes always produce the chunked format (a legacy value is replaced by a
//header the first time the session is re-saved).
//------------------------------------------------------------
#define QML_CHUNK_SIZE  (4u << 20)
#define QML_CHUNK_MAGIC 0x434c4d51u /* "QMLC" little-endian */

typedef struct
{
    uint32_t magic;
    uint32_t chunk_size;
    uint64_t total_size;
} qml_chunk_hdr_t;

// format the LMDB key of chunk idx as "<key>#NNNNNN" into buf
//  in: buf (out) — destination buffer for the formatted chunk key
//  in: buf_sz — capacity of buf in bytes
//  in: key — base blob key
//  in: idx — zero-based chunk index
// out: none (void); the formatted key is written into buf
static void chunk_key(char * buf, size_t buf_sz, const char * key, uint32_t idx)
{
    snprintf(buf, buf_sz, "%s#%06u", key, idx);
}

//------------------------------------------------------------
//Func : lmdb_log_fail
// Writes one line about a failed LMDB call to stderr (the engine log).
// Why (2026-10-02, H200 load run): blob writes failed without any log line
// while the 64 GiB map was 99.7% used, and the save that depended on them was
// dropped silently. MDB_MAP_FULL now shows up by name.
//  in: where — failing step; key — blob key; size — blob bytes; rc — LMDB code
// out: none
//------------------------------------------------------------
static void lmdb_log_fail(const char * where, const char * key, size_t size, int rc)
{
    fprintf(stderr, "[qml][lmdb] %s failed key=%s size=%zu rc=%d (%s)\n",
            where, key ? key : "-", size, rc, mdb_strerror(rc));
    fflush(stderr);
}

//------------------------------------------------------------
//Func : qml_lmdb_put_blob
//------------------------------------------------------------
// store a blob as fixed 4 MiB chunks plus a 16-byte header, one write txn/commit
//  in: key — base key; the header lives here, chunks under "<key>#NNNNNN"
//  in: data — blob bytes (must not be NULL)
//  in: size — blob length in bytes (must be > 0)
// out: 0 on success, -1 on failure (txn aborted, nothing committed)
// side effects: writes header + chunk records; deletes stale chunks when the blob shrank
// [WA-SKIP 2026-09-02] Write amplification counters -- used to report the savings from
// measurement. Cumulative over the process lifetime, read through query functions (lock-free approximate
// counter: effectively exact because the persister thread is the single writer).
static unsigned long long g_wa_written_bytes = 0;
static unsigned long long g_wa_skipped_bytes = 0;

void qml_lmdb_counters(unsigned long long * written,
                          unsigned long long * skipped)
{
    if (written != NULL) { *written = g_wa_written_bytes; }
    if (skipped != NULL) { *skipped = g_wa_skipped_bytes; }
}

int qml_lmdb_put_blob(const char * key, const void * data, size_t size)
{
    if (g_env == NULL || data == NULL || size == 0)
    {
        return -1;
    }

    MDB_txn * txn = NULL;
    int       mrc = mdb_txn_begin(g_env, NULL, 0, &txn);
    if (mrc != 0)
    {
        lmdb_log_fail("put_blob txn_begin", key, size, mrc);
        return -1;
    }

    MDB_val k;
    k.mv_size = strlen(key);
    k.mv_data = (void *) key;

    // how many chunks did the previous save leave behind?
    uint32_t old_chunks = 0;
    {
        MDB_val old;
        if (mdb_get(txn, g_dbi, &k, &old) == 0 && old.mv_size == sizeof(qml_chunk_hdr_t))
        {
            const qml_chunk_hdr_t * h = (const qml_chunk_hdr_t *) old.mv_data;
            if (h->magic == QML_CHUNK_MAGIC && h->chunk_size > 0)
            {
                old_chunks = (uint32_t) ((h->total_size + h->chunk_size - 1) / h->chunk_size);
            }
        }
    }

    qml_chunk_hdr_t hdr;
    hdr.magic      = QML_CHUNK_MAGIC;
    hdr.chunk_size = QML_CHUNK_SIZE;
    hdr.total_size = (uint64_t) size;

    MDB_val v;
    v.mv_size = sizeof(hdr);
    v.mv_data = &hdr;

    mrc = mdb_put(txn, g_dbi, &k, &v, 0);
    if (mrc != 0)
    {
        lmdb_log_fail("put_blob header", key, size, mrc);
        mdb_txn_abort(txn);
        return -1;
    }

    const uint32_t n_chunks = (uint32_t) ((size + QML_CHUNK_SIZE - 1) / QML_CHUNK_SIZE);
    char           ck[512];

    for (uint32_t i = 0; i < n_chunks; ++i)
    {
        const size_t off  = (size_t) i * QML_CHUNK_SIZE;
        const size_t part = (size - off < QML_CHUNK_SIZE) ? (size - off) : QML_CHUNK_SIZE;

        chunk_key(ck, sizeof(ck), key, i);

        MDB_val kk;
        kk.mv_size = strlen(ck);
        kk.mv_data = ck;

        MDB_val vv;
        vv.mv_size = part;
        vv.mv_data = (void *) ((const uint8_t *) data + off);

        // [WA-SKIP 2026-09-02] Do not rewrite a chunk whose content is identical.
        // Most re-saves are identical to the existing bytes (a re-save of the same length
        // is entirely identical), and LMDB uses MDB_WRITEMAP, so reading an existing chunk is a
        // page-cache mmap access and comparing is much cheaper than a write. First-stage measure to reduce
        // write amplification (TBW) on edge flash -- implemented in response to a review comment (requested a
        // mechanism to suppress write amplification); the actual savings are measured with the counters below
        // and reported.
        {
            MDB_val old_v;
            if (i < old_chunks &&
                mdb_get(txn, g_dbi, &kk, &old_v) == 0 &&
                old_v.mv_size == part &&
                memcmp(old_v.mv_data, vv.mv_data, part) == 0)
            {
                g_wa_skipped_bytes += part;
                continue;   // identical -- not written
            }
        }
        g_wa_written_bytes += part;

        mrc = mdb_put(txn, g_dbi, &kk, &vv, 0);
        if (mrc != 0)
        {
            lmdb_log_fail("put_blob chunk", key, size, mrc);
            mdb_txn_abort(txn);
            return -1;
        }
    }

    // the blob shrank (or the legacy value was replaced): drop stale chunks
    for (uint32_t i = n_chunks; i < old_chunks; ++i)
    {
        chunk_key(ck, sizeof(ck), key, i);

        MDB_val kk;
        kk.mv_size = strlen(ck);
        kk.mv_data = ck;

        (void) mdb_del(txn, g_dbi, &kk, NULL); // NOTFOUND is fine
    }

    mrc = mdb_txn_commit(txn);
    if (mrc != 0)
    {
        lmdb_log_fail("put_blob commit", key, size, mrc);
        return -1;
    }
    return 0;
}

#define QML_BLOB_READ_THREADS 4u

typedef struct
{
    const char * key;
    uint8_t    * buf;
    size_t       total;
    uint32_t     csize;
    uint32_t     n_chunks;
    uint32_t     t_idx;
    uint32_t     n_threads;
    int          rc;      // 0 ok, -2 snapshot mismatch, -1 error
} blob_read_task_t;

//each worker opens its OWN read txn (LMDB txns must not be shared across
//threads). Different workers can therefore land on different snapshots if a
//writer commits in between -- each worker re-reads the header inside its own
//txn and reports -2 on any disagreement, and the caller retries on the
//single-txn path, which is snapshot-consistent by construction. (The llama
//hook additionally excludes same-key races up front via pending_save_lookup.)
// pthread worker: copy every n_threads-th chunk of a blob into the shared buffer
//  in: arg — blob_read_task_t with key, dest buffer, sizes and this worker's index/stride
// out: always NULL; t->rc carries the result (0 ok, -2 snapshot mismatch, -1 error)
static void * blob_read_worker(void * arg)
{
    blob_read_task_t * t = (blob_read_task_t *) arg;
    t->rc = -1;

    MDB_txn * txn = NULL;
    if (mdb_txn_begin(g_env, NULL, MDB_RDONLY, &txn) != 0)
    {
        return NULL;
    }

    MDB_val k;
    k.mv_size = strlen(t->key);
    k.mv_data = (void *) t->key;

    MDB_val hv;
    if (mdb_get(txn, g_dbi, &k, &hv) != 0 || hv.mv_size != sizeof(qml_chunk_hdr_t))
    {
        mdb_txn_abort(txn);
        t->rc = -2;
        return NULL;
    }
    const qml_chunk_hdr_t * h = (const qml_chunk_hdr_t *) hv.mv_data;
    if (h->magic != QML_CHUNK_MAGIC || (size_t) h->total_size != t->total ||
        h->chunk_size != t->csize)
    {
        mdb_txn_abort(txn);
        t->rc = -2;
        return NULL;
    }

    char ck[512];
    for (uint32_t i = t->t_idx; i < t->n_chunks; i += t->n_threads)
    {
        chunk_key(ck, sizeof(ck), t->key, i);

        MDB_val kk;
        kk.mv_size = strlen(ck);
        kk.mv_data = ck;

        MDB_val vv;
        if (mdb_get(txn, g_dbi, &kk, &vv) != 0)
        {
            mdb_txn_abort(txn);
            t->rc = -2;
            return NULL;
        }

        const size_t off      = (size_t) i * t->csize;
        const size_t expected = (t->total - off < t->csize) ? (t->total - off) : t->csize;
        if (vv.mv_size != expected)
        {
            mdb_txn_abort(txn);
            t->rc = -2;
            return NULL;
        }

        memcpy(t->buf + off, vv.mv_data, vv.mv_size);
    }

    mdb_txn_abort(txn);
    t->rc = 0;
    return NULL;
}

//single-txn fallback: same semantics as qml_lmdb_get_blob but never
//multi-threaded -- used when parallel workers detect a snapshot mismatch.
//  in: key — base key of the blob (chunked or legacy single-value format)
//  in: out_data (out) — receives the malloc'd contiguous copy; caller frees
//  in: out_size (out) — receives the blob length in bytes
// out: 0 on success, 1 if key not found, -1 on failure
static int qml_lmdb_get_one(const char * key, void ** out_data, size_t * out_size)
{
    *out_data = NULL;
    *out_size = 0;

    MDB_txn * txn = NULL;
    if (mdb_txn_begin(g_env, NULL, MDB_RDONLY, &txn) != 0)
    {
        return -1;
    }

    MDB_val k;
    k.mv_size = strlen(key);
    k.mv_data = (void *) key;

    MDB_val v;
    int rc = mdb_get(txn, g_dbi, &k, &v);
    if (rc == MDB_NOTFOUND)
    {
        mdb_txn_abort(txn);
        return 1;
    }
    if (rc != 0)
    {
        mdb_txn_abort(txn);
        return -1;
    }

    const qml_chunk_hdr_t * h = (const qml_chunk_hdr_t *) v.mv_data;

    if (v.mv_size != sizeof(qml_chunk_hdr_t) || h->magic != QML_CHUNK_MAGIC)
    {
        uint8_t * buf = (uint8_t *) malloc(v.mv_size);
        if (buf == NULL)
        {
            mdb_txn_abort(txn);
            return -1;
        }
        memcpy(buf, v.mv_data, v.mv_size);
        *out_data = buf;
        *out_size = v.mv_size;
        mdb_txn_abort(txn);
        return 0;
    }

    const size_t   total    = (size_t) h->total_size;
    const uint32_t csize    = h->chunk_size;
    const uint32_t n_chunks = (uint32_t) ((total + csize - 1) / csize);

    uint8_t * buf = (uint8_t *) malloc(total);
    if (buf == NULL)
    {
        mdb_txn_abort(txn);
        return -1;
    }

    char ck[512];
    for (uint32_t i = 0; i < n_chunks; ++i)
    {
        chunk_key(ck, sizeof(ck), key, i);

        MDB_val kk;
        kk.mv_size = strlen(ck);
        kk.mv_data = ck;

        MDB_val vv;
        if (mdb_get(txn, g_dbi, &kk, &vv) != 0)
        {
            free(buf);
            mdb_txn_abort(txn);
            return -1;
        }

        const size_t off      = (size_t) i * csize;
        const size_t expected = (total - off < csize) ? (total - off) : csize;
        if (vv.mv_size != expected)
        {
            free(buf);
            mdb_txn_abort(txn);
            return -1;
        }

        memcpy(buf + off, vv.mv_data, vv.mv_size);
    }

    mdb_txn_abort(txn);

    *out_data = buf;
    *out_size = total;
    return 0;
}

//------------------------------------------------------------
//Func : qml_lmdb_get_blob
//
//Returns a malloc'd contiguous copy (caller frees). Copying is deliberate:
//the read txn closes before this returns, so restores no longer hold a
//reader open across the GPU upload -- long-lived readers were pinning old
//page versions and blocking freelist reuse (the other growth source).
//Blobs of 8+ chunks are assembled by QML_BLOB_READ_THREADS workers.
//------------------------------------------------------------
//  in: key — base key of the blob (chunked or legacy single-value format)
//  in: out_data (out) — receives the malloc'd contiguous copy; caller frees
//  in: out_size (out) — receives the blob length in bytes
// out: 0 on success, 1 if key not found, -1 on failure
int qml_lmdb_get_blob(const char * key, void ** out_data, size_t * out_size)
{
    *out_data = NULL;
    *out_size = 0;

    if (g_env == NULL)
    {
        return -1;
    }

    MDB_txn * txn = NULL;
    if (mdb_txn_begin(g_env, NULL, MDB_RDONLY, &txn) != 0)
    {
        return -1;
    }

    MDB_val k;
    k.mv_size = strlen(key);
    k.mv_data = (void *) key;

    MDB_val v;
    int rc = mdb_get(txn, g_dbi, &k, &v);
    if (rc == MDB_NOTFOUND)
    {
        mdb_txn_abort(txn);
        return 1;
    }
    if (rc != 0)
    {
        mdb_txn_abort(txn);
        return -1;
    }

    const qml_chunk_hdr_t * h = (const qml_chunk_hdr_t *) v.mv_data;

    if (v.mv_size != sizeof(qml_chunk_hdr_t) || h->magic != QML_CHUNK_MAGIC)
    {
        // legacy single-value blob
        uint8_t * buf = (uint8_t *) malloc(v.mv_size);
        if (buf == NULL)
        {
            mdb_txn_abort(txn);
            return -1;
        }
        memcpy(buf, v.mv_data, v.mv_size);
        *out_data = buf;
        *out_size = v.mv_size;
        mdb_txn_abort(txn);
        return 0;
    }

    const size_t   total    = (size_t) h->total_size;
    const uint32_t csize    = h->chunk_size;
    const uint32_t n_chunks = (uint32_t) ((total + csize - 1) / csize);

    uint8_t * buf = (uint8_t *) malloc(total);
    if (buf == NULL)
    {
        mdb_txn_abort(txn);
        return -1;
    }

    // Parallel assembly pays only when the pages are already in RAM
    // (measured: page-cache reads scale 5.4 -> 15.2 GB/s at 4 threads,
    // while cold-disk reads are pinned at the device's 0.52 GB/s no matter
    // the thread count). Small blobs stay on the single-txn path.
    const uint32_t n_threads =
        (n_chunks >= 8) ? (n_chunks < QML_BLOB_READ_THREADS ? n_chunks : QML_BLOB_READ_THREADS) : 1;

    if (n_threads > 1)
    {
        mdb_txn_abort(txn); // workers each open their own read txn

        blob_read_task_t tasks[QML_BLOB_READ_THREADS];
        pthread_t        tids[QML_BLOB_READ_THREADS];

        for (uint32_t t = 0; t < n_threads; ++t)
        {
            tasks[t].key       = key;
            tasks[t].buf       = buf;
            tasks[t].total     = total;
            tasks[t].csize     = csize;
            tasks[t].n_chunks  = n_chunks;
            tasks[t].t_idx     = t;
            tasks[t].n_threads = n_threads;
            tasks[t].rc        = -1;
            pthread_create(&tids[t], NULL, blob_read_worker, &tasks[t]);
        }

        int rc_all = 0;
        for (uint32_t t = 0; t < n_threads; ++t)
        {
            pthread_join(tids[t], NULL);
            if (tasks[t].rc != 0)
            {
                rc_all = tasks[t].rc;
            }
        }

        if (rc_all == 0)
        {
            *out_data = buf;
            *out_size = total;
            return 0;
        }

        // a worker saw a different header (a writer committed a new save
        // between the workers' txn begins) or a missing chunk: retry once
        // on the plain single-txn path, which is snapshot-consistent.
        free(buf);
        return qml_lmdb_get_one(key, out_data, out_size);
    }

    char ck[512];
    for (uint32_t i = 0; i < n_chunks; ++i)
    {
        chunk_key(ck, sizeof(ck), key, i);

        MDB_val kk;
        kk.mv_size = strlen(ck);
        kk.mv_data = ck;

        MDB_val vv;
        if (mdb_get(txn, g_dbi, &kk, &vv) != 0)
        {
            free(buf);
            mdb_txn_abort(txn);
            return -1; // header says the chunk exists -- missing = corrupt
        }

        const size_t off      = (size_t) i * csize;
        const size_t expected = (total - off < csize) ? (total - off) : csize;
        if (vv.mv_size != expected)
        {
            free(buf);
            mdb_txn_abort(txn);
            return -1;
        }

        memcpy(buf + off, vv.mv_data, vv.mv_size);
    }

    mdb_txn_abort(txn);

    *out_data = buf;
    *out_size = total;
    return 0;
}

//------------------------------------------------------------
//Func : qml_lmdb_blob_stat
//
//Existence + size check WITHOUT copying the blob. The token-peek path only
//needs "is there a usable state, and how big" -- reading the header (16
//bytes) answers that; assembling a multi-GB copy to answer it was measured
//as a whole second of the warm restore at 128K tokens.
//returns 0 found (*out_size = stored blob size), 1 not found, -1 error.
//------------------------------------------------------------
//  in: key — base key of the blob
//  in: out_size (out) — receives the stored blob size in bytes
// out: 0 found, 1 not found, -1 on failure
int qml_lmdb_blob_stat(const char * key, size_t * out_size)
{
    *out_size = 0;

    if (g_env == NULL)
    {
        return -1;
    }

    MDB_txn * txn = NULL;
    if (mdb_txn_begin(g_env, NULL, MDB_RDONLY, &txn) != 0)
    {
        return -1;
    }

    MDB_val k;
    k.mv_size = strlen(key);
    k.mv_data = (void *) key;

    MDB_val v;
    int rc = mdb_get(txn, g_dbi, &k, &v);
    if (rc == MDB_NOTFOUND)
    {
        mdb_txn_abort(txn);
        return 1;
    }
    if (rc != 0)
    {
        mdb_txn_abort(txn);
        return -1;
    }

    const qml_chunk_hdr_t * h = (const qml_chunk_hdr_t *) v.mv_data;
    if (v.mv_size == sizeof(qml_chunk_hdr_t) && h->magic == QML_CHUNK_MAGIC)
    {
        *out_size = (size_t) h->total_size;
    }
    else
    {
        *out_size = v.mv_size; // legacy single-value blob
    }

    mdb_txn_abort(txn);
    return 0;
}

//------------------------------------------------------------
//Func : qml_lmdb_delete_blob
//------------------------------------------------------------
// delete a blob's header and every chunk it records, in one write txn/commit
//  in: key — base key of the blob
// out: 0 on success (or key absent), -1 on failure
// side effects: deletes the header and chunk records from the DB (durable commit)
int qml_lmdb_delete_blob(const char * key)
{
    if (g_env == NULL)
    {
        return -1;
    }

    MDB_txn * txn = NULL;
    if (mdb_txn_begin(g_env, NULL, 0, &txn) != 0)
    {
        return -1;
    }

    MDB_val k;
    k.mv_size = strlen(key);
    k.mv_data = (void *) key;

    uint32_t n_chunks = 0;
    {
        MDB_val v;
        if (mdb_get(txn, g_dbi, &k, &v) == 0 && v.mv_size == sizeof(qml_chunk_hdr_t))
        {
            const qml_chunk_hdr_t * h = (const qml_chunk_hdr_t *) v.mv_data;
            if (h->magic == QML_CHUNK_MAGIC && h->chunk_size > 0)
            {
                n_chunks = (uint32_t) ((h->total_size + h->chunk_size - 1) / h->chunk_size);
            }
        }
    }

    (void) mdb_del(txn, g_dbi, &k, NULL);

    char ck[512];
    for (uint32_t i = 0; i < n_chunks; ++i)
    {
        chunk_key(ck, sizeof(ck), key, i);

        MDB_val kk;
        kk.mv_size = strlen(ck);
        kk.mv_data = ck;

        (void) mdb_del(txn, g_dbi, &kk, NULL);
    }

    return (mdb_txn_commit(txn) == 0) ? 0 : -1;
}

//------------------------------------------------------------
//Func : qml_lmdb_list_blobs
// Lists every chunked blob (its header entry) in the main DB with its size.
// Used by the SWEEP_BLOBS control command to find blobs no session refers to.
//  in: out_keys — receives a malloc'd array of malloc'd key strings
//  in: out_sizes — receives a malloc'd array of blob sizes in bytes (total_size)
//  in: out_n — receives the number of blobs
// out: 0 on success; -1 on error (nothing allocated)
// side effects: one read-only transaction; only entries whose value is a
//               chunk header (QML_CHUNK_MAGIC) are listed, so chunk entries
//               ("<key>#NNNNNN") and any non-blob entries are never returned
//------------------------------------------------------------
int qml_lmdb_list_blobs(char *** out_keys, uint64_t ** out_sizes, size_t * out_n)
{
    if (g_env == NULL || out_keys == NULL || out_sizes == NULL || out_n == NULL)
    {
        return -1;
    }
    MDB_txn * txn = NULL;
    int       mrc = mdb_txn_begin(g_env, NULL, MDB_RDONLY, &txn);
    if (mrc != 0)
    {
        lmdb_log_fail("list_blobs txn_begin", NULL, 0, mrc);
        return -1;
    }
    MDB_cursor * cur = NULL;
    mrc = mdb_cursor_open(txn, g_dbi, &cur);
    if (mrc != 0)
    {
        lmdb_log_fail("list_blobs cursor", NULL, 0, mrc);
        mdb_txn_abort(txn);
        return -1;
    }
    size_t     cap = 256, n = 0;
    char    ** keys  = (char **) malloc(cap * sizeof(char *));
    uint64_t * sizes = (uint64_t *) malloc(cap * sizeof(uint64_t));
    int        ok    = (keys != NULL && sizes != NULL);
    MDB_val    k, v;
    while (ok && mdb_cursor_get(cur, &k, &v, MDB_NEXT) == 0)
    {
        if (v.mv_size != sizeof(qml_chunk_hdr_t))
        {
            continue;
        }
        const qml_chunk_hdr_t * h = (const qml_chunk_hdr_t *) v.mv_data;
        if (h->magic != QML_CHUNK_MAGIC)
        {
            continue;
        }
        if (n == cap)
        {
            cap *= 2;
            char    ** nk = (char **) realloc(keys, cap * sizeof(char *));
            uint64_t * ns = (uint64_t *) realloc(sizes, cap * sizeof(uint64_t));
            if (nk != NULL)
            {
                keys = nk;
            }
            if (ns != NULL)
            {
                sizes = ns;
            }
            if (nk == NULL || ns == NULL)
            {
                ok = 0;
                break;
            }
        }
        keys[n] = (char *) malloc(k.mv_size + 1);
        if (keys[n] == NULL)
        {
            ok = 0;
            break;
        }
        memcpy(keys[n], k.mv_data, k.mv_size);
        keys[n][k.mv_size] = '\0';
        sizes[n] = h->total_size;
        ++n;
    }
    mdb_cursor_close(cur);
    mdb_txn_abort(txn);
    if (!ok)
    {
        for (size_t i = 0; keys != NULL && i < n; ++i)
        {
            free(keys[i]);
        }
        free(keys);
        free(sizes);
        return -1;
    }
    *out_keys  = keys;
    *out_sizes = sizes;
    *out_n     = n;
    return 0;
}

//------------------------------------------------------------
//Func : qml_lmdb_free_list
// Frees the arrays returned by qml_lmdb_list_blobs.
//  in: keys, sizes, n — as returned (NULL-safe)
// out: none
//------------------------------------------------------------
void qml_lmdb_free_list(char ** keys, uint64_t * sizes, size_t n)
{
    for (size_t i = 0; keys != NULL && i < n; ++i)
    {
        free(keys[i]);
    }
    free(keys);
    free(sizes);
}

//------------------------------------------------------------
//Func : qml_lmdb_get_many
//------------------------------------------------------------
// batched fixed-stride read: copy up to n_keys blobs under one read txn
//  in: keys — array of n_keys NUL-terminated key strings (a NULL entry ends the run)
//  in: n_keys — number of keys
//  in: out_buf (out) — caller-owned buffer; blob i is copied to offset i*stride
//  in: stride — exact required size of every blob, and the output spacing
// out: number of blobs copied contiguously from index 0, or -1 on failure
long qml_lmdb_get_many(const char * const * keys, size_t n_keys, void * out_buf, size_t stride)
{
    if (g_env == NULL || keys == NULL || out_buf == NULL || stride == 0)
    {
        return -1;
    }

    MDB_txn * txn = NULL;
    int rc = mdb_txn_begin(g_env, NULL, MDB_RDONLY, &txn);
    if (rc != 0)
    {
        return -1;
    }

    unsigned char * dst = (unsigned char *) out_buf;
    size_t          n   = 0;

    for (; n < n_keys; ++n)
    {
        if (keys[n] == NULL)
        {
            break;
        }

        MDB_val k;
        k.mv_size = strlen(keys[n]);
        k.mv_data = (void *) keys[n];

        MDB_val v;
        rc = mdb_get(txn, g_dbi, &k, &v);

        //a miss or a wrong-sized blob ends the contiguous run -- same
        //policy as the per-key path, which stops the load there and
        //lets the engine recompute the rest
        if (rc != 0 || v.mv_size != stride)
        {
            break;
        }

        memcpy(dst + n * stride, v.mv_data, stride);
    }

    mdb_txn_abort(txn);

    return (long) n;
}

//------------------------------------------------------------
//Func : qml_lmdb_put_many
//------------------------------------------------------------
// batched fixed-stride write: n_keys blobs under one write txn and one commit
//  in: keys — array of n_keys key strings (NULL entries are skipped)
//  in: n_keys — number of keys
//  in: buf — source buffer; blob i is read from offset i*stride
//  in: stride — exact size of every blob
// out: 0 on success, -1 on failure (txn aborted, nothing committed)
// side effects: writes n_keys records in one durable commit
int qml_lmdb_put_many(const char * const * keys, size_t n_keys, const void * buf, size_t stride)
{
    if (g_env == NULL || keys == NULL || buf == NULL || stride == 0)
    {
        return -1;
    }

    if (n_keys == 0)
    {
        return 0;
    }

    MDB_txn * txn = NULL;
    int rc = mdb_txn_begin(g_env, NULL, 0, &txn);
    if (rc != 0)
    {
        return -1;
    }

    const unsigned char * src = (const unsigned char *) buf;

    for (size_t i = 0; i < n_keys; ++i)
    {
        if (keys[i] == NULL)
        {
            continue;
        }

        MDB_val k;
        k.mv_size = strlen(keys[i]);
        k.mv_data = (void *) keys[i];

        MDB_val v;
        v.mv_size = stride;
        v.mv_data = (void *) (src + i * stride);

        rc = mdb_put(txn, g_dbi, &k, &v, 0);
        if (rc != 0)
        {
            mdb_txn_abort(txn);
            return -1;
        }
    }

    //one commit for the whole layer instead of one per block: a write txn
    //per block turns a single sequential write into thousands of separate
    //durable commits, inside the forward pass
    return (mdb_txn_commit(txn) == 0) ? 0 : -1;
}

//------------------------------------------------------------
//Func : qml_lmdb_get_scatter
//------------------------------------------------------------
// batched read into separate per-key destination buffers under one read txn
//  in: keys — array of n_keys key strings (a NULL entry ends the run)
//  in: n_keys — number of keys
//  in: dsts (out) — per-key destination buffers (a NULL entry ends the run)
//  in: sizes — exact expected size of each blob, per key
// out: count copied contiguously from index 0 (miss or size mismatch stops the run), or -1 on failure
long qml_lmdb_get_scatter(const char * const * keys, size_t n_keys, void * const * dsts, const size_t * sizes)
{
    if (g_env == NULL || keys == NULL || dsts == NULL || sizes == NULL)
    {
        return -1;
    }

    MDB_txn * txn = NULL;
    if (mdb_txn_begin(g_env, NULL, MDB_RDONLY, &txn) != 0)
    {
        return -1;
    }

    size_t n = 0;

    for (; n < n_keys; ++n)
    {
        if (keys[n] == NULL || dsts[n] == NULL)
        {
            break;
        }

        MDB_val k;
        k.mv_size = strlen(keys[n]);
        k.mv_data = (void *) keys[n];

        MDB_val v;
        if (mdb_get(txn, g_dbi, &k, &v) != 0 || v.mv_size != sizes[n])
        {
            break;
        }

        memcpy(dsts[n], v.mv_data, sizes[n]);
    }

    mdb_txn_abort(txn);

    return (long) n;
}

//------------------------------------------------------------
//Func : qml_lmdb_put_gather
//------------------------------------------------------------
// batched write from separate per-key source buffers, one write txn and one commit
//  in: keys — array of n_keys key strings (NULL entries are skipped)
//  in: n_keys — number of keys
//  in: srcs — per-key source buffers (NULL entries are skipped)
//  in: sizes — size of each blob, per key
// out: 0 on success, -1 on failure (txn aborted, nothing committed)
// side effects: writes the batch in one durable commit
int qml_lmdb_put_gather(const char * const * keys, size_t n_keys, const void * const * srcs, const size_t * sizes)
{
    if (g_env == NULL || keys == NULL || srcs == NULL || sizes == NULL)
    {
        return -1;
    }

    if (n_keys == 0)
    {
        return 0;
    }

    MDB_txn * txn = NULL;
    if (mdb_txn_begin(g_env, NULL, 0, &txn) != 0)
    {
        return -1;
    }

    for (size_t i = 0; i < n_keys; ++i)
    {
        if (keys[i] == NULL || srcs[i] == NULL)
        {
            continue;
        }

        MDB_val k;
        k.mv_size = strlen(keys[i]);
        k.mv_data = (void *) keys[i];

        MDB_val v;
        v.mv_size = sizes[i];
        v.mv_data = (void *) srcs[i];

        if (mdb_put(txn, g_dbi, &k, &v, 0) != 0)
        {
            mdb_txn_abort(txn);
            return -1;
        }
    }

    return (mdb_txn_commit(txn) == 0) ? 0 : -1;
}

//------------------------------------------------------------
//Func : qml_lmdb_exists_run
//------------------------------------------------------------
// count keys present contiguously from index 0 (stops at first miss), one read txn
//  in: keys — array of n_keys key strings (a NULL entry ends the run)
//  in: n_keys — number of keys
// out: the count of keys found, or -1 on failure
long qml_lmdb_exists_run(const char * const * keys, size_t n_keys)
{
    if (g_env == NULL || keys == NULL)
    {
        return -1;
    }

    MDB_txn * txn = NULL;
    int rc = mdb_txn_begin(g_env, NULL, MDB_RDONLY, &txn);
    if (rc != 0)
    {
        return -1;
    }

    size_t n = 0;

    for (; n < n_keys; ++n)
    {
        if (keys[n] == NULL)
        {
            break;
        }

        MDB_val k;
        k.mv_size = strlen(keys[n]);
        k.mv_data = (void *) keys[n];

        MDB_val v;
        if (mdb_get(txn, g_dbi, &k, &v) != 0)
        {
            break;
        }
    }

    mdb_txn_abort(txn);

    return (long) n;
}

//------------------------------------------------------------
//Func : qml_lmdb_delete
//------------------------------------------------------------
// delete one key in a single committed write txn (absent key counts as success)
//  in: key — key to delete
// out: 0 on success (or key already absent), -1 on failure
// side effects: deletes the record from the DB (durable commit)
int qml_lmdb_delete(const char * key)
{
    if (g_env == NULL)
    {
        return -1;
    }

    MDB_txn * txn = NULL;
    int rc = mdb_txn_begin(g_env, NULL, 0, &txn);
    if (rc != 0)
    {
        return -1;
    }

    MDB_val k;
    k.mv_size = strlen(key);
    k.mv_data = (void *) key;

    rc = mdb_del(txn, g_dbi, &k, NULL);
    if (rc != 0 && rc != MDB_NOTFOUND)
    {
        mdb_txn_abort(txn);
        return -1;
    }

    rc = mdb_txn_commit(txn);
    return (rc == 0) ? 0 : -1;
}

//------------------------------------------------------------
//Func : qml_lmdb_delete_many
//deletes keys in one write transaction (a window trim drops a session's
//whole set of KV blocks at once); missing keys are skipped
//  in: keys / n — keys to delete
// out: number of keys deleted (>= 0); -1 on error
//------------------------------------------------------------
long qml_lmdb_delete_many(const char * const * keys, size_t n)
{
    if (g_env == NULL || keys == NULL)
    {
        return -1;
    }
    if (n == 0)
    {
        return 0;
    }
    MDB_txn * txn = NULL;
    int rc = mdb_txn_begin(g_env, NULL, 0, &txn);
    if (rc != 0)
    {
        lmdb_log_fail("delete_many txn_begin", NULL, 0, rc);
        return -1;
    }
    long deleted = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (keys[i] == NULL)
        {
            continue;
        }
        MDB_val k;
        k.mv_size = strlen(keys[i]);
        k.mv_data = (void *) keys[i];
        rc = mdb_del(txn, g_dbi, &k, NULL);
        if (rc == 0)
        {
            deleted++;
        }
        else if (rc != MDB_NOTFOUND)
        {
            lmdb_log_fail("delete_many del", keys[i], 0, rc);
            mdb_txn_abort(txn);
            return -1;
        }
    }
    rc = mdb_txn_commit(txn);
    if (rc != 0)
    {
        lmdb_log_fail("delete_many commit", NULL, 0, rc);
        return -1;
    }
    return deleted;
}
