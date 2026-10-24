// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
// storage/lmdb/qml_lmdb_store.h
//
// mmap-backed heavy blob store (KV cache state buffers).
// LMDB pages are mmap'd by the OS already, so qml_lmdb_get() hands back
// a pointer straight into that mapping instead of copying — call
// qml_lmdb_get_done() once the caller is finished reading it.

#ifndef QML_LMDB_STORE_H
#define QML_LMDB_STORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qml_lmdb_txn qml_lmdb_txn_t;

//------------------------------------------------------------
//Func : qml_lmdb_open
//lmdb_path : directory holding the LMDB data/lock files (created if absent)
//map_size_bytes : max size of the memory-mapped region (LMDB env mapsize)
//returns 0 on success, -1 on failure
//------------------------------------------------------------
// open (or create) the process-wide LMDB environment and default database
//  in: lmdb_path — directory holding the LMDB data/lock files (created if absent)
//  in: map_size_bytes — maximum size of the memory-mapped region (LMDB env mapsize)
// out: 0 on success, -1 on failure
// side effects: creates the directory and LMDB data/lock files on first open
int qml_lmdb_open(const char * lmdb_path, size_t map_size_bytes);

//------------------------------------------------------------
//Func : qml_lmdb_close
//------------------------------------------------------------
// close the global LMDB environment (safe to call when already closed)
//  in: none
// out: none (void)
void qml_lmdb_close(void);

//------------------------------------------------------------
//Func : qml_lmdb_put
//stores/overwrites the blob under key. single write txn, commits before
//returning. used for persisting llama_state_seq_get_data() output.
//returns 0 on success, -1 on failure
//------------------------------------------------------------
// store/overwrite one value under key in a single committed write txn
//  in: key — NUL-terminated key string
//  in: data — value bytes to store
//  in: size — value length in bytes
// out: 0 on success, -1 on failure
// side effects: writes one record to the LMDB database (durable commit)
int qml_lmdb_put(const char * key, const void * data, size_t size);

//------------------------------------------------------------
//Func : qml_lmdb_get
//zero-copy read: *out_data points directly into the mmap'd LMDB page,
//*out_size is the stored blob length, *out_txn is the read txn that
//must stay open while *out_data is in use.
//returns 0 on success (key found), 1 if key not found, -1 on failure
//------------------------------------------------------------
// zero-copy read: hand back a pointer directly into the mmap'd LMDB page
//  in: key — NUL-terminated key string
//  in: out_data (out) — receives pointer into the LMDB mapping (valid until qml_lmdb_get_done)
//  in: out_size (out) — receives the stored blob length in bytes
//  in: out_txn (out) — receives the read txn that must stay open while *out_data is in use
// out: 0 on success (key found), 1 if key not found, -1 on failure
int qml_lmdb_get(const char * key, const void ** out_data, size_t * out_size, qml_lmdb_txn_t ** out_txn);

//------------------------------------------------------------
//Func : qml_lmdb_get_done
//releases the read txn opened by qml_lmdb_get. after this call,
//the pointer previously returned in *out_data is no longer valid.
//------------------------------------------------------------
// release the read txn opened by qml_lmdb_get (invalidates the returned pointer)
//  in: txn — txn handle returned by qml_lmdb_get (NULL is a no-op)
// out: none (void)
void qml_lmdb_get_done(qml_lmdb_txn_t * txn);

//------------------------------------------------------------
//chunked blob API (large, frequently re-saved values)
//
//Stores a blob as fixed 4 MiB chunks so that a re-save of a grown session
//rewrites every full chunk IN PLACE (LMDB overwrites an overflow value
//without copy-on-write when the page count is unchanged under MDB_WRITEMAP).
//The single-value path re-allocates the whole blob on every save because the
//size always changes, which is what made the store file grow ~2 GB per
//500-request churn round. qml_lmdb_get_blob returns a malloc'd copy (caller
//frees) and closes its read txn before returning, so no reader stays open
//across a GPU upload. Reads understand both the chunked format and the
//legacy single-value format.
//returns 0 on success, 1 if key not found, -1 on failure
//------------------------------------------------------------
// store a blob as fixed 4 MiB chunks plus a small header (in-place chunk rewrite on re-save)
//  in: key — base key; header lives here, chunks under "<key>#NNNNNN"
//  in: data — blob bytes (must not be NULL)
//  in: size — blob length in bytes (must be > 0)
// out: 0 on success, -1 on failure (nothing committed)
// side effects: writes header + chunk records and deletes stale chunks (durable commit)
int qml_lmdb_put_blob(const char * key, const void * data, size_t size);

//Func : qml_lmdb_counters
//Write amplification counters (process cumulative): written = bytes actually written,
//skipped = bytes skipped because the content was identical. Savings ratio = skipped/(written+skipped).
void qml_lmdb_counters(unsigned long long * written,
                          unsigned long long * skipped);
// read a chunked (or legacy single-value) blob into a malloc'd contiguous copy
//  in: key — base key of the blob
//  in: out_data (out) — receives the malloc'd copy; caller frees
//  in: out_size (out) — receives blob length in bytes
// out: 0 on success, 1 if key not found, -1 on failure
int qml_lmdb_get_blob(const char * key, void ** out_data, size_t * out_size);
// report a blob's stored size without copying it (header-only read)
//  in: key — base key of the blob
//  in: out_size (out) — receives the stored blob size in bytes
// out: 0 found, 1 not found, -1 on failure
int qml_lmdb_blob_stat(const char * key, size_t * out_size);
// delete a blob's header and all of its chunk records in one txn
//  in: key — base key of the blob
// out: 0 on success (or key absent), -1 on failure
// side effects: deletes header and chunk records from the DB (durable commit)
int qml_lmdb_delete_blob(const char * key);

//------------------------------------------------------------
//Func : qml_lmdb_list_blobs / qml_lmdb_free_list
//lists every chunked blob header key with its size (read-only); chunk entries
//and non-blob entries are skipped. 0 on success, -1 on error.
//------------------------------------------------------------
int  qml_lmdb_list_blobs(char *** out_keys, uint64_t ** out_sizes, size_t * out_n);
void qml_lmdb_free_list(char ** keys, uint64_t * sizes, size_t n);

//------------------------------------------------------------
//Func : qml_lmdb_get_many
//reads n_keys blobs under ONE read txn and copies each into
//out_buf at i*stride. every blob must be exactly stride bytes;
//a shorter/longer one is treated as the end of the run (the
//caller decides what a partial result means).
//
//this exists because the per-key path (txn begin + abort + a
//copy into a freshly allocated host buffer, per key) is what
//makes a KV restore slower than recomputing it on a GPU: a
//7k-token session is ~14,000 keys, and at ~270us each that is
//nearly 4 seconds of pure call overhead. one txn and one memcpy
//per key into a buffer the caller already owns removes it.
//
//returns the number of blobs copied contiguously from index 0
//(== n_keys when everything was present and correctly sized),
//or -1 on failure.
//------------------------------------------------------------
// batched fixed-stride read: copy up to n_keys blobs under one read txn
//  in: keys — array of n_keys NUL-terminated key strings (a NULL entry ends the run)
//  in: n_keys — number of keys
//  in: out_buf (out) — caller-owned buffer; blob i is copied to offset i*stride
//  in: stride — exact required size of every blob, and the output spacing
// out: number of blobs copied contiguously from index 0, or -1 on failure
long qml_lmdb_get_many(const char * const * keys, size_t n_keys, void * out_buf, size_t stride);

//------------------------------------------------------------
//Func : qml_lmdb_put_many
//writes n_keys blobs of exactly stride bytes each, taken from
//buf at i*stride, under ONE write txn and ONE commit. the
//mirror of qml_lmdb_get_many for the save path, where a txn per
//block runs inside the engine's forward pass.
//returns 0 on success, -1 on failure (nothing is committed).
//------------------------------------------------------------
// batched fixed-stride write: n_keys blobs under one write txn and one commit
//  in: keys — array of n_keys key strings (NULL entries are skipped)
//  in: n_keys — number of keys
//  in: buf — source buffer; blob i is read from offset i*stride
//  in: stride — exact size of every blob
// out: 0 on success, -1 on failure (nothing is committed)
// side effects: writes n_keys records in one durable commit
int qml_lmdb_put_many(const char * const * keys, size_t n_keys, const void * buf, size_t stride);

//------------------------------------------------------------
//Func : qml_lmdb_exists_run
//counts how many of keys[0..n_keys) exist, from index 0, stopping
//at the first miss -- one read txn, no value copies. this is the
//"how long is the cached prefix" question a KV connector asks on
//every request, and asking it key by key costs a txn per block.
//returns the count, or -1 on failure.
//------------------------------------------------------------
// count keys present contiguously from index 0 (stops at first miss), one read txn
//  in: keys — array of n_keys key strings (a NULL entry ends the run)
//  in: n_keys — number of keys
// out: the count of keys found, or -1 on failure
long qml_lmdb_exists_run(const char * const * keys, size_t n_keys);

//------------------------------------------------------------
//Func : qml_lmdb_get_scatter / qml_lmdb_put_gather
//per-key destination/source POINTER variants of get_many/put_many,
//for callers whose buffers are separate allocations (SGLang hands the
//backend one pre-allocated host tensor PER PAGE). one read txn / one
//write commit for the whole batch. get returns the count copied
//contiguously from index 0 (a miss or size mismatch stops the run);
//put returns 0/-1 and commits nothing on failure.
//------------------------------------------------------------
// batched read into separate per-key destination buffers under one read txn
//  in: keys — array of n_keys key strings (a NULL entry ends the run)
//  in: n_keys — number of keys
//  in: dsts (out) — per-key destination buffers (a NULL entry ends the run)
//  in: sizes — exact expected size of each blob, per key
// out: count copied contiguously from index 0 (miss or size mismatch stops the run), or -1 on failure
long qml_lmdb_get_scatter(const char * const * keys, size_t n_keys, void * const * dsts, const size_t * sizes);
// batched write from separate per-key source buffers, one write txn and one commit
//  in: keys — array of n_keys key strings (NULL entries are skipped)
//  in: n_keys — number of keys
//  in: srcs — per-key source buffers (NULL entries are skipped)
//  in: sizes — size of each blob, per key
// out: 0 on success, -1 on failure (nothing is committed)
// side effects: writes the batch in one durable commit
int  qml_lmdb_put_gather(const char * const * keys, size_t n_keys, const void * const * srcs, const size_t * sizes);

//------------------------------------------------------------
//Func : qml_lmdb_delete
//returns 0 on success (or key already absent), -1 on failure
//------------------------------------------------------------
// delete one key in a single committed write txn (absent key counts as success)
//  in: key — key to delete
// out: 0 on success (or key already absent), -1 on failure
// side effects: deletes the record from the DB (durable commit)
int qml_lmdb_delete(const char * key);

//------------------------------------------------------------
//Func : qml_lmdb_delete_many
//deletes keys in one write transaction; missing keys are not an error
//  in: keys / n — keys to delete
// out: number of keys deleted (>= 0); -1 on error
//------------------------------------------------------------
long qml_lmdb_delete_many(const char * const * keys, size_t n);

#ifdef __cplusplus
}
#endif

#endif // QML_LMDB_STORE_H
