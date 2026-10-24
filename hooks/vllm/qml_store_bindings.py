# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
"""qml_store_bindings.py

ctypes binding over libqml_session_store.so (storage/sqlite + storage/lmdb,
compiled via qml-router/CMakeLists.txt's `qml_session_store_shared` target).

This is the ONLY place vLLM-side Python code should touch qml-router's C
storage layer directly -- everything else (hooks/vllm/qml_vllm_connector.py)
should go through the functions here, not ctypes calls of its own, so the
struct layouts only need to be kept in sync with qml_format.h/
qml_sqlite_store.h in one place.

Struct layouts below are hand-mirrored from:
  - include/qml_format.h            (qml_session_meta_t, qml_vllm_session_meta_t)
  - storage/sqlite/qml_sqlite_store.h (qml_sqlite_upsert/select_vllm_session)
  - storage/lmdb/qml_lmdb_store.h    (qml_lmdb_put/get/get_done/delete)

If those C headers change shape, this file must be updated to match -- there
is no automatic binding generation here.
"""

import ctypes
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

QML_ENGINE_LLAMA_CPP = 0
QML_ENGINE_VLLM = 1
QML_ENGINE_SGLANG = 2

_USER_ID_LEN = 128
_SESSION_ID_LEN = 256
_MODEL_HASH_LEN = 65  # 64 hex chars (SHA256) + NUL -- see core/qml_model_hash.c
_LMDB_PREFIX_LEN = 256


class _QmlSessionMetaBase(ctypes.Structure):
    _fields_ = [
        ("user_id", ctypes.c_char * _USER_ID_LEN),
        ("session_id", ctypes.c_char * _SESSION_ID_LEN),
        ("engine_type", ctypes.c_int32),
        ("seq_id", ctypes.c_int32),
        ("last_token", ctypes.c_int32),
        ("last_pos", ctypes.c_int32),
        ("n_tokens", ctypes.c_int64),
        ("updated_at", ctypes.c_int64),
    ]


class QmlVllmSessionMeta(ctypes.Structure):
    _fields_ = [
        ("base", _QmlSessionMetaBase),
        ("model_hash", ctypes.c_char * _MODEL_HASH_LEN),
        ("lmdb_prefix", ctypes.c_char * _LMDB_PREFIX_LEN),
        ("block_size", ctypes.c_int32),
        ("num_blocks", ctypes.c_int32),
    ]


class QmlControlCmd(ctypes.Structure):
    """Mirrors storage/sqlite/qml_sqlite_store.h's qml_control_cmd_t --
    one claimed row of the UDF control-plane command queue (see
    docs/udf-control-plane-proposal.md)."""
    _fields_ = [
        ("cmd_id", ctypes.c_int64),
        ("target_engine", ctypes.c_char * 32),
        ("cmd_type", ctypes.c_char * 32),
        ("session_id", ctypes.c_char * 256),
        ("created_at", ctypes.c_int64),
    ]


class QmlTokenRec(ctypes.Structure):
    """Mirrors storage/sqlite/qml_sqlite_store.h's qml_token_rec_t --
    one row of the (engine-agnostic) qml_token_log table."""
    _fields_ = [
        ("pos", ctypes.c_int32),
        ("seq_id", ctypes.c_int32),
        ("token_id", ctypes.c_int32),
    ]


def _default_lib():
    """Locates libqml_session_store.so relative to this file's build tree.

    in: none
    out: absolute path string to the shared library, or None if not found
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "build", "libqml_session_store.so"),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.exists(c):
            return c
    return None


class QmlStore:
    """Thin, explicit wrapper -- one process should hold exactly one of
    these (mirrors qml_store_init/shutdown's process-global
    nature on the C side; the underlying store is a single global
    connection, not per-instance)."""

    def __init__(self, lib_path=None):
        """Loads the C store library and prepares (but does not open) the store.

        in: lib_path -- explicit path to libqml_session_store.so, or None to
            search the default build location
        out: None; raises FileNotFoundError when the library cannot be located
        """
        lib_path = lib_path or _default_lib()
        if lib_path is None:
            raise FileNotFoundError(
                "libqml_session_store.so not found -- pass lib_path= explicitly "
                "or build it via qml-router's CMakeLists.txt (qml_session_store_shared target)"
            )
        self._lib = ctypes.CDLL(lib_path)
        self._configure_signatures()
        self._opened = False
        # lazily created worker pool for blocks_get_into(threads=N)
        self._pool_obj = None
        self._pool_threads = 0
        self._pool_lock = threading.Lock()
        self._touch_q = None

    def _configure_signatures(self):
        """Declares argtypes/restype for every C entry point this wrapper calls.

        in: none
        out: None (mutates self._lib's function signature metadata only)
        """
        lib = self._lib

        lib.qml_store_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.qml_store_init.restype = ctypes.c_int

        lib.qml_store_close.argtypes = []
        lib.qml_store_close.restype = None

        lib.qml_sqlite_put_vllm_session.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(QmlVllmSessionMeta),
        ]
        lib.qml_sqlite_put_vllm_session.restype = ctypes.c_int

        lib.qml_sqlite_get_vllm_session.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(QmlVllmSessionMeta),
        ]
        lib.qml_sqlite_get_vllm_session.restype = ctypes.c_int

        lib.qml_lmdb_put.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.qml_lmdb_put.restype = ctypes.c_int

        lib.qml_lmdb_get.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.qml_lmdb_get.restype = ctypes.c_int

        lib.qml_lmdb_get_done.argtypes = [ctypes.c_void_p]
        lib.qml_lmdb_get_done.restype = None

        lib.qml_lmdb_delete.argtypes = [ctypes.c_char_p]
        lib.qml_lmdb_delete.restype = ctypes.c_int

        # task #65: batched read straight into a caller-owned buffer
        lib.qml_lmdb_get_many.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.qml_lmdb_get_many.restype = ctypes.c_long
        lib.qml_lmdb_put_many.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.qml_lmdb_put_many.restype = ctypes.c_int
        lib.qml_lmdb_exists_run.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t]
        lib.qml_lmdb_exists_run.restype = ctypes.c_long
        # task #67: per-key pointer variants for callers whose buffers are
        # separate allocations (SGLang: one host tensor per page)
        lib.qml_lmdb_get_scatter.argtypes = [
            ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.qml_lmdb_get_scatter.restype = ctypes.c_long
        lib.qml_lmdb_put_gather.argtypes = [
            ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.qml_lmdb_put_gather.restype = ctypes.c_int

        # task #56: LRU accounting for content-addressed blocks
        lib.qml_sqlite_block_touch.argtypes = [ctypes.c_char_p, ctypes.c_int64]
        lib.qml_sqlite_block_touch.restype = ctypes.c_int
        lib.qml_sqlite_block_touch_many.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_size_t,
            ctypes.c_int64,
        ]
        lib.qml_sqlite_block_touch_many.restype = ctypes.c_int
        lib.qml_sqlite_block_total_bytes.argtypes = [ctypes.POINTER(ctypes.c_int64)]
        lib.qml_sqlite_block_total_bytes.restype = ctypes.c_int
        lib.qml_sqlite_block_evict_lru.argtypes = [
            ctypes.c_int64,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.qml_sqlite_block_evict_lru.restype = ctypes.c_int
        lib.qml_sqlite_block_evict_free.argtypes = [
            ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t
        ]
        lib.qml_sqlite_block_evict_free.restype = None

        lib.qml_sqlite_put_tokens.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(QmlTokenRec),
            ctypes.c_size_t,
        ]
        lib.qml_sqlite_put_tokens.restype = ctypes.c_int

        lib.qml_sqlite_get_tokens.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.POINTER(QmlTokenRec)),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.qml_sqlite_get_tokens.restype = ctypes.c_int

        lib.qml_sqlite_free_tokens.argtypes = [ctypes.POINTER(QmlTokenRec)]
        lib.qml_sqlite_free_tokens.restype = None

        lib.qml_sqlite_last_pos.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32)]
        lib.qml_sqlite_last_pos.restype = ctypes.c_int

        lib.qml_sqlite_append_tokens.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_int32,
        ]
        lib.qml_sqlite_append_tokens.restype = ctypes.c_int

        lib.qml_sqlite_control_claim.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(QmlControlCmd),
        ]
        lib.qml_sqlite_control_claim.restype = ctypes.c_int

        lib.qml_sqlite_control_set_status.argtypes = [ctypes.c_int64, ctypes.c_char_p]
        lib.qml_sqlite_control_set_status.restype = ctypes.c_int

        lib.qml_sqlite_compact.argtypes = []
        lib.qml_sqlite_compact.restype = ctypes.c_int

    def open(self, sqlite_path, lmdb_path):
        """Opens the process-global SQLite+LMDB store.

        in: sqlite_path -- path to the SQLite metadata database file
        in: lmdb_path -- path to the LMDB blob store directory
        out: None; raises RuntimeError when qml_store_init fails
        side effects: initializes the process-global store connection
        """
        rc = self._lib.qml_store_init(
            sqlite_path.encode("utf-8"), lmdb_path.encode("utf-8")
        )
        if rc != 0:
            raise RuntimeError(f"qml_store_init failed (rc={rc})")
        self._opened = True

    def close(self):
        """Shuts the process-global store down if it was opened.

        in: none
        out: None
        side effects: closes the SQLite+LMDB connections
        """
        if self._opened:
            self._lib.qml_store_close()
            self._opened = False

    def __enter__(self):
        """Context-manager entry.

        in: none
        out: this QmlStore instance
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """Context-manager exit: closes the store.

        in: exc_type -- exception class if the with-block raised, else None
        in: exc -- exception instance or None
        in: tb -- traceback or None
        out: None (exceptions are never suppressed)
        side effects: closes the SQLite+LMDB connections
        """
        self.close()

    # ------------------------------------------------------------
    # block storage: raw content-hash-keyed bytes, direct LMDB passthrough.
    # no session concept here on purpose -- dedup is inherent to using the
    # same key for identical content, regardless of which session touches it.
    # ------------------------------------------------------------
    def block_put(self, key: str, data: bytes) -> None:
        """Stores one blob under key and records its LRU accounting row.

        in: key -- content-hash string key
        in: data -- blob bytes to store
        out: None; raises RuntimeError on LMDB write failure
        side effects: one durable LMDB commit plus one SQLite recency/size row
        """
        buf = ctypes.create_string_buffer(data, len(data))
        rc = self._lib.qml_lmdb_put(key.encode("utf-8"), buf, len(data))
        if rc != 0:
            raise RuntimeError(f"qml_lmdb_put failed for key={key!r} (rc={rc})")
        # LRU accounting (task #56): without this the store has no size or
        # age information at all and nothing can ever be reclaimed
        self._lib.qml_sqlite_block_touch(key.encode("utf-8"), len(data))

    def block_get(self, key: str):
        """Returns the stored bytes for key, or None if not found.

        in: key -- content-hash string key
        out: bytes copy of the stored blob, or None on a miss
        side effects: refreshes the block's LRU recency row on a hit
        """
        out_data = ctypes.c_void_p()
        out_size = ctypes.c_size_t()
        out_txn = ctypes.c_void_p()

        rc = self._lib.qml_lmdb_get(
            key.encode("utf-8"),
            ctypes.byref(out_data),
            ctypes.byref(out_size),
            ctypes.byref(out_txn),
        )
        if rc == 1:
            return None
        if rc != 0:
            raise RuntimeError(f"qml_lmdb_get failed for key={key!r} (rc={rc})")

        try:
            # refresh recency on a HIT, not just on write -- otherwise eviction
            # would be FIFO and would evict exactly the hot shared prefixes
            self._lib.qml_sqlite_block_touch(key.encode("utf-8"), out_size.value)
            return ctypes.string_at(out_data, out_size.value)
        finally:
            self._lib.qml_lmdb_get_done(out_txn)

    def blocks_put_from(self, keys, src_ptr: int, stride: int) -> None:
        """Writes len(keys) equally sized blobs from memory at src_ptr (blob i
        at src_ptr + i*stride) in ONE LMDB write txn, with the LRU accounting
        for the batch in one SQLite transaction on the background writer.

        The save side of the same problem blocks_get_into() solves: block_put()
        per block costs one durable LMDB commit plus one SQLite commit EACH,
        and on the vLLM path that runs inside the engine's forward pass.

        in: keys -- per-blob string keys; blob i is stored under keys[i]
        in: src_ptr -- address of the first blob's bytes (caller-owned memory
            that must stay alive for the duration of the call)
        in: stride -- bytes per blob; blob i starts at src_ptr + i*stride
        out: None; raises RuntimeError on write failure
        side effects: one durable LMDB commit covering all blobs; LRU recency
            accounting is handed to the background touch writer
        """
        if not keys:
            return
        arr = (ctypes.c_char_p * len(keys))(*[k.encode("utf-8") for k in keys])
        rc = self._lib.qml_lmdb_put_many(arr, len(keys), ctypes.c_void_p(src_ptr), stride)
        if rc != 0:
            raise RuntimeError("qml_lmdb_put_many failed")
        self._touch_async(arr, len(keys), stride)

    def blocks_get_into(self, keys, dest_ptr: int, stride: int, threads: int = 1) -> int:
        """Reads keys under ONE read txn straight into memory at dest_ptr,
        block i at dest_ptr + i*stride. Returns how many were copied
        contiguously from index 0 -- a miss or a wrong-sized blob stops the
        run there, same policy as block_get() returning None.

        dest_ptr must point at n*stride writable bytes the caller owns (in
        the connector it is a pinned host tensor). This is the read side of
        task #65: the per-key path costs a txn begin/abort, a SQLite touch
        and two host copies PER BLOCK, which on a 14,000-block restore is
        seconds of overhead on top of a transfer that should be one memcpy.

        in: keys -- string keys to read, in order
        in: dest_ptr -- address of len(keys)*stride writable caller-owned bytes
        in: stride -- expected byte size of every blob; a differently sized
            blob ends the run (the format/shape mismatch guard)
        in: threads -- parallel reader count (1 = one read txn on this thread)
        out: number of blobs copied contiguously from index 0 (a miss or a
            wrong-sized blob stops the count there)
        side effects: queues LRU recency updates for the copied prefix on the
            background touch writer
        """
        threads = max(1, min(threads, len(keys)))
        arr = (ctypes.c_char_p * len(keys))(*[k.encode("utf-8") for k in keys])

        if threads == 1:
            n = self._lib.qml_lmdb_get_many(arr, len(keys), ctypes.c_void_p(dest_ptr), stride)
            if n < 0:
                raise RuntimeError("qml_lmdb_get_many failed")
        else:
            # ctypes drops the GIL around the C call, so these really run in
            # parallel. Most of the read is not the memcpy itself but faulting
            # LMDB's mmap pages in (~200k minor faults for a 900 MiB session),
            # and that parallelises.
            span = (len(keys) + threads - 1) // threads
            chunks = [(off, min(span, len(keys) - off)) for off in range(0, len(keys), span)]

            def _read(chunk):
                """Reads one contiguous key chunk in its own LMDB read txn.

                in: chunk -- (offset, count) slice of the key array
                out: number of blobs qml_lmdb_get_many copied for this chunk
                """
                off, cnt = chunk
                sub = (ctypes.c_char_p * cnt)(*arr[off:off + cnt])
                return self._lib.qml_lmdb_get_many(
                    sub, cnt, ctypes.c_void_p(dest_ptr + off * stride), stride)

            counts = list(self._pool(threads).map(_read, chunks))
            if any(c < 0 for c in counts):
                raise RuntimeError("qml_lmdb_get_many failed")
            # only a run starting at index 0 is usable: stop at the first
            # chunk that did not come back complete
            n = 0
            for (off, cnt), got in zip(chunks, counts):
                n += got
                if got < cnt:
                    break

        if n:
            # Recency is bookkeeping for eviction, not correctness, so it must
            # not sit in the read's critical path: a restore touches thousands
            # of blocks and even batched into one transaction those UPDATEs
            # cost as much as the KV read itself. Handed to a single
            # background writer instead; if it falls behind, the touch is
            # dropped rather than made to block (worst case a block looks
            # colder than it is and gets evicted earlier).
            self._touch_async(arr, int(n), stride)
        return int(n)

    def _touch_async(self, arr, n: int, stride: int) -> None:
        """Hands a batch recency update to the background touch thread (drop-on-full).

        in: arr -- ctypes c_char_p key array (kept alive by the queue reference)
        in: n -- number of leading keys in arr to touch
        in: stride -- byte size recorded per block
        out: None; the update is silently dropped if the queue is full (a block
            then merely looks colder than it is and may be evicted earlier)
        side effects: lazily starts the qml-touch daemon thread; SQLite recency
            rows are eventually updated off the read/write critical path
        """
        with self._pool_lock:
            if self._touch_q is None:
                self._touch_q = queue.Queue(maxsize=64)
                threading.Thread(target=self._touch_worker, name="qml-touch",
                                 daemon=True).start()
        try:
            self._touch_q.put_nowait((arr, n, stride))
        except queue.Full:
            pass

    def _touch_worker(self) -> None:
        """Touch-thread body: applies queued batch recency updates to SQLite.

        in: none
        out: None (never returns; daemon thread)
        side effects: writes block recency/size rows via
            qml_sqlite_block_touch_many; all errors are swallowed so
            bookkeeping can never kill the server
        """
        while True:
            arr, n, stride = self._touch_q.get()
            try:
                self._lib.qml_sqlite_block_touch_many(arr, n, stride)
            except Exception:  # never let bookkeeping kill the server
                pass

    def _pool(self, threads: int):
        """Returns the shared reader thread pool, growing it when more workers are needed.

        in: threads -- minimum worker count required
        out: ThreadPoolExecutor with at least `threads` workers (lazily created;
            a smaller existing pool is replaced)
        """
        with self._pool_lock:
            if self._pool_obj is None or self._pool_threads < threads:
                if self._pool_obj is not None:
                    self._pool_obj.shutdown(wait=False)
                self._pool_obj = ThreadPoolExecutor(max_workers=threads,
                                                    thread_name_prefix="qml-read")
                self._pool_threads = threads
            return self._pool_obj

    def blocks_get_scatter(self, keys, dst_ptrs, sizes, threads: int = 1) -> int:
        """One read txn per worker for the whole batch; blob i is copied to
        dst_ptrs[i] (must hold sizes[i] writable bytes). Returns the count
        copied contiguously from index 0 -- a miss or size mismatch stops the
        run, matching every other batched read here. Recency accounting for
        the copied prefix goes to the background writer.

        threads > 1 splits the key range across workers exactly like
        blocks_get_into(): ctypes drops the GIL around the C call and most of
        the read is faulting LMDB's mmap pages in, which parallelises. LMDB
        read txns are concurrent by design, one per worker.

        in: keys -- string keys to read, in order
        in: dst_ptrs -- destination address for blob i at dst_ptrs[i]
        in: sizes -- expected byte size of blob i at sizes[i]
        in: threads -- parallel reader count
        out: number of blobs copied contiguously from index 0 (a miss or a
            size mismatch stops the count there)
        side effects: queues LRU recency updates for the copied prefix on the
            background touch writer
        """
        n = len(keys)
        if n == 0:
            return 0
        arr = (ctypes.c_char_p * n)(*[k.encode("utf-8") for k in keys])
        threads = max(1, min(threads, n))
        if threads == 1:
            ptrs = (ctypes.c_void_p * n)(*dst_ptrs)
            szs = (ctypes.c_size_t * n)(*sizes)
            got = self._lib.qml_lmdb_get_scatter(arr, n, ptrs, szs)
            if got < 0:
                raise RuntimeError("qml_lmdb_get_scatter failed")
        else:
            span = (n + threads - 1) // threads
            chunks = [(off, min(span, n - off)) for off in range(0, n, span)]

            def _read(chunk):
                """Reads one contiguous key chunk via qml_lmdb_get_scatter in its own read txn.

                in: chunk -- (offset, count) slice of the key/pointer/size arrays
                out: number of blobs copied for this chunk
                """
                off, cnt = chunk
                sub = (ctypes.c_char_p * cnt)(*arr[off:off + cnt])
                sp = (ctypes.c_void_p * cnt)(*dst_ptrs[off:off + cnt])
                ss = (ctypes.c_size_t * cnt)(*sizes[off:off + cnt])
                return self._lib.qml_lmdb_get_scatter(sub, cnt, sp, ss)

            counts = list(self._pool(threads).map(_read, chunks))
            if any(c < 0 for c in counts):
                raise RuntimeError("qml_lmdb_get_scatter failed")
            got = 0
            for (off, cnt), g in zip(chunks, counts):
                got += g
                if g < cnt:
                    break
        if got:
            self._touch_async(arr, int(got), int(sizes[0]))
        return int(got)

    def blocks_put_gather(self, keys, src_ptrs, sizes) -> None:
        """One write txn + one commit for the whole batch; blob i is read
        from src_ptrs[i] (sizes[i] bytes).

        in: keys -- per-blob string keys
        in: src_ptrs -- source address for blob i at src_ptrs[i]
        in: sizes -- byte size of blob i at sizes[i]
        out: None; raises RuntimeError on write failure
        side effects: one durable LMDB commit covering all blobs; LRU recency
            accounting handed to the background touch writer
        """
        n = len(keys)
        if n == 0:
            return
        arr = (ctypes.c_char_p * n)(*[k.encode("utf-8") for k in keys])
        ptrs = (ctypes.c_void_p * n)(*src_ptrs)
        szs = (ctypes.c_size_t * n)(*sizes)
        if self._lib.qml_lmdb_put_gather(arr, n, ptrs, szs) != 0:
            raise RuntimeError("qml_lmdb_put_gather failed")
        self._touch_async(arr, n, int(sizes[0]))

    def blocks_exist_run(self, keys) -> int:
        """How many of keys are present, counting from index 0 and stopping at
        the first miss, in ONE read txn. block_exists() in a loop asks the same
        question with one transaction per key, which on a few hundred blocks
        costs more than the answer is worth.

        in: keys -- string keys to probe, in order
        out: count of consecutive keys present, starting at index 0 and
            stopping at the first miss
        """
        if not keys:
            return 0
        arr = (ctypes.c_char_p * len(keys))(*[k.encode("utf-8") for k in keys])
        n = self._lib.qml_lmdb_exists_run(arr, len(keys))
        if n < 0:
            raise RuntimeError("qml_lmdb_exists_run failed")
        return int(n)

    def block_exists(self, key: str) -> bool:
        """Like block_get(key) is not None, but skips the string_at() copy --
        matters on hit-rate check paths (e.g. SGLang's batch_exists) that may
        be called once per candidate page per request, well before any of
        that page's bytes are actually needed.

        in: key -- content-hash string key
        out: True if the key is present in LMDB, False otherwise
        """
        out_data = ctypes.c_void_p()
        out_size = ctypes.c_size_t()
        out_txn = ctypes.c_void_p()

        rc = self._lib.qml_lmdb_get(
            key.encode("utf-8"),
            ctypes.byref(out_data),
            ctypes.byref(out_size),
            ctypes.byref(out_txn),
        )
        if rc == 1:
            return False
        if rc != 0:
            raise RuntimeError(f"qml_lmdb_get failed for key={key!r} (rc={rc})")
        self._lib.qml_lmdb_get_done(out_txn)
        return True

    def block_delete(self, key: str) -> None:
        """Deletes one blob from LMDB.

        in: key -- content-hash string key
        out: None; raises RuntimeError on failure (including a missing key)
        side effects: removes the blob from the LMDB store
        """
        rc = self._lib.qml_lmdb_delete(key.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"qml_lmdb_delete failed for key={key!r} (rc={rc})")

    def block_total_bytes(self) -> int:
        """Returns the SQLite-tracked total byte size of all stored KV blocks.

        in: none
        out: total tracked bytes as an int
        """
        total = ctypes.c_int64()
        rc = self._lib.qml_sqlite_block_total_bytes(ctypes.byref(total))
        if rc != 0:
            raise RuntimeError(f"qml_sqlite_block_total_bytes failed (rc={rc})")
        return int(total.value)

    def block_evict_lru(self, max_bytes: int) -> int:
        """Evicts coldest blocks until the tracked total fits max_bytes.
        The C side removes the accounting rows and returns the keys; deleting
        the actual LMDB bytes is done here (that split keeps the SQLite layer
        free of blob-store dependencies). Returns how many blocks were
        dropped. max_bytes <= 0 is a no-op.

        in: max_bytes -- byte budget the tracked total must fit within
        out: number of blocks dropped (0 when already within budget)
        side effects: deletes the coldest blocks' accounting rows from SQLite
            and their blobs from LMDB
        """
        if max_bytes <= 0:
            return 0
        keys_ptr = ctypes.POINTER(ctypes.c_char_p)()
        n = ctypes.c_size_t()
        rc = self._lib.qml_sqlite_block_evict_lru(
            ctypes.c_int64(max_bytes), ctypes.byref(keys_ptr), ctypes.byref(n)
        )
        if rc != 0:
            raise RuntimeError(f"qml_sqlite_block_evict_lru failed (rc={rc})")
        count = int(n.value)
        try:
            for i in range(count):
                raw = keys_ptr[i]
                if raw is None:
                    continue
                # a missing LMDB key here is fine: the row is already gone, so
                # the only consequence is that this block is no longer tracked
                try:
                    self.block_delete(raw.decode("utf-8"))
                except RuntimeError:
                    pass
        finally:
            if count:
                self._lib.qml_sqlite_block_evict_free(keys_ptr, n)
        return count

    # ------------------------------------------------------------
    # token log: qml_token_log, the SAME engine-agnostic table llama.cpp's
    # path already writes (storage/sqlite/qml_sqlite_schema.h). This is
    # the only genuinely portable "currency" across engines -- KV tensor
    # bytes are not (see docs/qml-router-technical-report.md) -- so a
    # cross-engine handoff (vLLM session continuing on llama.cpp or vice
    # versa) means reconstructing this token sequence and feeding it to
    # the target engine as a fresh prompt, never transferring KV bytes.
    # ------------------------------------------------------------
    def tokens_put(self, session_id: str, token_ids, seq_id: int = 0, start_pos: int = 0) -> None:
        """Writes token_ids as consecutive rows starting at start_pos (default
        0). Position-offset continuation across multiple engines touching the
        same session_id is NOT done here on purpose -- confirmed (user's own
        prior testing) that llama.cpp retains reasonable context even with
        non-continuous/reset positions, since token embeddings carry content
        regardless of position; only RoPE's relative-distance encoding is
        affected. Precise cross-engine pos continuity is a later refinement,
        not required for a first working cross-engine handoff.

        in: session_id -- session the rows belong to
        in: token_ids -- token ids written as consecutive qml_token_log rows
        in: seq_id -- sequence id recorded on every row (default 0)
        in: start_pos -- position of the first row (default 0)
        out: None; raises RuntimeError on insert failure
        side effects: bulk-inserts rows into the SQLite qml_token_log table
        """
        n = len(token_ids)
        if n == 0:
            return

        recs = (QmlTokenRec * n)()
        for i, tid in enumerate(token_ids):
            recs[i].pos = start_pos + i
            recs[i].seq_id = seq_id
            recs[i].token_id = int(tid)

        rc = self._lib.qml_sqlite_put_tokens(session_id.encode("utf-8"), recs, n)
        if rc != 0:
            raise RuntimeError(f"qml_sqlite_put_tokens failed for session_id={session_id!r} (rc={rc})")

    def get_last_pos(self, session_id: str):
        """Returns the session's current last_pos (int), or None if the
        session_id is entirely unknown. -1 means known but no tokens
        recorded yet (matches qml_session's own last_pos DEFAULT).

        in: session_id -- session to look up
        out: last_pos as int; -1 when known but empty; None when the
            session_id is unknown
        """
        out = ctypes.c_int32()
        rc = self._lib.qml_sqlite_last_pos(session_id.encode("utf-8"), ctypes.byref(out))
        if rc == 1:
            return None
        if rc != 0:
            raise RuntimeError(f"qml_sqlite_last_pos failed for session_id={session_id!r} (rc={rc})")
        return out.value

    def tokens_append(self, session_id: str, token_ids, seq_id: int = 0) -> None:
        """Cross-engine/incremental counterpart to tokens_put:
        appends token_ids starting right after this session's current
        last_pos, instead of always starting at pos 0. Use this (rather
        than tokens_put) when token_ids does NOT represent
        this engine's own complete view of the conversation -- e.g. a
        genuinely new turn's tokens only, continuing a session another
        engine already wrote rows for. See
        docs/udf-control-plane-proposal.md's position-offset-precision
        note.

        in: session_id -- session to append to
        in: token_ids -- new tokens, appended right after the current last_pos
        in: seq_id -- sequence id recorded on every row (default 0)
        out: None; raises RuntimeError on failure
        side effects: appends rows to qml_token_log and advances the
            session's last_pos
        """
        n = len(token_ids)
        if n == 0:
            return

        arr = (ctypes.c_int32 * n)(*[int(t) for t in token_ids])
        rc = self._lib.qml_sqlite_append_tokens(session_id.encode("utf-8"), arr, n, seq_id)
        if rc != 0:
            raise RuntimeError(f"qml_sqlite_append_tokens failed for session_id={session_id!r} (rc={rc})")

    def tokens_get(self, session_id: str):
        """Returns the full token_id list for session_id, ordered by pos
        (empty list if none). Backs the UDF control plane's vLLM RESTORE/
        PREWARM dispatch (hooks/vllm/qml_vllm_connector.py) -- it needs the
        known token sequence to prime via a self-issued completion call.

        in: session_id -- session whose token log is wanted
        out: list of token ids ordered by pos (empty list if none)
        """
        out_ptr = ctypes.POINTER(QmlTokenRec)()
        out_n = ctypes.c_size_t()
        rc = self._lib.qml_sqlite_get_tokens(session_id.encode("utf-8"), ctypes.byref(out_ptr), ctypes.byref(out_n))
        if rc != 0:
            raise RuntimeError(f"qml_sqlite_get_tokens failed for session_id={session_id!r} (rc={rc})")

        try:
            return [out_ptr[i].token_id for i in range(out_n.value)]
        finally:
            self._lib.qml_sqlite_free_tokens(out_ptr)

    # ------------------------------------------------------------
    # session manifest: qml_session (base) + qml_vllm_session_ext, for the
    # management plane (qml_manager.py) -- NOT on the KV load/save hot path.
    # ------------------------------------------------------------
    def session_touch(
        self,
        session_id: str,
        user_id: str,
        model_hash: str,
        lmdb_prefix: str,
        block_size: int,
        num_blocks: int,
        seq_id: int = 0,
        n_tokens: int = 0,
        engine_type: int = QML_ENGINE_VLLM,
    ) -> None:
        """Upserts this session's manifest row (qml_session base + vLLM extension).

        in: session_id -- session key
        in: user_id -- user recorded on the row (may be empty)
        in: model_hash -- storage identity (cache seed) a later restore must match
        in: lmdb_prefix -- LMDB key prefix recorded for the session
        in: block_size -- KV block size in tokens
        in: num_blocks -- total prompt blocks recorded for the session
        in: seq_id -- sequence id (default 0)
        in: n_tokens -- token count recorded on the row (default 0)
        in: engine_type -- engine tag (default QML_ENGINE_VLLM; see comment below)
        out: None; raises RuntimeError on upsert failure
        side effects: writes/updates the session row in SQLite
        """
        # engine_type is a parameter (default unchanged) because the SGLang
        # backend reuses this upsert for ITS session rows -- the row layout is
        # identical, only the engine tag differs (task #37).
        meta = QmlVllmSessionMeta()
        meta.base.user_id = user_id.encode("utf-8")[: _USER_ID_LEN - 1]
        meta.base.session_id = session_id.encode("utf-8")[: _SESSION_ID_LEN - 1]
        meta.base.engine_type = engine_type
        meta.base.seq_id = seq_id
        meta.base.last_token = -1
        meta.base.last_pos = -1
        meta.base.n_tokens = n_tokens
        meta.model_hash = model_hash.encode("utf-8")[: _MODEL_HASH_LEN - 1]
        meta.lmdb_prefix = lmdb_prefix.encode("utf-8")[: _LMDB_PREFIX_LEN - 1]
        meta.block_size = block_size
        meta.num_blocks = num_blocks

        rc = self._lib.qml_sqlite_put_vllm_session(session_id.encode("utf-8"), ctypes.byref(meta))
        if rc != 0:
            raise RuntimeError(f"qml_sqlite_put_vllm_session failed for session_id={session_id!r} (rc={rc})")

    # ------------------------------------------------------------
    # UDF control plane (see docs/udf-control-plane-proposal.md) -- for a
    # polling thread to call periodically, target_pid == os.getpid().
    # ------------------------------------------------------------
    def control_claim(self, target_pid: int):
        """Atomically claims (PENDING -> CLAIMED) the oldest pending
        command for target_pid, if any. Returns a dict, or None if
        nothing is pending.

        in: target_pid -- pid of this process (commands are addressed per pid)
        out: dict with cmd_id/target_engine/cmd_type/session_id/created_at,
            or None when no command is pending
        side effects: marks the claimed command row CLAIMED in SQLite
        """
        cmd = QmlControlCmd()
        rc = self._lib.qml_sqlite_control_claim(target_pid, ctypes.byref(cmd))
        if rc == 1:
            return None
        if rc != 0:
            raise RuntimeError(f"qml_sqlite_control_claim failed (rc={rc})")

        return {
            "cmd_id": cmd.cmd_id,
            "target_engine": cmd.target_engine.decode("utf-8"),
            "cmd_type": cmd.cmd_type.decode("utf-8"),
            "session_id": cmd.session_id.decode("utf-8"),
            "created_at": cmd.created_at,
        }

    def control_set_status(self, cmd_id: int, status: str) -> None:
        """Writes a claimed control command's final status.

        in: cmd_id -- id of the claimed command row
        in: status -- new status string (e.g. "DONE" or "FAILED")
        out: None; raises RuntimeError on failure
        side effects: updates the command row in SQLite
        """
        rc = self._lib.qml_sqlite_control_set_status(cmd_id, status.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"qml_sqlite_control_set_status failed for cmd_id={cmd_id} (rc={rc})")

    def compact(self) -> bool:
        """Runs a WAL checkpoint + VACUUM on the SQLite metadata db.
        Returns True on success. Blocks the caller for the duration --
        call from a background thread, not the request-serving path.

        in: none
        out: True on success, False otherwise
        side effects: checkpoints the WAL and VACUUMs the SQLite database
        """
        return self._lib.qml_sqlite_compact() == 0

    def session_get(self, session_id: str):
        """Returns a dict of fields, or None if not found.

        in: session_id -- session key to look up
        out: dict (session_id/user_id/engine_type/seq_id/n_tokens/model_hash/
            lmdb_prefix/block_size/num_blocks), or None when the session is unknown
        """
        meta = QmlVllmSessionMeta()
        rc = self._lib.qml_sqlite_get_vllm_session(session_id.encode("utf-8"), ctypes.byref(meta))
        if rc == 1:
            return None
        if rc != 0:
            raise RuntimeError(f"qml_sqlite_get_vllm_session failed for session_id={session_id!r} (rc={rc})")

        return {
            "session_id": meta.base.session_id.decode("utf-8"),
            "user_id": meta.base.user_id.decode("utf-8"),
            "engine_type": meta.base.engine_type,
            "seq_id": meta.base.seq_id,
            "n_tokens": meta.base.n_tokens,
            "model_hash": meta.model_hash.decode("utf-8"),
            "lmdb_prefix": meta.lmdb_prefix.decode("utf-8"),
            "block_size": meta.block_size,
            "num_blocks": meta.num_blocks,
        }

    # ------------------------------------------------------------
    # session layer: window trim (TAWT), archive, FTS index and the RIG
    # decision helpers of the C core. Used by a session layer in front of an
    # engine that has no hook of its own for them (router/qml_openai_gateway.py
    # for vLLM and SGLang). Signatures are declared on first use, so a library
    # built without these entry points still serves the connector paths above.
    # ------------------------------------------------------------
    def _session_layer(self):
        """Declares the session-layer entry points once and returns the library.

        in: none
        out: the ctypes library handle with the session-layer signatures set
        """
        if getattr(self, "_sl_ready", False):
            return self._lib
        lib = self._lib
        c_i64p = ctypes.POINTER(ctypes.c_int64)
        lib.qml_window_trim.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int32]
        lib.qml_window_trim.restype = ctypes.c_long
        lib.qml_sqlite_fts_status.argtypes = [ctypes.c_char_p]
        lib.qml_sqlite_fts_status.restype = ctypes.c_int
        lib.qml_sqlite_archive_fetch.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.qml_sqlite_archive_fetch.restype = ctypes.c_int
        lib.qml_sqlite_fts_add.argtypes = [
            ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_char_p,
        ]
        lib.qml_sqlite_fts_add.restype = ctypes.c_int
        lib.qml_sqlite_fts_begin.argtypes = []
        lib.qml_sqlite_fts_begin.restype = ctypes.c_int
        lib.qml_sqlite_fts_end.argtypes = [ctypes.c_int]
        lib.qml_sqlite_fts_end.restype = ctypes.c_int
        lib.qml_sqlite_fts_search.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
            c_i64p, c_i64p, ctypes.POINTER(ctypes.c_double),
        ]
        lib.qml_sqlite_fts_search.restype = ctypes.c_int
        lib.qml_sqlite_fts_text.argtypes = [ctypes.c_char_p, ctypes.c_int64]
        lib.qml_sqlite_fts_text.restype = ctypes.c_void_p
        lib.qml_sqlite_archive_range.argtypes = [
            ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.qml_sqlite_archive_range.restype = ctypes.c_int
        lib.qml_rig_query.argtypes = [ctypes.c_char_p]
        lib.qml_rig_query.restype = ctypes.c_void_p
        lib.qml_rig_n_words.argtypes = [ctypes.c_char_p]
        lib.qml_rig_n_words.restype = ctypes.c_int
        lib.qml_rig_gate.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ]
        lib.qml_rig_gate.restype = ctypes.c_int
        lib.qml_rig_match_count.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.qml_rig_match_count.restype = ctypes.c_int
        lib.qml_sqlite_owner_add.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t,
        ]
        lib.qml_sqlite_owner_add.restype = ctypes.c_int
        lib.qml_store_drop_kv.argtypes = [ctypes.c_char_p]
        lib.qml_store_drop_kv.restype = ctypes.c_long
        self._libc_free = ctypes.CDLL(None).free
        self._libc_free.argtypes = [ctypes.c_void_p]
        self._libc_free.restype = None
        self._sl_ready = True
        return lib

    def _take_str(self, ptr):
        """Copies a malloc'd C string into Python and frees it.

        in: ptr -- address returned by a C function (0 = NULL)
        out: str, or None for NULL
        """
        if not ptr:
            return None
        try:
            return ctypes.string_at(ptr).decode("utf-8", "replace")
        finally:
            self._libc_free(ptr)

    def window_trim(self, session_id: str, window: int, eot_token: int) -> int:
        """Applies the core TAWT (qml_window_trim) to one session.

        in: session_id -- session to consider
        in: window -- live-window token budget (0 disables)
        in: eot_token -- end-of-turn token id of the serving model
        out: tokens moved to the archive (0 = under budget or no fitting user-turn
            boundary); raises RuntimeError on a storage error
        side effects: archive insert, log renumbering and session counters in one
            transaction; drops the session's FTS rows so the next recall re-indexes
        """
        lib = self._session_layer()
        n = lib.qml_window_trim(session_id.encode("utf-8"), int(window), int(eot_token))
        if n < 0:
            raise RuntimeError(f"qml_window_trim failed for session_id={session_id!r}")
        return n

    def fts_status(self, session_id: str) -> int:
        """2 = nothing archived, 1 = FTS rows exist, 0 = archive not indexed yet, -1 = no FTS5."""
        return self._session_layer().qml_sqlite_fts_status(session_id.encode("utf-8"))

    def archive_fetch(self, session_id: str):
        """Returns (positions, token_ids) of every archived row in position order ([] when none)."""
        lib = self._session_layer()
        pos = ctypes.POINTER(ctypes.c_int32)()
        tok = ctypes.POINTER(ctypes.c_int32)()
        n = ctypes.c_size_t()
        if lib.qml_sqlite_archive_fetch(session_id.encode("utf-8"), ctypes.byref(pos),
                                        ctypes.byref(tok), ctypes.byref(n)) != 0:
            return [], []
        try:
            return pos[:n.value], tok[:n.value]
        finally:
            self._libc_free(ctypes.cast(pos, ctypes.c_void_p))
            self._libc_free(ctypes.cast(tok, ctypes.c_void_p))

    def fts_add(self, session_id: str, docs) -> int:
        """Indexes documents [(turn_pos, turn_end, text), ...] in one transaction; returns the count."""
        lib = self._session_layer()
        sid = session_id.encode("utf-8")
        lib.qml_sqlite_fts_begin()
        n = 0
        ok = False
        try:
            for lo, hi, text in docs:
                if lib.qml_sqlite_fts_add(sid, int(lo), int(hi), text.encode("utf-8")) == 0:
                    n += 1
            ok = True
        finally:
            lib.qml_sqlite_fts_end(1 if ok else 0)
        return n

    def fts_search(self, session_id: str, match_query: str, k: int):
        """FTS5 search scoped to this session; returns [(turn_pos, turn_end, bm25), ...] best first."""
        lib = self._session_layer()
        k = max(1, int(k))
        tp = (ctypes.c_int64 * k)()
        te = (ctypes.c_int64 * k)()
        sc = (ctypes.c_double * k)()
        n = lib.qml_sqlite_fts_search(session_id.encode("utf-8"), match_query.encode("utf-8"),
                                              k, tp, te, sc)
        return [(tp[i], te[i], sc[i]) for i in range(max(0, n))]

    def fts_text(self, session_id: str, turn_pos: int):
        """Indexed text of one archived document, or None."""
        lib = self._session_layer()
        return self._take_str(lib.qml_sqlite_fts_text(session_id.encode("utf-8"), int(turn_pos)))

    def archive_range(self, session_id: str, pos_start: int, pos_end: int):
        """Archived token ids of original positions [pos_start, pos_end) ([] on error or empty)."""
        lib = self._session_layer()
        out = ctypes.POINTER(ctypes.c_int32)()
        n = ctypes.c_size_t()
        if lib.qml_sqlite_archive_range(session_id.encode("utf-8"), int(pos_start), int(pos_end),
                                        ctypes.byref(out), ctypes.byref(n)) != 0:
            return []
        try:
            return out[:n.value]
        finally:
            self._libc_free(ctypes.cast(out, ctypes.c_void_p))

    def rig_query(self, text: str) -> str:
        """FTS5 OR-query of the text's content words (core qml_rig_query); '' when none."""
        lib = self._session_layer()
        return self._take_str(lib.qml_rig_query(text.encode("utf-8"))) or ""

    def rig_n_words(self, text: str) -> int:
        """Effective word count of a question (core qml_rig_n_words)."""
        return self._session_layer().qml_rig_n_words(text.encode("utf-8"))

    def rig_gate(self, score: float, best: float, n_eff: int, n_min: int, conf: float) -> bool:
        """RIG injection verdict for one candidate (core qml_rig_gate)."""
        return self._session_layer().qml_rig_gate(score, best, n_eff, n_min, conf) == 1

    def rig_match_count(self, question: str, candidate: str) -> int:
        """Number of the question's query words found in the candidate (core qml_rig_match_count)."""
        return self._session_layer().qml_rig_match_count(
            question.encode("utf-8"), (candidate or "").encode("utf-8"))

    def owner_add(self, owner: str, keys) -> None:
        """Records that owner stored these blocks, so a window trim can drop them.

        in: owner -- session id
        in: keys -- block keys as written to LMDB
        out: None; raises RuntimeError on failure
        """
        if not owner or not keys:
            return
        lib = self._session_layer()
        arr = (ctypes.c_char_p * len(keys))(*[k.encode("utf-8") for k in keys])
        if lib.qml_sqlite_owner_add(owner.encode("utf-8"), arr, len(keys)) != 0:
            raise RuntimeError(f"qml_sqlite_owner_add failed for owner={owner!r}")

    def kv_drop(self, owner: str) -> int:
        """Drops the KV blocks only this session stored (core qml_store_drop_kv).

        in: owner -- session id
        out: number of LMDB values deleted
        """
        n = self._session_layer().qml_store_drop_kv(owner.encode("utf-8"))
        if n < 0:
            raise RuntimeError(f"qml_store_drop_kv failed for owner={owner!r}")
        return n
