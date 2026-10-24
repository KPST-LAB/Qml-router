# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
"""hooks/sglang/qml_sglang_storage.py

QmlSglangStorage: a HiCacheStorage backend (SGLang's public external-KV-cache
plugin interface -- python/sglang/srt/mem_cache/hicache_storage.py) backed by
qml-router's existing LMDB block store.

Verified (not assumed) against a real clone of SGLang's source before writing
this file:
  - HiCacheStorage is a real, documented ABC. Backends are loaded dynamically
    by python/sglang/srt/mem_cache/storage/backend_factory.py via
    --hicache-storage-backend dynamic --hicache-storage-backend-extra-config
    '{"module_path": "qml_sglang_storage", "class_name": "QmlSglangStorage",
      "qml_sqlite_path": "...", "qml_lmdb_path": "..."}' -- no SGLang source
    patch needed (same conclusion as vLLM's KVConnectorBase_V1, unlike
    llama.cpp which genuinely required one).
  - Unlike vLLM, WE DO NOT compute the block/page hash ourselves. SGLang's own
    radix cache already derives a chained page hash (parent_hash + token_ids,
    see mem_cache/utils.py get_hash_str / compute_node_hash_values) and hands
    it to us as an opaque `key: str` in every get/set/exists call. There is no
    block-chain slicing to (re)implement here -- SGLang already did it.
  - `cache_salt`/`extra_key` are ALSO already folded into that hash upstream
    (schedule_batch.py Req.extra_key -> RadixKey.extra_key participates in the
    hash), so two sessions sharing a token prefix but different cache_salt
    naturally get different keys without any action on our part -- unlike
    vLLM, where we had to capture request.cache_salt ourselves via
    on_new_request() because NewRequestData didn't carry it.
  - The concrete get/set contract below (byte-copy into/out of a pre-allocated
    target_location tensor via a torch.uint8 view) mirrors SGLang's own
    built-in reference backend, HiCacheFile
    (mem_cache/hicache_storage.py:359 class HiCacheFile) -- read directly
    rather than guessed, since HiCacheStorage's docstrings alone don't pin
    down the exact tensor-view convention.
  - NOT yet resolved: this backend has no session_id/user_id available at
    these call sites (see storage/sqlite/qml_sqlite_schema.h's
    QML_DDL_CREATE_SGLANG_EXT comment) -- so unlike hooks/vllm/qml_vllm_connector.py,
    this file does NOT write qml_session/qml_sglang_session_ext or
    qml_token_log. Manifest/token-log visibility for SGLang needs a separate
    hook at SGLang's OpenAI-compatible serving layer, not here. Tracked as
    qml-router task #37.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, List, Optional

import torch

# qml_store_bindings.py is engine-agnostic despite living under hooks/vllm/
# (it wraps libqml_session_store.so's generic LMDB block_put/get/exists and
# SQLite session-manifest calls) -- reused as-is rather than duplicated.
# Follow-up: move it to a shared location (e.g. hooks/common/) instead of
# reaching across engine directories; not done here to avoid touching the
# already-verified-live vLLM import path in the same change.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from qml_store_bindings import QML_ENGINE_SGLANG, QmlStore  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig

logger = logging.getLogger(__name__)


def _host_contig(t: torch.Tensor) -> torch.Tensor:
    """CPU + contiguous guarantee before handing a raw data_ptr to C.

    SGLang's host pool hands out VIEWS (non-contiguous strides), and a view's
    data_ptr() points at the view's first element while numel*element_size
    walks PAST interleaved foreign bytes; a CUDA tensor's data_ptr is a device
    address a host memcpy segfaults on -- both observed live (scheduler died
    with exit code -11 on the first batched run). The copy this makes for
    non-contiguous inputs is the same copy _tensor_view_bytes always made.

    in: t — torch.Tensor of any device/layout (typically an SGLang host-pool view)
    out: torch.Tensor guaranteed CPU-resident and contiguous — the same object
         when t already satisfies both, otherwise a fresh copy"""
    if t.device.type != "cpu":
        t = t.detach().cpu()
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def _tensor_view_bytes(t: torch.Tensor) -> memoryview:
    """Raw-byte view of a tensor's storage, regardless of dtype -- reinterprets
    bits via torch.uint8 rather than going through .numpy() on the original
    dtype. This sidesteps the exact bug we hit in the vLLM connector
    (NumPy has no bfloat16, see hooks/vllm/qml_vllm_connector.py's
    _tensor_to_bytes): viewing as uint8 first works for every dtype, since it
    never asks NumPy to represent the original dtype at all.

    in: t — torch.Tensor (CPU-resident, any dtype) whose storage bytes are wanted
    out: memoryview over the tensor's raw bytes (numel * element_size long);
         writable, so it also serves as a copy-into target"""
    return memoryview(t.view(torch.uint8).contiguous().numpy())


class QmlSglangStorage(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig, _factory_kwargs: Optional[dict] = None):
        """Open the qml SQLite/LMDB store and install the identity recorder.

        in: storage_config — SGLang HiCacheStorageConfig; its extra_config dict
            must carry qml_sqlite_path and qml_lmdb_path (qml_read_threads
            optional, default 4)
        in: _factory_kwargs — backend_factory's own **kwargs; always passed by
            SGLang but intentionally unused (see comment below)
        out: None (constructor); raises ValueError when either path is missing
        side effects: opens the SQLite/LMDB store; monkey-patches
            HiRadixCache.write_backup_storage for session-identity recording
        """
        # Verified against real installed SGLang source
        # (mem_cache/storage/backend_factory.py):
        #   backend_config = storage_config.extra_config   # our FULL
        #                                                   # --hicache-storage-backend-extra-config dict
        #   ...
        #   return backend_class(storage_config, kwargs)    # `kwargs` here is
        #                                                    # create_backend()'s OWN **kwargs
        #                                                    # (a separate, unrelated, empty dict
        #                                                    # in practice) -- NOT backend_config.
        # So storage_config.extra_config (not the 2nd positional arg) is the
        # right place to read qml_sqlite_path/qml_lmdb_path from. An earlier
        # pass at this file misread `kwargs` as being backend_config and
        # "fixed" this incorrectly without re-tracing the actual call site --
        # corrected after a live crash showed the previous version's own
        # ValueError text. The 2nd positional param is still accepted (SGLang
        # always passes it) but intentionally unused.
        extra = storage_config.extra_config or {}
        sqlite_path = extra.get("qml_sqlite_path")
        lmdb_path = extra.get("qml_lmdb_path")
        if not sqlite_path or not lmdb_path:
            raise ValueError(
                "QmlSglangStorage requires qml_sqlite_path and qml_lmdb_path in "
                "--hicache-storage-backend-extra-config"
            )
        self._store = QmlStore()
        self._store.open(sqlite_path, lmdb_path)

        # task #37: session identity + token log. The storage ABC only ever
        # sees opaque page hashes, so the identity is captured one layer up,
        # where it still exists in plaintext (RadixKey.extra_key + token_ids).
        self._model_hash = (getattr(storage_config, "model_name", "") or "")[:64]
        self._identity_warned = False
        self._hook_identity()
        # parallel readers for batch_get; same rationale as the vLLM
        # connector's qml_read_threads (mmap page faults parallelise)
        self._read_threads = int(extra.get("qml_read_threads", 4) or 4)
        # qml_token_log: "on" (default) records the token sequence of backed-up
        # nodes; "off" leaves qml_token_log to a session layer in front of the
        # engine (router/qml_openai_gateway.py), whose prompts also carry recalled
        # archive turns. The session row is recorded either way.
        self._record_token_log = str(extra.get("qml_token_log", "on")).lower() != "off"

    def get(
        self,
        key: str,
        target_location: Optional[torch.Tensor] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """Fetch one page's KV bytes into the caller's pre-allocated tensor.

        in: key — SGLang's opaque chained page hash
        in: target_location — pre-allocated tensor to fill; required (a call
            without it is logged and treated as a miss)
        in: target_sizes — unused (HiCacheStorage ABC signature compatibility)
        out: target_location filled with the stored bytes, or None on miss /
             missing target_location / stored-vs-expected size mismatch
        """
        data = self._store.block_get(key)
        if data is None:
            return None
        if target_location is None:
            logger.warning("QmlSglangStorage.get called without target_location for key=%s", key)
            return None
        expected = target_location.numel() * target_location.element_size()
        if len(data) != expected:
            logger.warning(
                "QmlSglangStorage.get size mismatch for key=%s: stored=%d expected=%d",
                key, len(data), expected,
            )
            return None
        _tensor_view_bytes(target_location)[:] = data
        return target_location

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        """Batched get: fetch many pages in one LMDB read transaction.

        in: keys — page hashes in radix order (index 0 closest to the root)
        in: target_locations — one pre-allocated tensor per key; if any entry
            is None, falls back to per-key get()
        in: target_sizes — unused (HiCacheStorage ABC signature compatibility)
        out: list aligned with keys — the filled tensor for each fetched key
             (a contiguous run from index 0), None from the first miss onward
        """
        # ONE LMDB read txn for the whole batch (task #67). The per-key loop
        # this replaces cost a txn + a python bytes copy per PAGE, and with
        # SGLang's default page_size=1 that meant one round trip per TOKEN --
        # measured 3x slower than recomputing the prompt on the GPU.
        locs = target_locations or [None] * len(keys)
        if any(loc is None for loc in locs):
            return [self.get(k, loc) for k, loc in zip(keys, locs)]
        # keep the normalized tensors referenced until the C call returns
        t0 = time.perf_counter()
        staged = [_host_contig(loc) for loc in locs]
        got = self._store.blocks_get_scatter(
            keys,
            [t.data_ptr() for t in staged],
            [t.numel() * t.element_size() for t in staged],
            threads=self._read_threads,
        )
        if os.environ.get("QML_LOAD_TIMING") == "1":
            el = (time.perf_counter() - t0) * 1e3
            mb = sum(t.numel() * t.element_size() for t in staged[:max(got, 0)]) / 1048576
            logger.warning("qml sglang batch_get: %.0f ms, %d/%d keys, %.1f MiB",
                           el, got, len(keys), mb)
        # a staged tensor that had to be copied is not the caller's buffer --
        # copy the fetched bytes back into the real target
        for i in range(got):
            if staged[i] is not locs[i]:
                locs[i].copy_(staged[i])
        return [locs[i] if i < got else None for i in range(len(keys))]

    def set(
        self,
        key: str,
        value: Optional[torch.Tensor] = None,
        target_location: Optional[torch.Tensor] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """Store one page's KV bytes under its content-addressed hash.

        in: key — SGLang's opaque chained page hash
        in: value — source tensor (preferred when both are given)
        in: target_location — fallback source when value is None
        in: target_sizes — unused (HiCacheStorage ABC signature compatibility)
        out: True when the bytes are stored or already present (identical key
             implies identical bytes), False when no source was given or the
             LMDB write failed
        """
        src = value if value is not None else target_location
        if src is None:
            logger.warning("QmlSglangStorage.set called with no value/target_location for key=%s", key)
            return False
        if self._store.block_exists(key):
            return True  # content-addressed: identical key implies identical bytes already
        try:
            self._store.block_put(key, bytes(_tensor_view_bytes(src)))
            return True
        except RuntimeError:
            logger.warning("QmlSglangStorage.set failed for key=%s", key, exc_info=True)
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[List[torch.Tensor]] = None,
        target_locations: Optional[List[torch.Tensor]] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """Batched set: persist many pages in one LMDB write transaction.

        in: keys — page hashes to store under
        in: values — source tensors (preferred when both lists are given)
        in: target_locations — fallback sources when values is None
        in: target_sizes — unused (HiCacheStorage ABC signature compatibility)
        out: True when every page was written (falls back to per-key set() if
             any source entry is None), False on a write failure
        """
        # ONE write txn + ONE commit for the whole batch (task #67); the
        # replaced loop committed once per page. Content-addressed keys make
        # overwrites harmless, so no per-key exists() pre-check either.
        srcs = values or target_locations or [None] * len(keys)
        if any(v is None for v in srcs):
            return all(self.set(k, v) for k, v in zip(keys, srcs))
        try:
            staged = [_host_contig(v) for v in srcs]
            self._store.blocks_put_gather(
                keys,
                [t.data_ptr() for t in staged],
                [t.numel() * t.element_size() for t in staged],
            )
            return True
        except RuntimeError:
            logger.warning("QmlSglangStorage.batch_set failed", exc_info=True)
            return False

    def exists(self, key: str) -> bool:
        """Report whether one page hash is already stored.

        in: key — SGLang's opaque chained page hash
        out: True if the LMDB block exists, False otherwise
        """
        return self._store.block_exists(key)

    # ------------------------------------------------------------
    # task #37: session identity + token log via the hicache write path.
    #
    # The HiCacheStorage ABC receives only opaque page hashes, so by the time
    # set()/batch_set() run, the session identity is gone. One layer up,
    # HiRadixCache.write_backup_storage(node) still holds it in plaintext:
    # node.key is a RadixKey carrying BOTH token_ids and extra_key (the value
    # SGLang's OpenAI layer builds from the request's cache_salt/extra_key
    # fields -- serving_base._compute_extra_key). Wrapping that one method at
    # backend-construction time records qml_session + qml_token_log rows
    # WITHOUT patching SGLang source, keeping the plugin-only contract.
    #
    # Completeness argument: under the write-through policy, backed-up nodes
    # form a gapless prefix from the root (invariant asserted in
    # write_backup()), so accumulating each node's segment at its ancestor
    # offset yields the full token sequence. INSERT OR REPLACE keying makes
    # re-backups idempotent.
    #
    # Version safety: every access is guarded; on any mismatch the recorder
    # logs once and records nothing -- KV storage itself is unaffected.
    # ------------------------------------------------------------
    def _hook_identity(self) -> None:
        """Wrap the radix cache's write_backup_storage to record session identity.

        in: (none)
        out: None
        side effects: replaces write_backup_storage of HiRadixCache (SGLang up to
            0.5.20) or UnifiedRadixCache (SGLang 0.5.22, which takes a node id)
            with a wrapper that calls _record_node() first; idempotent (a
            second install is a no-op) and self-disabling when neither class
            exists -- KV storage itself is unaffected either way
        """
        target, by_id = None, False
        try:
            from sglang.srt.mem_cache.hiradix_cache import HiRadixCache as target
        except Exception:
            try:
                from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as target
                by_id = True
            except Exception as e:  # pragma: no cover - version drift
                logger.warning("qml: identity recorder disabled (no HiRadixCache or"
                               " UnifiedRadixCache: %s)", e)
                return
        orig = target.write_backup_storage
        if getattr(orig, "_qml_identity_wrapped", False):
            return
        backend = self

        def wrapped(cache_self, node, *args, **kwargs):
            """Record session identity, then run the original write_backup_storage.

            in: cache_self — the radix cache instance (original method's self)
            in: node — radix-tree node being backed up, or its id (UnifiedRadixCache)
            in: args / kwargs — passed through (backup_len on HiRadixCache)
            out: whatever the original write_backup_storage returns
            """
            try:
                if by_id:
                    backend._record_by_id(cache_self, node)
                else:
                    backup_len = args[0] if args else kwargs.get("backup_len")
                    backend._record_node(node, backup_len)
            except Exception as e:
                if not backend._identity_warned:
                    backend._identity_warned = True
                    logger.warning("qml: identity record skipped (%s) -- KV storage unaffected", e)
            return orig(cache_self, node, *args, **kwargs)

        wrapped._qml_identity_wrapped = True
        target.write_backup_storage = wrapped
        logger.info("qml: session identity recorder installed on %s.write_backup_storage",
                    target.__name__)

    def _record_by_id(self, cache, node_id) -> None:
        """Write qml_session + qml_token_log rows for one node of UnifiedRadixCache.

        SGLang 0.5.22 keeps the radix tree in a Rust core that exposes no node
        objects (node_by_id is not ported), so this reads the same facts through
        the tree core's own calls, which both the Python and the Rust core have.

        in: cache — the UnifiedRadixCache instance
        in: node_id — id of the node being backed up
        out: None; returns early recording nothing when the request carried no
            extra_key or cache_salt, or when the node has no backup spec
        side effects: upserts the qml_session row and bulk-inserts token-log
            rows at the node's absolute offset
        """
        core = cache.tree_core
        extra_key, cache_salt = core.prefetch_anchor_info(node_id)
        extra = extra_key or cache_salt
        if not extra:
            return
        spec = core.build_storage_backup_spec(node_id, False)
        if spec is None:
            return
        seg = list(spec.token_ids)
        # pages before this node = the prefix hash chain up to and including the
        # node minus the node's own pages
        n_prev = len(core.get_prefix_hash_values(node_id)) - len(core.get_hash_values(node_id))
        start_pos = max(0, n_prev) * int(getattr(cache, "page_size", 1) or 1)
        # the page hashes are the keys batch_set writes: record them as this
        # session's blocks so a window trim can drop them
        self._store.owner_add(extra, list(spec.hash_value))
        self._write_session(extra, seg, start_pos)

    def _record_node(self, node, backup_len) -> None:
        """Write qml_session + qml_token_log rows for one backed-up radix node.

        in: node — radix node whose RadixKey holds token_ids and extra_key
        in: backup_len — length SGLang reports for this backup; a mismatch with
            the node's own segment length marks a split-chain backup (skipped)
        out: None; returns early recording nothing when the request carried no
            extra_key or the backup is split-chain
        side effects: upserts the qml_session row and bulk-inserts token-log
            rows at the node's absolute ancestor offset
        """
        key = getattr(node, "key", None)
        # SGLang 0.5.20 and later keep the request's cache_salt apart from extra_key
        extra = getattr(key, "extra_key", None) or getattr(key, "cache_salt", None)
        if not extra:
            return  # request carried no cache_salt/extra_key -- nothing to record
        seg = list(getattr(key, "token_ids"))
        if backup_len is not None and backup_len != len(seg):
            # split-chain backup: the true range spans several nodes and is
            # reassembled inside SGLang (_concat_split_chain). Recording a
            # wrong offset would corrupt the log, so skip this rare shape.
            logger.debug("qml: skipping split-chain backup for %s", extra)
            return
        # absolute position of this node's segment = total length of ancestors
        start_pos = 0
        parent = getattr(node, "parent", None)
        while parent is not None and getattr(parent, "key", None) is not None:
            start_pos += len(getattr(parent.key, "token_ids", parent.key))
            parent = getattr(parent, "parent", None)

        hashes = getattr(node, "hash_value", None)
        if hashes:
            self._store.owner_add(extra, list(hashes))
        self._write_session(extra, seg, start_pos)

    def _write_session(self, extra, seg, start_pos) -> None:
        """Upsert the session row and the token-log rows of one backed-up segment."""
        user_id = extra.split("::", 1)[0] if "::" in extra else ""
        self._store.session_touch(
            session_id=extra,
            user_id=user_id,
            model_hash=self._model_hash,
            lmdb_prefix="",
            block_size=0,
            num_blocks=0,
            n_tokens=start_pos + len(seg),
            engine_type=QML_ENGINE_SGLANG,
        )
        if self._record_token_log:
            self._store.tokens_put(extra, seg, seq_id=0, start_pos=start_pos)


    def batch_exists(self, keys: List[str], extra_info: Optional[Any] = None) -> int:
        """Count how many keys exist contiguously from index 0 (one LMDB txn).

        in: keys — page hashes in radix order
        in: extra_info — unused (HiCacheStorage ABC signature compatibility)
        out: int N such that keys[0..N-1] all exist and keys[N] (if any) does not
        """
        # contract per hicache_storage.py: return how many keys exist
        # CONTIGUOUSLY from index 0 -- exactly qml_lmdb_exists_run, one txn
        return self._store.blocks_exist_run(keys)
