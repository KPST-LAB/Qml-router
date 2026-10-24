# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
"""hooks/vllm/qml_vllm_connector.py

KVConnectorBase_V1 implementation backed by qml-router's shared SQLite+LMDB
store (qml_store_bindings.py). Mirrors the layered design already built and
verified for llama.cpp, adapted to how vLLM's connector API actually works:

  Layer 1: qml_session_store (C, shared across engines) -- storage/sqlite,
           storage/lmdb, core/qml_session.c. Unmodified by this file.
  Layer 2: THIS FILE -- vLLM-aware adapter, all block-hashing/store calls
           live here.
  (no Layer 3 here -- unlike llama.cpp, vLLM's public KVConnectorBase_V1
   plugin mechanism IS the call site. No vLLM source is patched.)

Key design facts, confirmed by reading third_party/vllm/vllm's actual
source (not assumed):
  - vLLM's scheduler asks the connector "how many MORE tokens beyond what
    you've already computed (num_computed_tokens) can you supply" via
    get_num_new_matched_tokens() -- a block-level prefix question, not a
    "restore this session" question. There is no llama.cpp-style whole-
    session-blob restore concept in this API at all.
  - request.cache_salt (vllm/v1/request.py) is the closest native
    equivalent of qml's session_id: an opaque, client-settable string
    (exposed in the OpenAI-compatible chat completion request schema)
    used to scope/namespace cache entries across a multi-turn conversation.
    It is what this file uses as session_id.
  - vLLM's internal Request object has NO user_id/user passthrough -- the
    OpenAI protocol's `user` field does not reach this layer (confirmed by
    grep: vllm/v1/request.py's Request.__init__ has no such parameter).
    user_id is therefore left empty here; wiring a real value through
    would need a vLLM-side change, out of scope for this connector alone.
  - KV load/save happens per LAYER (save_kv_layer/start_load_kv), not per
    whole session -- this maps naturally onto qml's content-addressable
    block storage (key = hash(model, layer, token block)), unlike
    llama.cpp's one opaque per-sequence blob.

Registration (no vLLM source patch needed):
    --kv-transfer-config '{
        "kv_connector": "QmlVllmConnector",
        "kv_connector_module_path": "hooks.vllm.qml_vllm_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "qml_sqlite_path": "/path/to/qml.db",
            "qml_lmdb_path": "/path/to/qml_lmdb"
        }
    }'

VERIFICATION STATUS (be precise, do not overclaim):
  - Fully verified live against a real vLLM 0.26.0 install + real GPU
    (google/gemma-3-4b-it, HF safetensors -- see docs/4-engines/engine-vllm.md).
    Both the store path (save_kv_layer) and the load/reuse path
    (start_load_kv) confirmed via vLLM's own num_cached_tokens field and
    bit-identical regenerated output across a process restart. Five real
    bugs were found and fixed this way (bytearray range, NumPy bfloat16,
    cache_salt capture, tensor reshape, CPU/GPU device mismatch) -- see
    docs/4-engines/engine-vllm.md for the exact fixes. The standard test suite (short/long/multi-turn/
    needle) and all three cross-engine scenarios (a->b, b->a, c=a+b with
    llama.cpp) also passed live.
"""

import hashlib
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qml_store_bindings import QmlStore  # noqa: E402

logger = logging.getLogger(__name__)

# Upper bound on how long save_kv_layer may wait for room in the save
# buffer before giving up on persisting THIS request. Without a bound a
# saturated writer (slow storage, large sessions) blocks inside the
# forward pass for as long as the drain takes, which was measured as
# tens of seconds per request. Giving up costs a later recompute, never
# correctness: the token log, not the KV blob, is the canonical record.
_SAVE_WAIT_BUDGET_S = 0.2

# Restore-versus-recompute decision (the cost model of the paper's Section V-D,
# applied per request). Both rates are measured by this connector while it runs:
# the store read rate comes from completed loads, the engine's token rate from
# the scheduler steps. Restoring a prefix is only worth it when reading its
# bytes back is faster than letting the GPU recompute the same tokens, which on
# low-bandwidth storage it is not. Until both rates have a sample the connector
# restores, which is the behaviour it had before this gate existed.
_cost_lock = threading.Lock()
_restore_bytes_per_s = 0.0
_engine_tokens_per_s = 0.0
_COST_EMA = 0.3          # weight of the newest sample
_COST_MARGIN = 1.0       # restore must be at least this much faster to be used


def _cost_update(restore_bps: float = 0.0, engine_tps: float = 0.0) -> None:
    """Folds one measured rate into the shared cost-model estimate.

    in: restore_bps -- bytes per second observed for a completed store read
    in: engine_tps -- tokens per second observed for one scheduler step
    out: None; updates the module-level exponential moving averages
    """
    global _restore_bytes_per_s, _engine_tokens_per_s
    with _cost_lock:
        if restore_bps > 0:
            _restore_bytes_per_s = (restore_bps if _restore_bytes_per_s <= 0 else
                                    (1 - _COST_EMA) * _restore_bytes_per_s + _COST_EMA * restore_bps)
        if engine_tps > 0:
            _engine_tokens_per_s = (engine_tps if _engine_tokens_per_s <= 0 else
                                    (1 - _COST_EMA) * _engine_tokens_per_s + _COST_EMA * engine_tps)


def _restore_is_worth_it(n_tokens: int, bytes_per_token: float) -> bool:
    """Answers whether reading n_tokens back beats recomputing them.

    in: n_tokens -- tokens the store could supply beyond what the engine has
    in: bytes_per_token -- stored KV bytes per token for this model and layout
    out: True to restore, False to let the engine recompute (the blocks stay in
        the store, so a faster device or a warmer cache can still use them)
    """
    with _cost_lock:
        bps, tps = _restore_bytes_per_s, _engine_tokens_per_s
    if bps <= 0 or tps <= 0 or bytes_per_token <= 0:
        return True
    return (n_tokens * bytes_per_token) / bps <= (n_tokens / tps) * _COST_MARGIN

# Same interval as hooks/llama.cpp/qml_llama_hook.cpp's
# kControlCmdPollInterval -- not latency-critical, see
# docs/udf-control-plane-proposal.md's blocker #1 resolution.
_CONTROL_CMD_POLL_INTERVAL_SEC = 0.2


def _prime(self_url: str, model_name: str, token_ids: list, session_id: str) -> None:
    """RESTORE/PREWARM's actual mechanism for vLLM.

    Verified against vLLM's real, installed source (0.26.0) before writing
    this, not assumed: vLLM's KV blocks are managed entirely by
    v1/core/kv_cache_manager.py's KVCacheManager, whose only public entry
    points (get_computed_blocks/allocate_slots/free/evict_blocks) all key
    off a live `Request` object -- there is no "just load these blocks"
    API reachable without one. KVConnectorBase_V1's own __init__ (see
    distributed/kv_transfer/kv_connector/v1/base.py) only exposes
    vllm_config/kv_cache_config/role -- no handle to the scheduler or
    KVCacheManager at all. So the only real way to make our connector's
    already-proven start_load_kv/save_kv_layer path run for a token
    sequence ahead of any real client request is to submit an actual
    (self-issued) request for it -- there is no lower-level shortcut.

    A `max_tokens=1` completion over the known token_ids, with
    cache_salt=session_id (vLLM's own cache-isolation field, confirmed on
    CompletionRequest in entrypoints/openai/completion/protocol.py, which
    also confirms `prompt` accepts a raw `list[int]`), does exactly this:
    it drives get_num_new_matched_tokens/build_connector_meta/
    start_load_kv or save_kv_layer through the normal, already-verified
    request path -- just triggered by us instead of an external client.

    in: self_url -- base URL of THIS vLLM server (e.g. "http://127.0.0.1:8000")
    in: model_name -- served model name to put in the completion request body
    in: token_ids -- full token sequence to prime, sent as a raw list[int] prompt
    in: session_id -- qml session id, sent as cache_salt for cache isolation
    out: None; raises on a non-2xx HTTP response or timeout
    side effects: drives the connector's normal save/load path via a
        self-issued max_tokens=1 /v1/completions request, which populates
        GPU KV blocks (and, via save, the SQLite+LMDB store) for this session
    """
    import urllib.request as _urlreq

    body = json.dumps({
        "model": model_name,
        "prompt": token_ids,
        "max_tokens": 1,
        "temperature": 0,
        "cache_salt": session_id,
    }).encode("utf-8")
    req = _urlreq.Request(
        f"{self_url}/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with _urlreq.urlopen(req, timeout=30) as resp:
        resp.read()  # raises on non-2xx; body itself unused


def _dispatch_control_cmd(store: "QmlStore", cmd: dict, self_url, model_name: str) -> None:
    """Mirrors hooks/llama.cpp/qml_llama_hook.cpp's dispatch_control_cmd
    in spirit (same cmd_types get a real DONE/FAILED verdict from an
    external controller's point of view), but NOT in mechanism -- vLLM's
    block-pool architecture genuinely has no per-conversation "slot" to
    inject into like llama.cpp does, so each cmd_type had to be
    individually re-derived against vLLM's real source rather than ported
    1:1. See _prime's docstring for RESTORE/PREWARM;
    BACKUP_NOW and MIGRATE below have their own real architectural
    reasoning, not workarounds.

    in: store -- open QmlStore, used for status updates and token-log reads
    in: cmd -- claimed command dict (cmd_id/cmd_type/session_id/...)
    in: self_url -- this server's base URL or None (required for RESTORE/PREWARM)
    in: model_name -- served model name for the self-issued completion
    out: None
    side effects: writes the command's final status (DONE/FAILED) to the
        SQLite control queue; COMPACT_NOW compacts the SQLite db;
        RESTORE/PREWARM trigger a self-completion that loads KV into the GPU
    """
    cmd_id = cmd["cmd_id"]
    cmd_type = cmd["cmd_type"]
    session_id = cmd["session_id"]

    if cmd_type == "MONITOR_PING":
        logger.info(
            "[qml][control] MONITOR_PING cmd_id=%s session_id=%s handled by pid=%d",
            cmd_id, session_id, os.getpid(),
        )
        store.control_set_status(cmd_id, "DONE")
        return

    if cmd_type == "COMPACT_NOW":
        ok = store.compact()
        logger.info(
            "[qml][control] COMPACT_NOW cmd_id=%s handled by pid=%d ok=%s",
            cmd_id, os.getpid(), ok,
        )
        store.control_set_status(cmd_id, "DONE" if ok else "FAILED")
        return

    if cmd_type == "BACKUP_NOW":
        # Real architectural fact, not a shortcut: unlike llama.cpp (which
        # only persists a session's KV at end-of-turn, in release()),
        # this connector's save_kv_layer already writes every newly
        # computed block to LMDB as it's produced, every decode step (see
        # wait_for_save()/save_kv_layer() above). There is no "not yet
        # saved" state for BACKUP_NOW to force past -- whatever's in GPU
        # memory for this session right now is already durably persisted.
        logger.info(
            "[qml][control] BACKUP_NOW cmd_id=%s session_id=%s: no-op -- vLLM connector "
            "already persists every block as it's computed (save_kv_layer), nothing pending",
            cmd_id, session_id,
        )
        store.control_set_status(cmd_id, "DONE")
        return

    if cmd_type in ("RESTORE", "PREWARM"):
        if not self_url:
            logger.warning(
                "[qml][control] cmd_id=%s (%s) FAILED: qml_self_url not configured "
                "(required for vLLM RESTORE/PREWARM -- see docstring)",
                cmd_id, cmd_type,
            )
            store.control_set_status(cmd_id, "FAILED")
            return

        token_ids = store.tokens_get(session_id)
        if not token_ids:
            logger.warning(
                "[qml][control] cmd_id=%s (%s) FAILED: no qml_token_log rows for session_id=%s",
                cmd_id, cmd_type, session_id,
            )
            store.control_set_status(cmd_id, "FAILED")
            return

        try:
            _prime(self_url, model_name, token_ids, session_id)
            logger.info(
                "[qml][control] cmd_id=%s (%s) primed session_id=%s via self-completion: %d tokens",
                cmd_id, cmd_type, session_id, len(token_ids),
            )
            store.control_set_status(cmd_id, "DONE")
        except Exception:
            logger.exception("[qml][control] cmd_id=%s (%s) self-completion request failed", cmd_id, cmd_type)
            store.control_set_status(cmd_id, "FAILED")
        return

    if cmd_type == "MIGRATE":
        # Genuinely not reachable, not merely unimplemented: verified
        # against vLLM's real source that KVConnectorBase_V1.__init__ (see
        # distributed/kv_transfer/kv_connector/v1/base.py) exposes only
        # vllm_config/kv_cache_config/role -- no handle to the scheduler
        # or KVCacheManager, which is the only thing with evict_blocks()/
        # free(). Forcing eviction of a specific session's blocks isn't
        # possible from inside a connector as vLLM's API is designed.
        # vLLM's own KVCacheManager already evicts LRU blocks automatically
        # under memory pressure -- there's no external "free this session
        # now" lever to pull, unlike llama.cpp's explicit per-slot model.
        logger.warning(
            "[qml][control] cmd_id=%s (MIGRATE) FAILED: not reachable from a KVConnectorBase_V1 "
            "connector -- vLLM exposes no block-eviction handle to it (see docstring); vLLM's own "
            "KVCacheManager evicts LRU blocks automatically instead",
            cmd_id,
        )
        store.control_set_status(cmd_id, "FAILED")
        return

    logger.warning("[qml][control] cmd_id=%s cmd_type=%s UNRECOGNIZED -- marking FAILED", cmd_id, cmd_type)
    store.control_set_status(cmd_id, "FAILED")


# run an eviction pass at most every N poll iterations -- the budget check is a single
# SUM() over qml_kv_block, but there is no reason to run it 5x/second
_BLOCK_EVICT_EVERY_N_POLLS = 30


def _control_loop(store: "QmlStore", stop_event: threading.Event, self_url,
                           model_name: str, block_budget_bytes: int = 0) -> None:
    """Background thread body: polls the SQLite control-command queue and runs LRU eviction.

    in: store -- open QmlStore to claim commands from and evict blocks in
    in: stop_event -- set by shutdown() to end the loop
    in: self_url -- this server's base URL or None, passed through to dispatch
    in: model_name -- served model name, passed through to dispatch
    in: block_budget_bytes -- LMDB block byte budget for LRU eviction; 0 disables it
    out: None (returns only once stop_event is set)
    side effects: dispatches claimed control commands (may write store/GPU
        state -- see _dispatch_control_cmd) and deletes cold KV blocks from
        SQLite+LMDB once every _BLOCK_EVICT_EVERY_N_POLLS iterations
    """
    target_pid = os.getpid()
    evict_countdown = 0
    while not stop_event.is_set():
        try:
            cmd = store.control_claim(target_pid)
            while cmd is not None:
                _dispatch_control_cmd(store, cmd, self_url, model_name)
                cmd = store.control_claim(target_pid)
        except Exception:
            logger.exception("[qml][control] poll loop iteration failed")

        # task #56: LRU eviction runs on THIS thread, never on the request path --
        # save_kv_layer/start_load_kv are inside the forward pass and must not
        # take a SQLite write lock or delete LMDB pages.
        if block_budget_bytes > 0:
            evict_countdown -= 1
            if evict_countdown <= 0:
                evict_countdown = _BLOCK_EVICT_EVERY_N_POLLS
                try:
                    dropped = store.block_evict_lru(block_budget_bytes)
                    if dropped:
                        logger.info("[qml][lru] evicted %d cold KV block(s); tracked total now %d bytes "
                                    "(budget %d)", dropped, store.block_total_bytes(), block_budget_bytes)
                except Exception:
                    logger.exception("[qml][lru] block eviction failed")

        stop_event.wait(_CONTROL_CMD_POLL_INTERVAL_SEC)


def _align_to_block_size(num_tokens: int, block_size: int) -> int:
    """Same alignment rule as vLLM's own example_connector.py -- only whole
    blocks are ever considered "cacheable", the trailing partial block is
    always recomputed locally.

    in: num_tokens -- token count to align down
    in: block_size -- KV block size in tokens
    out: largest whole-block-aligned token count strictly below num_tokens
        ((num_tokens - 1) // block_size * block_size)
    """
    return (num_tokens - 1) // block_size * block_size


def _block_token_bytes(token_ids) -> bytes:
    """Serializes one block's token ids into canonical bytes for hashing.

    in: token_ids -- iterable of token ids for one block
    out: bytes -- each token id packed as 4 little-endian bytes, concatenated
    """
    # NOTE: bytearray(int_generator) requires each element to be a single
    # byte (0-255) -- a real, previously-hit bug here masked `int(t) &
    # 0xFFFFFFFF` (a 32-bit value) straight into bytearray() and only broke
    # on real vocab token ids (>255), never on the small [1,2,3,4]-style
    # ids used in earlier standalone unit tests. Pack each token id as its
    # own 4 bytes instead.
    return b"".join((int(t) & 0xFFFFFFFF).to_bytes(4, "little") for t in token_ids)


def _file_fingerprint(path: str) -> str:
    """size + mtime + first/last 4MiB of the largest weight files under path.
    Mirrors core/qml_model_hash.c's sampling strategy (a full hash of a
    multi-GB checkpoint measured 13.6s, too slow for every startup). Only
    usable when the model is a LOCAL directory; HF-cache ids resolve
    elsewhere and are covered by the id+revision part of the descriptor
    instead -- documented as a known narrower guarantee, same scope decision
    as qml_model_file_hash().

    in: path -- local model checkpoint directory to fingerprint
    out: 32-hex-char sampled SHA256 fingerprint string, or "nofp" when
        sampling failed (the cache key then relies on the config descriptor only)
    """
    import os
    CHUNK = 4 * 1024 * 1024
    h = hashlib.sha256()
    try:
        weights = []
        for root, _dirs, files in os.walk(path):
            for f in files:
                if f.endswith((".safetensors", ".bin", ".gguf", ".pt")):
                    fp = os.path.join(root, f)
                    weights.append((os.path.getsize(fp), fp))
        weights.sort(reverse=True)
        for size, fp in weights[:4]:          # largest shards are enough
            st = os.stat(fp)
            h.update(f"{os.path.basename(fp)}|{size}|{int(st.st_mtime)}|".encode())
            with open(fp, "rb") as fh:
                h.update(fh.read(CHUNK))
                if size > CHUNK:
                    fh.seek(max(0, size - CHUNK))
                    h.update(fh.read(CHUNK))
    except OSError as e:
        logger.warning("qml: weight fingerprint sampling failed for %s (%s) -- "
                       "cache key falls back to the config descriptor only", path, e)
        return "nofp"
    return h.hexdigest()[:32]


def _compute_cache_seed(vllm_config, kv_cache_config, block_size: int) -> str:
    """Canonical descriptor of everything the stored KV bytes depend on.

    The old seed was just vllm_config.model_config.model -- a path/HF-id
    STRING. Persistent LMDB blocks outlive the process, so that is unsafe:
    the same model served with a different kv_cache_dtype produces IDENTICAL
    keys, and the load path reinterprets the bytes with the CURRENT layer
    dtype (torch.frombuffer(raw, dtype=kv_cache_layer.dtype)) -- wrong
    precision, silently, no error. stock llama.cpp does not have this hole:
    llama_kv_cache::state_read_data() reads the blob header and rejects a
    mismatched layer count / V transposition / per-layer K/V type / row size.
    The connector path has no such header, so the guarantee has to live in
    the key instead (plus the byte-length check on load).

    in: vllm_config -- vLLM engine config (model/parallel/cache sub-configs read)
    in: kv_cache_config -- vLLM KV cache layout config (used for the layer count)
    in: block_size -- KV block size in tokens
    out: 64-hex-char SHA256 seed string identifying the stored-KV byte format
    """
    mc = vllm_config.model_config
    pc = vllm_config.parallel_config
    cc = vllm_config.cache_config

    def _attr(obj, name, default="?"):
        """Reads obj.<name> as a string, tolerating missing attributes.

        in: obj -- object to read from
        in: name -- attribute name to read
        in: default -- value used when the attribute is missing
        out: str() of the attribute value (or of the default)
        """
        v = getattr(obj, name, default)
        return str(v)

    n_layers = 0
    try:
        n_layers = len(kv_cache_config.kv_cache_tensors)
    except Exception:
        try:
            n_layers = len(getattr(kv_cache_config, "kv_cache_groups", []) or [])
        except Exception:
            n_layers = 0

    parts = [
        "qmlseed1",
        _attr(mc, "model"),
        _attr(mc, "revision"),
        _attr(mc, "dtype"),
        _attr(cc, "cache_dtype"),
        _attr(pc, "tensor_parallel_size"),
        _attr(pc, "pipeline_parallel_size"),
        str(block_size),
        str(n_layers),
    ]
    try:
        parts.append(str(mc.get_num_kv_heads(pc)))
    except Exception:
        parts.append("?")
    try:
        parts.append(str(mc.get_head_size()))
    except Exception:
        parts.append("?")

    import os
    model_path = _attr(mc, "model")
    if os.path.isdir(model_path):
        parts.append(_file_fingerprint(model_path))
    else:
        # not a local checkpoint dir (HF id / remote) -- the id+revision above
        # is the only weight identity available without network access
        parts.append("nolocalfp")

    descriptor = "|".join(parts)
    seed = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()
    logger.info("qml: cache seed=%s from descriptor=%s", seed[:16], descriptor)
    return seed


def _scoped_seed(cache_seed: str, share_scope: str, session_id, tenant_id) -> str:
    """Applies the qml_share_scope policy to the chain seed.

    global : seed unchanged -> blocks are shared across ALL sessions/users.
             This is what the code did unconditionally before, and it
             reintroduces at the storage layer exactly the cross-request
             prefix-cache timing side channel that vLLM's own cache_salt
             exists to prevent -- so it is now an explicit opt-in.
    tenant : blocks shared only within one tenant.
    none   : (default) seed includes the session id, so nothing is shared
             between sessions. Matches the isolation the caller already
             asked for by sending cache_salt=session_id.

    in: cache_seed -- model/config-level seed from _compute_cache_seed
    in: share_scope -- sharing policy: "global", "tenant" or "none"
    in: session_id -- request's cache_salt (mixed in when scope is "none")
    in: tenant_id -- tenant name (mixed in when scope is "tenant")
    out: seed string actually used to anchor the block hash chain
    """
    if share_scope == "global":
        return cache_seed
    if share_scope == "tenant":
        return cache_seed + "|tenant=" + str(tenant_id or "")
    return cache_seed + "|session=" + str(session_id or "")


def _compute_chain(model_hash: str, token_ids, block_size: int) -> list:
    """Per-block hash CHAIN over aligned blocks of token_ids: chain[i] =
    sha256(chain[i-1] | block_i's tokens), seeded with model_hash for
    chain[0]. This is what makes genuine PARTIAL-prefix reuse detectable
    -- two different requests (any session/user) that share the same
    leading N blocks compute identical chain[0..N-1] regardless of what
    follows, so get_num_new_matched_tokens can walk the chain and count
    how many leading blocks are actually cached instead of an all-or-
    nothing check on the whole prefix (which is what vLLM's own
    example_connector.py does, and what this file's first cut also did --
    see qml-router task #28).

    in: model_hash -- scoped cache seed string anchoring chain[0]
    in: token_ids -- prompt token ids (only whole blocks are hashed)
    in: block_size -- KV block size in tokens
    out: list of len(token_ids)//block_size 32-byte sha256 digests,
        chain[i] covering the token content of blocks 0..i
    """
    n_blocks = len(token_ids) // block_size
    chain = []
    prev = model_hash.encode("utf-8")
    for i in range(n_blocks):
        block = token_ids[i * block_size:(i + 1) * block_size]
        prev = hashlib.sha256(prev + b"|" + _block_token_bytes(block)).digest()
        chain.append(prev)
    return chain


def _layer_block_key(layer_name: str, chain_hash: bytes) -> str:
    """Per-(layer, exact-prefix-through-this-block) content key for the
    actual KV tensor bytes. Keyed on the chain hash (not just this block's
    own tokens) because KV values causally depend on everything before
    them -- two blocks with identical tokens but different preceding
    context are NOT numerically identical, so they must not share a key.

    in: layer_name -- vLLM layer name the KV bytes belong to
    in: chain_hash -- 32-byte chain digest for this block's exact prefix
    out: hex SHA256 string used as the LMDB key for this (layer, block)
    """
    return hashlib.sha256(layer_name.encode("utf-8") + b"|" + chain_hash).hexdigest()


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Kept for reference and for any non-batched caller: the save path now
    writes straight from a contiguous host tensor's data_ptr (see
    blocks_put_from), which skips numpy entirely.

    NumPy has no bfloat16 dtype -- tensor.numpy() raises TypeError directly on
    bf16 tensors (a real bug hit live: vLLM's default KV cache dtype is
    bfloat16), hence the uint16 reinterpret below. The raw data_ptr write used
    now produces the SAME bytes, and the load side reads them back with
    torch.frombuffer(..., dtype=<layer dtype>), which accepts bfloat16
    directly (verified bit-exact).

    in: tensor -- KV tensor (any device; bf16 handled via uint16 reinterpret)
    out: raw bytes of the tensor's elements in row-major element order
    """
    t = tensor.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.uint16)
    return t.numpy().tobytes()


def _manifest_block_key(chain_hash: bytes) -> str:
    """Presence sentinel for one block (not the whole prefix): written
    once this block's real per-layer data has all been saved (see
    wait_for_save), so get_num_new_matched_tokens can cheaply ask "is
    block i cached" without needing to know how many layers exist.

    in: chain_hash -- 32-byte chain digest for the block
    out: hex SHA256 string keying the block's presence marker in LMDB
    """
    return hashlib.sha256(b"__qml_manifest__|" + chain_hash).hexdigest()


def _find_matched_blocks(store: "QmlStore", model_hash: str, token_ids, block_size: int):
    """Walks the chain from the start; returns (full_chain, matched_blocks)
    where matched_blocks is the count of CONSECUTIVE leading blocks whose
    manifest marker is already present. This is the actual block-level
    slicing -- a request can reuse the first K blocks of a previously
    cached prefix even if what follows differs or belongs to a different
    session/user.

    in: store -- open QmlStore to query manifest markers in
    in: model_hash -- scoped cache seed string for the chain
    in: token_ids -- whole-block-aligned token prefix to match
    in: block_size -- KV block size in tokens
    out: (full_chain, matched_blocks) -- the per-block hash chain over
        token_ids and the count of consecutive leading blocks already cached
    """
    chain = _compute_chain(model_hash, token_ids, block_size)
    # one transaction for the whole run, not one per block: this is asked on
    # every request before anything is transferred, so its cost is pure added
    # TTFT (measured at 101 ms for 413 blocks when done key by key)
    matched = store.blocks_exist_run([_manifest_block_key(h) for h in chain])
    return chain, matched


@dataclass
class ReqMeta:
    request_id: str
    token_ids: list
    block_ids: list
    chain: list  # per-block chain hashes (see _compute_chain), aligned with block_ids
    matched_blocks: int  # only meaningful when is_store=False -- see _find_matched_blocks
    is_store: bool
    session_id: str | None
    # blocks in the WHOLE prompt, not just this chunk's slice of it. Under
    # chunked prefill one request produces several store ReqMetas, so
    # len(chain) is a per-chunk count and cannot be used as the session's
    # block count. session_id is set on the last chunk only, and that chunk
    # carries the total here.
    total_blocks: int = 0


@dataclass
class QmlConnectorMetadata(KVConnectorMetadata):
    requests: list = field(default_factory=list)


class QmlVllmConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config):
        """Opens the shared SQLite+LMDB store and starts the connector's worker threads.

        in: vllm_config -- vLLM engine config; cache/model/parallel settings and
            kv_transfer_config extra config (qml_sqlite_path, qml_lmdb_path,
            qml_read_threads, qml_kv_compress, qml_self_url, qml_share_scope,
            qml_tenant_id, qml_block_budget_bytes, qml_token_log) are read here
        in: role -- KVConnectorRole (scheduler or worker side)
        in: kv_cache_config -- vLLM KV cache layout config (feeds the cache seed)
        out: None; raises ValueError on missing or invalid extra config
        side effects: opens the process-global SQLite+LMDB store and spawns the
            qml-save-writer, qml-load-prefetch and qml-control-cmd-poll threads
        """
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)

        self._block_size = vllm_config.cache_config.block_size
        # stored KV bytes per token, used by the restore-versus-recompute gate
        self._bytes_per_token = 0.0
        try:
            mc, pc = vllm_config.model_config, vllm_config.parallel_config
            n_layer = mc.get_num_layers(pc)
            n_kv = mc.get_num_kv_heads(pc)
            head = mc.get_head_size()
            item = torch.tensor([], dtype=mc.dtype).element_size()
            self._bytes_per_token = float(2 * n_kv * head * item * n_layer)
        except Exception:
            logger.warning("qml: KV layout unavailable; restore gate stays open",
                           exc_info=True)
        # NOTE: two DIFFERENT things that used to be one field.
        #  _model_name is the served model name -- it goes into the
        #    self-issued /v1/completions body for RESTORE/PREWARM, so it must
        #    stay the plain name vLLM was started with.
        #  _cache_seed is the storage identity of the KV bytes (task #53):
        #    model id+revision, model dtype, kv cache dtype, TP/PP size,
        #    block_size, layer count, num_kv_heads, head_size, plus a sampled
        #    weight fingerprint for local checkpoints. Using the model name
        #    for this (as this file did before) let incompatible configs share
        #    keys -- see _compute_cache_seed()'s docstring.
        self._model_name = vllm_config.model_config.model
        self._cache_seed = _compute_cache_seed(vllm_config, kv_cache_config, self._block_size)
        self._requests_need_load: dict[str, Any] = {}
        # requests whose KV save was abandoned because the save buffer
        # stayed full (see _SAVE_WAIT_BUDGET_S). No manifest marker is
        # written for them, so nothing later matches half-written data.
        self._save_dropped: set[str] = set()
        # one wait deadline per request per step: the budget below is what the
        # WHOLE request may spend waiting for save-buffer room, not what each
        # of its layers may spend (28 layers x the budget was measured as
        # seconds of forward-pass stall, TEST-LEDGER row 89-1)
        self._save_deadline: dict[str, float] = {}
        self._save_dropped_total = 0
        # per-request store progress across chunked-prefill steps:
        #   req_id -> {chain, blocks, token_ids, session_id, saved_blocks}
        # A prompt longer than max_num_batched_tokens is prefilled over
        # several steps; only the blocks whose tokens were actually computed
        # in a step may be saved, and the blocks for the later chunks are
        # allocated later. Saving the whole chain on the first step both
        # indexes past the end of block_ids and would store uncomputed KV.
        self._store_progress: dict[str, dict] = {}
        # req_id -> (chain, matched_blocks) handed from
        # get_num_new_matched_tokens to build_connector_meta (task #65)
        self._match_cache: dict[str, tuple] = {}
        # accumulated time inside save_kv_layer for the current request,
        # reported by wait_for_save under QML_LOAD_TIMING=1
        self._save_ms = 0.0
        self._num_layers_seen = 0
        # workers for the batched store read. Most of that read is faulting
        # LMDB's mmap pages in, which parallelises; the ctypes call releases
        # the GIL so these are real parallel readers. qml_read_threads=1
        # restores the single-threaded path.
        self._read_threads = int(
            self._kv_transfer_config.get_from_extra_config("qml_read_threads", 4) or 4
        )
        # NewRequestData (scheduler_output.scheduled_new_reqs' element type)
        # has NO cache_salt field -- confirmed via dataclasses.fields(), a
        # real bug hit live where getattr(new_req, "cache_salt", None)
        # silently always returned None. cache_salt only lives on the full
        # Request object, which on_new_request() below receives.
        self._request_cache_salt: dict[str, str] = {}

        sqlite_path = self._kv_transfer_config.get_from_extra_config("qml_sqlite_path", None)
        lmdb_path = self._kv_transfer_config.get_from_extra_config("qml_lmdb_path", None)
        if not sqlite_path or not lmdb_path:
            raise ValueError(
                "QmlVllmConnector requires kv_connector_extra_config = "
                "{'qml_sqlite_path': ..., 'qml_lmdb_path': ...}"
            )

        self._store = QmlStore()
        self._store.open(sqlite_path, lmdb_path)
        logger.info("QmlVllmConnector: opened store sqlite=%s lmdb=%s", sqlite_path, lmdb_path)

        # async save (2026-08-02). The durable LMDB write used to run inside
        # the forward pass; at 32K tokens / FP16 KV that is ~4 GB per session
        # pushed through a ~0.5 GB/s disk, and it collapsed the whole server
        # under mixed load (15 short / 4 long completions vs stock vLLM's
        # 99/32 -- docs/4-engines/engine-comparison.md). Split it
        # like the llama.cpp persister: the device->host copy stays
        # synchronous (the engine reuses that GPU memory next step); only the
        # durable write moves to this writer thread.
        #
        # Correctness needs NO read-your-writes here, unlike llama.cpp's
        # restore: a block only becomes matchable once its manifest marker is
        # durable, and wait_for_save() enqueues the markers AFTER the layer
        # data on this same FIFO single-consumer queue -- so a scheduler-side
        # match can never see a block whose bytes are not fully written. A
        # crash with jobs still queued just leaves blocks unmatchable
        # (recomputed later, correct if slower).
        #
        # QML_SAVE_RAM_MB bounds the buffered bytes (default 6144); at the
        # limit save_kv_layer blocks, degrading to the old synchronous
        # behavior instead of exhausting host RAM.
        self._save_ram_budget = int(os.environ.get("QML_SAVE_RAM_MB", "6144")) * 1024 * 1024
        self._save_q: "queue.Queue[tuple]" = queue.Queue()
        self._save_buffered = 0
        self._save_cv = threading.Condition()
        self._save_thread = threading.Thread(
            target=self._save_writer, daemon=True, name="qml-save-writer"
        )
        self._save_thread.start()

        # restore-side pipeline (2026-08-02): overlap the NEXT layer's store
        # read with the CURRENT layer's host->device copy (double-buffered;
        # one prefetch worker + the already-parallel qml_read_threads inside
        # each read). Serial read->copy->read->copy leaves the disk idle
        # during every DMA and the DMA idle during every read.
        import concurrent.futures as _cf
        self._load_pool = _cf.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="qml-load-prefetch"
        )

        # at-rest KV compression (2026-08-02): the cold bottleneck is BYTES
        # (32K session = 4 GB FP16 KV through a ~0.5 GB/s disk; measured
        # 15/4 -> 35/11 after async save, still below stock's 99/32).
        # int8 halves what crosses the disk AND the PCIe bus in both
        # directions. Quantization is symmetric absmax per head_dim group
        # (finer than a per-block scale; same family as llama.cpp's q8_0,
        # which has produced zero wrong answers across this project), and
        # COMPUTE STAYS FP16 -- bytes are dequantized on the GPU before
        # injection, unlike vLLM's fp8 KV cache mode whose fp8 COMPUTE
        # produced 10.4% wrong answers on the 24K needle test.
        # Stored layout per block: [int8 data | fp16 scales].
        # Format detection is BY SIZE at read time (user direction,
        # 2026-08-02): for a given layer the fp16 block size and the int8
        # block size are both known constants and differ, and the store's
        # existing size check already refuses a mismatched value -- so the
        # loader tries one stride and falls back to the other per block run.
        # No key suffix, no separate namespace: an old fp16 store, a new
        # int8 store, and a MIXED store (fp16 prefix saved before the
        # upgrade + int8 tail after) all load correctly. QML_KV_COMPRESS=off
        # or extra config qml_kv_compress="off" restores raw fp16 SAVES;
        # reads always accept both formats.
        self._kv_compress = str(
            self._kv_transfer_config.get_from_extra_config("qml_kv_compress", "int8")
        ).lower()
        if os.environ.get("QML_KV_COMPRESS"):
            self._kv_compress = os.environ["QML_KV_COMPRESS"].lower()
        if os.environ.get("QML_SAVE_BATCH") == "0" and self._kv_compress != "off":
            logger.warning("QML_SAVE_BATCH=0 (legacy A/B path) forces qml_kv_compress=off")
            self._kv_compress = "off"
        if self._kv_compress not in ("int8", "off"):
            raise ValueError(f"qml_kv_compress must be int8/off, got {self._kv_compress!r}")
        logger.info("QmlVllmConnector: kv_compress=%s (reads auto-detect both formats)",
                    self._kv_compress)

        # Optional: only required for RESTORE/PREWARM control commands (see
        # _prime) -- e.g. "http://127.0.0.1:8000". Not
        # needed for MONITOR_PING/COMPACT_NOW, so not a hard requirement
        # like sqlite_path/lmdb_path above.
        self._qml_self_url = self._kv_transfer_config.get_from_extra_config("qml_self_url", None)

        # qml_token_log: "on" (default) records each request's token sequence in
        # qml_token_log. "off" leaves the token log to a session layer in front of
        # the engine (router/qml_openai_gateway.py): its prompts also carry
        # recalled archive turns, which are not part of the conversation record.
        # KV save/load and the session row are the same either way.
        self._record_token_log = str(
            self._kv_transfer_config.get_from_extra_config("qml_token_log", "on")
        ).lower() != "off"
        # GPU block ids whose promised KV could not be read back; reported to
        # vLLM through get_block_ids_with_load_errors() so it recomputes them
        # (kv_load_failure_policy="recompute") or fails the request ("fail")
        self._load_error_blocks: set = set()

        # task #54: cross-session block sharing is now opt-in. Default "none"
        # because the caller passing cache_salt=session_id has already asked
        # for isolation, and sharing at the storage layer would undo it.
        self._share_scope = str(
            self._kv_transfer_config.get_from_extra_config("qml_share_scope", "none")
        ).lower()
        if self._share_scope not in ("none", "tenant", "global"):
            raise ValueError(
                f"qml_share_scope must be one of none/tenant/global, got {self._share_scope!r}"
            )
        self._tenant_id = self._kv_transfer_config.get_from_extra_config("qml_tenant_id", None)
        if self._share_scope == "tenant" and not self._tenant_id:
            raise ValueError("qml_share_scope='tenant' requires qml_tenant_id")
        logger.info(
            "QmlVllmConnector: share_scope=%s tenant=%s (global means stored KV blocks "
            "are reused across all sessions -- see task #54)",
            self._share_scope, self._tenant_id,
        )

        # task #56: LMDB block budget for LRU eviction; 0 disables eviction
        try:
            self._block_budget_bytes = int(
                self._kv_transfer_config.get_from_extra_config("qml_block_budget_bytes", 0) or 0
            )
        except (TypeError, ValueError):
            self._block_budget_bytes = 0

        self._control_stop = threading.Event()
        self._control_thread = threading.Thread(
            target=_control_loop,
            args=(self._store, self._control_stop, self._qml_self_url, self._model_name,
                  self._block_budget_bytes),
            name="qml-control-cmd-poll",
            daemon=True,
        )
        self._control_thread.start()

    def shutdown(self):
        """Stops the control-poll thread and closes the store.

        in: none
        out: None
        side effects: closes the process-global SQLite+LMDB store connection
        """
        self._control_stop.set()
        self._control_thread.join(timeout=5.0)
        self._store.close()

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        """Worker-side load: restores each request's matched KV blocks from the store into the GPU KV cache.

        in: forward_context -- vLLM forward context; supplies attn_metadata and
            the per-layer GPU KV cache tensors (no_compile_layers)
        in: kwargs -- unused, accepted for API compatibility
        out: None
        side effects: writes restored KV bytes into the GPU KV cache pages of
            every matched block (LMDB reads pipelined with host->device copies);
            the stored format (int8 vs fp16) is auto-detected by blob size
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, QmlConnectorMetadata)

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning("start_load_kv: attn_metadata is None, skipping")
            return

        for req in metadata.requests:
            if req.is_store or req.matched_blocks == 0:
                continue

            # QML_LOAD_TIMING=1 splits the load into store-read vs
            # host->device copy. The two are what decide whether reusing a
            # stored prefix beats simply recomputing it on the GPU, and that
            # trade-off is far tighter here than on llama.cpp/CPU.
            timing = os.environ.get("QML_LOAD_TIMING") == "1"
            t_load0, t_read, t_copy, n_bytes, n_blocks = time.perf_counter(), 0.0, 0.0, 0, 0
            slot_mapping = None

            # collect the layers first so the loop below can prefetch layer
            # i+1's blocks from the store while layer i's bytes are on the
            # PCIe bus (double buffer, one prefetch worker)
            layer_items = []
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_layer = getattr(layer, "kv_cache", None)
                if kv_cache_layer is None:
                    continue
                layer_items.append((layer_name, kv_cache_layer))

            host_bufs: list = [None, None]
            min_loaded = req.matched_blocks

            def _read_layer(idx: int) -> tuple:
                """Reads one layer's matched blocks from the store into a pinned host buffer.

                in: idx -- index into layer_items selecting the layer to read
                out: (segments, per_block, g, dt_read) -- segments is a list of
                    (fmt, start_block, n_blocks) runs whose bytes sit at
                    start_block*stride(fmt) inside that format's host buffer;
                    per_block is the fp element count per block; g is head_dim;
                    dt_read is the store-read wall time in seconds
                """
                # format is detected BY SIZE per run of blocks: the store's
                # size check makes blocks_get_into() stop at the first value
                # whose byte count does not match the stride it was given, so
                # alternating between the int8 stride and the fp16 stride
                # walks a mixed store segment by segment. Returns
                # (segments, per_block, g, dt_read) where each segment is
                # (fmt, start_block, n_blocks) and its bytes live at
                # start_block*stride(fmt) inside that format's buffer.
                layer_name, kv_cache_layer = layer_items[idx]
                g = kv_cache_layer.shape[3]
                per_block = self._block_size * kv_cache_layer.shape[1] * g
                stride_q8 = per_block + (per_block // g) * 2
                stride_fp = per_block * kv_cache_layer.element_size()

                bufs = host_bufs[idx % 2]
                if bufs is None:
                    bufs = {}
                    host_bufs[idx % 2] = bufs

                def _buf(fmt):
                    """Returns (allocating or growing if needed) the pinned host buffer for one format.

                    in: fmt -- "q8" (int8-packed) or anything else (raw layer dtype)
                    out: pinned CPU torch.Tensor sized for all matched blocks
                    """
                    if fmt == "q8":
                        numel, dtype = req.matched_blocks * stride_q8, torch.uint8
                    else:
                        numel, dtype = req.matched_blocks * per_block, kv_cache_layer.dtype
                    b = bufs.get(fmt)
                    if b is None or b.numel() < numel or b.dtype != dtype:
                        # pinned so the copy is a real DMA, not a staging copy
                        b = torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)
                        bufs[fmt] = b
                    return b

                keys = [_layer_block_key(layer_name, req.chain[i])
                        for i in range(req.matched_blocks)]
                order = ("q8", "fp16") if self._kv_compress == "int8" else ("fp16", "q8")

                t0 = time.perf_counter()
                segments = []
                pos = 0
                while pos < req.matched_blocks:
                    advanced = False
                    for fmt in order:
                        stride = stride_q8 if fmt == "q8" else stride_fp
                        buf = _buf(fmt)
                        got = self._store.blocks_get_into(
                            keys[pos:],
                            buf.data_ptr() + pos * stride, stride,
                            threads=self._read_threads,
                        )
                        if got > 0:
                            segments.append((fmt, pos, got))
                            pos += got
                            advanced = True
                            break
                    if not advanced:
                        break  # genuine miss (or unknown format): engine recomputes the rest
                return segments, per_block, g, time.perf_counter() - t0

            pending = self._load_pool.submit(_read_layer, 0) if layer_items else None

            for idx in range(len(layer_items)):
                layer_name, kv_cache_layer = layer_items[idx]
                a_meta = attn_metadata[layer_name] if isinstance(attn_metadata, dict) else attn_metadata

                # NOTE: verified live (non-MLA path, TRITON_ATTN backend on
                # this model) that kv_cache_layer.shape is (num_pages,
                # num_kv_heads, page_size, head_dim) -- NOT the (num_pages,
                # 2[k/v], page_size, head_dim) layout example_connector.py's
                # comment assumes. layer[block_idxs, :, offsets] (see
                # _extract_kv) advanced-indexes dims 0 and 2,
                # keeps dim 1 (num_kv_heads) and dim 3 (head_dim) whole, so
                # the saved/reloaded per-block shape is
                # (block_size, num_kv_heads, head_dim) == shape[1], shape[3]
                # -- NOT shape[2:], which was the original (wrong) guess
                # that crashed live with a reshape size mismatch.
                # The destination KV cache buffer lives on the GPU -- confirmed
                # live: assigning a CPU tensor into it raises "Expected all
                # tensors to be on the same device".
                # Second line of defence for task #53: even with the hardened
                # cache seed, refuse bytes whose size does not match what THIS
                # layer expects, instead of reinterpreting a differently-shaped
                # or differently-typed payload. That check is the `v.mv_size !=
                # stride` test inside qml_lmdb_get_many() now that the read is
                # batched -- a mismatch ends the run and is reported below.
                # task #65: the whole layer in ONE store read and ONE host->
                # device copy (14,042 per-block round trips made restore slower
                # than recompute). NOW pipelined on top: this result was read
                # while the PREVIOUS layer's bytes were being uploaded.
                segments, per_block, g, dt_read = pending.result()
                t_read += dt_read
                layer_bufs = host_bufs[idx % 2]
                stride_q8 = per_block + (per_block // g) * 2
                loaded = sum(s[2] for s in segments)
                min_loaded = min(min_loaded, loaded)

                # prefetch the next layer into the OTHER buffer pair; safe
                # because the blocking .to(device) below finishes before this
                # buffer index comes around again
                if idx + 1 < len(layer_items):
                    pending = self._load_pool.submit(_read_layer, idx + 1)

                # loaded < matched_blocks means a block was missing or its
                # byte count matched NEITHER format this layer expects (task
                # #53's second line of defence): load what is there and let
                # the engine recompute the rest.
                if loaded < req.matched_blocks:
                    logger.warning(
                        "start_load_kv: layer '%s' supplied %d of %d matched blocks "
                        "(missing or unknown size for dtype=%s, block_size=%d, heads=%d, "
                        "head_dim=%d) -- loading the prefix and recomputing the rest",
                        layer_name, loaded, req.matched_blocks, kv_cache_layer.dtype,
                        self._block_size, kv_cache_layer.shape[1], kv_cache_layer.shape[3],
                    )
                if loaded == 0:
                    continue

                t1 = time.perf_counter()
                if slot_mapping is None or slot_mapping.numel() < loaded * self._block_size:
                    # positions for every restored token, built once per request
                    slot_mapping = torch.cat([
                        torch.arange(self._block_size, dtype=torch.long)
                        + req.block_ids[i] * self._block_size
                        for i in range(req.matched_blocks)
                    ]).to(kv_cache_layer.device)

                # blocking on purpose: the buffers are reused two layers later,
                # so each DMA must finish before they are overwritten
                for fmt, start, got in segments:
                    if fmt == "q8":
                        buf = layer_bufs["q8"]
                        raw = buf[start * stride_q8: (start + got) * stride_q8].to(
                            kv_cache_layer.device)
                        gpu = self._dequant_int8(
                            raw, got, per_block, g, kv_cache_layer.dtype
                        ).reshape(got * self._block_size, kv_cache_layer.shape[1], g)
                        n_bytes += got * stride_q8
                    else:
                        buf = layer_bufs["fp16"]
                        gpu = buf[start * per_block: (start + got) * per_block].to(
                            kv_cache_layer.device
                        ).reshape(got * self._block_size, kv_cache_layer.shape[1], g)
                        n_bytes += got * per_block * kv_cache_layer.element_size()
                    self._inject_kv(
                        kv_cache_layer, gpu,
                        slot_mapping[start * self._block_size:
                                     (start + got) * self._block_size],
                        a_meta,
                    )
                    n_blocks += got
                t_copy += time.perf_counter() - t1

            if min_loaded < req.matched_blocks:
                # blocks some layer could not supply hold no valid KV on the GPU
                self._load_error_blocks.update(req.block_ids[min_loaded:req.matched_blocks])
            total = time.perf_counter() - t_load0
            if n_bytes > 0 and t_read > 0:
                # store-read rate only: the host-to-device copy is not what the
                # recompute alternative would avoid
                _cost_update(restore_bps=n_bytes / t_read)
            if timing:
                # warning, not info: this module's logger is not one of the
                # "vllm.*" loggers vLLM configures, so it inherits root's
                # level and info() is swallowed
                logger.warning(
                    "qml load: %.0f ms total (store read %.0f ms, to-GPU %.0f ms) "
                    "for %d block-layers, %.1f MiB -> %.0f MiB/s",
                    total * 1e3, t_read * 1e3, t_copy * 1e3, n_blocks,
                    n_bytes / 1048576, n_bytes / 1048576 / max(total, 1e-9),
                )

    def get_block_ids_with_load_errors(self) -> set:
        """Worker-side: GPU block ids promised to the scheduler whose KV could not be loaded.

        in: none
        out: set of block ids since the previous call (the set is cleared)
        """
        errs, self._load_error_blocks = self._load_error_blocks, set()
        return errs

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No-op: start_load_kv already blocks until each layer's copy is done.

        in: layer_name -- vLLM layer name (unused)
        out: None
        """
        return

    def _save_writer(self) -> None:
        """Background writer-thread body: drains the save queue into LMDB/SQLite.

        in: none
        out: None (never returns; daemon thread)
        side effects: durably writes queued KV block bytes ("blocks" jobs) and
            manifest markers / session rows / token-log rows ("meta" jobs) to
            LMDB+SQLite, and releases the buffered-bytes budget so
            save_kv_layer can enqueue more
        """
        while True:
            job = self._save_q.get()
            kind = job[0]
            try:
                if kind == "blocks":
                    _, keys, host, stride, owner = job
                    self._store.blocks_put_from(keys, host.data_ptr(), stride)
                    if owner:
                        # ownership lets a window trim drop these blocks
                        self._store.owner_add(owner, keys)
                    with self._save_cv:
                        self._save_buffered -= host.numel() * host.element_size()
                        self._save_cv.notify_all()
                elif kind == "meta":
                    _, marker_keys, touch_kwargs, token_args = job
                    if marker_keys:
                        markers = torch.frombuffer(
                            bytearray(b"1" * len(marker_keys)), dtype=torch.uint8
                        )
                        self._store.blocks_put_from(marker_keys, markers.data_ptr(), 1)
                        if touch_kwargs:
                            self._store.owner_add(touch_kwargs["session_id"], marker_keys)
                    if touch_kwargs is not None:
                        self._store.session_touch(**touch_kwargs)
                    if token_args is not None:
                        self._store.tokens_put(*token_args)
            except Exception:
                logger.exception("qml save writer: %s job failed", kind)
                if kind == "blocks":
                    with self._save_cv:
                        self._save_buffered -= job[2].numel() * job[2].element_size()
                        self._save_cv.notify_all()

    def _enqueue_blocks(self, keys, host: torch.Tensor, stride: int, deadline: float,
                        owner=None) -> bool:
        """Queues one layer's host KV bytes for the background writer, honoring the RAM budget.

        in: keys -- per-block LMDB keys, aligned with the blocks packed in host
        in: host -- contiguous CPU tensor holding all blocks back to back
        in: stride -- bytes per block inside host
        in: deadline -- monotonic time after which this request stops waiting
        in: owner -- session id recorded as the blocks' owner (None: not recorded)
        out: True when the bytes were queued; False when the buffer stayed
            full for _SAVE_WAIT_BUDGET_S, in which case the caller must stop
            saving this request instead of waiting inside the forward pass
        side effects: increments the buffered-bytes counter and enqueues a
            "blocks" job for _save_writer to persist
        """
        nbytes = host.numel() * host.element_size()
        with self._save_cv:
            while self._save_buffered > 0 and self._save_buffered + nbytes > self._save_ram_budget:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._save_cv.wait(timeout=min(left, 0.05))
            self._save_buffered += nbytes
        self._save_q.put(("blocks", keys, host, stride, owner))
        return True

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata, **kwargs: Any) -> None:
        """Worker-side save: extracts this layer's newly computed KV blocks and queues them for durable write.

        in: layer_name -- vLLM layer name (part of each block's storage key)
        in: kv_layer -- this layer's GPU KV cache tensor
        in: attn_metadata -- attention metadata (selects MLA vs standard layout)
        in: kwargs -- unused, accepted for API compatibility
        out: None
        side effects: device->host copy (plus optional int8 quantization on the
            GPU) of every storable block, then enqueues the bytes for the
            background LMDB writer; QML_SAVE_BATCH=0 falls back to synchronous
            per-block LMDB/SQLite writes inside the forward pass
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, QmlConnectorMetadata)

        for req in metadata.requests:
            if not req.is_store:
                continue
            if req.request_id in self._save_dropped:
                # this request already gave up on persisting; saving the rest
                # of its layers would write blocks no marker will ever vouch for
                continue

            n = min(len(req.chain), len(req.block_ids))
            if n == 0:
                continue

            t0 = time.perf_counter()
            self._num_layers_seen += 1

            if os.environ.get("QML_SAVE_BATCH") == "0":
                # kept only so the batched path can be A/B'd against what it
                # replaced; one LMDB commit + one SQLite commit per block
                for i in range(n):
                    sm = torch.arange(self._block_size, dtype=torch.long) + req.block_ids[i] * self._block_size
                    self._store.block_put(
                        _layer_block_key(layer_name, req.chain[i]),
                        _tensor_to_bytes(self._extract_kv(kv_layer, sm, attn_metadata)),
                    )
                self._save_ms += (time.perf_counter() - t0) * 1e3
                continue

            # symmetric to start_load_kv: extract the whole layer at once and
            # write it under ONE LMDB txn. Per block this cost a device->host
            # copy, a numpy round trip, a durable LMDB commit and a SQLite
            # recency commit -- all inside the forward pass.
            slot_mapping = torch.cat([
                torch.arange(self._block_size, dtype=torch.long)
                + req.block_ids[i] * self._block_size
                for i in range(n)
            ]).to(kv_layer.device)
            kv_cache = self._extract_kv(kv_layer, slot_mapping, attn_metadata)

            if self._kv_compress == "int8":
                packed = self._quant_int8(kv_cache.detach(), n)
                host = packed.to("cpu", copy=True).contiguous()
                stride = host.shape[1]
            else:
                host = kv_cache.detach().to("cpu", copy=True).contiguous()
                stride = host.numel() // n * host.element_size()
            # durable write happens on the qml-save-writer thread; only the
            # quantize + device->host copy above stays in the forward pass
            deadline = self._save_deadline.get(req.request_id)
            if deadline is None:
                deadline = time.monotonic() + _SAVE_WAIT_BUDGET_S
                self._save_deadline[req.request_id] = deadline
            if not self._enqueue_blocks(
                [_layer_block_key(layer_name, req.chain[i]) for i in range(n)],
                host, stride, deadline, owner=req.session_id,
            ):
                self._save_dropped.add(req.request_id)
                self._save_dropped_total += 1
                logger.warning(
                    "qml save: buffer full for %.0f ms, dropping the KV save of "
                    "request %s (%d blocks); the token log still records the turn",
                    _SAVE_WAIT_BUDGET_S * 1e3, req.request_id, n)
            self._save_ms += (time.perf_counter() - t0) * 1e3

    def wait_for_save(self):
        """Ends the save step: queues manifest markers plus session/token-log metadata behind the layer data.

        in: none
        out: None
        side effects: enqueues per-block manifest markers (which is what makes
            blocks matchable), a qml_session upsert and a qml_token_log bulk
            insert (final chunk only) on the same FIFO queue as the block
            bytes, so a marker never becomes durable before the data it vouches for
        """
        # write one manifest sentinel per block now that all layers for
        # this step have been saved (see _manifest_block_key docstring)
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, QmlConnectorMetadata)

        for req in metadata.requests:
            if not req.is_store:
                continue

            # mark blocks cached only after all layers for this step have
            # been saved -- one marker per block, not one for the whole
            # prefix, so a later request sharing only the first K blocks
            # of this one is detected by _find_matched_blocks. Enqueued
            # BEHIND the layer data on the same FIFO queue: the marker only
            # becomes durable after every byte it vouches for.
            self._save_deadline.pop(req.request_id, None)
            if req.request_id in self._save_dropped:
                # data for this request is incomplete: write no marker, so no
                # later request can match it. The session row and token log
                # below still record the turn -- that is the canonical record.
                marker_keys = None
            else:
                marker_keys = [_manifest_block_key(h) for h in req.chain] if req.chain else None

            if os.environ.get("QML_LOAD_TIMING") == "1" and self._save_ms:
                logger.warning("qml save: %.0f ms in save_kv_layer (%d blocks x %d layers)",
                               self._save_ms, len(req.chain), self._num_layers_seen)
                self._save_ms = 0.0

            touch_kwargs = None
            token_args = None
            # set on the final chunk only (see ReqMeta.total_blocks)
            if req.session_id:
                # user_id passthrough (2026-08-02): vLLM's Request carries no
                # user field to a connector, so adopt a salt convention --
                # cache_salt "user123::sess42" records user_id="user123" in
                # qml_session while the FULL salt stays the session/isolation
                # key (unchanged hashing, so this cannot alter which KV is
                # matched). Salts without "::" behave exactly as before.
                _uid = ""
                if "::" in req.session_id:
                    _uid = req.session_id.split("::", 1)[0]
                touch_kwargs = dict(
                    session_id=req.session_id,
                    user_id=_uid,
                    # session metadata records the STORAGE identity (cache seed),
                    # not the served model name -- this is the value a later
                    # restore must match against, same role as llama.cpp's
                    # qml_model_fingerprint in qml_session table
                    model_hash=self._cache_seed[:64],
                    lmdb_prefix=req.session_id,
                    block_size=self._block_size,
                    num_blocks=req.total_blocks or len(req.chain),
                    n_tokens=len(req.token_ids),
                )
                # persist the actual token sequence into qml_token_log --
                # the SAME engine-agnostic table llama.cpp's path already
                # writes. This is what makes cross-engine handoff possible
                # at all (KV tensor bytes are NOT portable across engines;
                # the token sequence is the only portable currency -- see
                # docs/4-engines/engine-comparison.md). Position
                # offset continuation across engines is deliberately NOT
                # done here (see tokens_put docstring).
                if self._record_token_log:
                    token_args = (req.session_id, req.token_ids)

            if marker_keys or touch_kwargs or token_args:
                self._save_q.put(("meta", marker_keys, touch_kwargs, token_args))

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> tuple[int | None, bool]:
        """Scheduler hook: answers how many MORE prompt tokens the store can supply beyond what vLLM computed.

        in: request -- vLLM Request (prompt_token_ids, cache_salt, mm_features read)
        in: num_computed_tokens -- tokens vLLM already has locally (its own prefix cache)
        out: (n, False) -- n is the number of additional leading prompt tokens
            whose KV blocks are cached in the store (0 for multimodal requests
            or when the cached prefix does not exceed num_computed_tokens);
            the False means the load is synchronous, never async
        side effects: caches (chain, matched_blocks) in self._match_cache so
            build_connector_meta does not walk the store a second time
        """
        if request.mm_features:
            # SAFETY: multimodal models leave only a placeholder token
            # sequence in the text stream where an image/audio embedding
            # goes; the real content lives outside prompt_token_ids. Our
            # block-chain hash only ever looks at prompt_token_ids, so two
            # DIFFERENT images sharing the same placeholder tokens would
            # hash-collide and silently serve the WRONG image's cached KV
            # (a wrong-answer bug, not a crash -- confirmed by inspecting
            # mm_features' role, not guessed). llama.cpp's own path guards
            # the equivalent case with !input_tokens.has_media() -- mirror
            # that here by refusing to serve anything external for
            # multimodal requests, falling back to native full compute.
            return 0, False

        token_ids = list(request.prompt_token_ids or [])
        num_tokens_to_check = _align_to_block_size(len(token_ids) - 1, self._block_size)

        if num_tokens_to_check <= num_computed_tokens:
            return 0, False

        seed = _scoped_seed(self._cache_seed, self._share_scope,
                            getattr(request, "cache_salt", None), self._tenant_id)
        t0 = time.perf_counter()
        _chain, matched_blocks = _find_matched_blocks(
            self._store, seed, token_ids[:num_tokens_to_check], self._block_size
        )
        matched_tokens = matched_blocks * self._block_size
        if os.environ.get("QML_LOAD_TIMING") == "1":
            # scheduler-side cost, paid before the load loop even starts: one
            # manifest lookup per block. Counted separately because it lands
            # in TTFT just as much as the transfer does.
            logger.warning("qml match: %.0f ms for %d blocks",
                           (time.perf_counter() - t0) * 1e3, matched_blocks)

        # build_connector_meta() needs exactly this chain and count a moment
        # later, in the same process; walking the store twice per request was
        # measured at another 93 ms
        self._match_cache[request.request_id] = (_chain, matched_blocks)

        if matched_tokens <= num_computed_tokens:
            return 0, False

        extra = matched_tokens - num_computed_tokens
        if not _restore_is_worth_it(extra, self._bytes_per_token):
            # reading those bytes back would cost more than recomputing them on
            # this device and storage. The blocks stay in the store for a run
            # where the arithmetic goes the other way; build_connector_meta
            # still starts this request's save past the matched prefix, so
            # nothing already stored is written again.
            return 0, False

        return extra, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        """Scheduler hook: remembers which requests were allocated blocks for an external (store) load.

        in: request -- vLLM Request whose KV blocks were just allocated
        in: blocks -- the allocated KV cache blocks (unused here)
        in: num_external_tokens -- tokens the connector promised to supply
        out: None; requests with num_external_tokens > 0 are marked in
            self._requests_need_load for the load path in build_connector_meta
        """
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def on_new_request(self, request) -> None:
        """Captures the request's cache_salt (qml's session id) while the full Request object is available.

        in: request -- vLLM Request; only request_id and cache_salt are read
        out: None (the salt is stored in self._request_cache_salt for
            build_connector_meta and released on request finish)
        """
        # this is the ONLY place the full Request object (with .cache_salt)
        # is available -- scheduler_output.scheduled_new_reqs' elements
        # (NewRequestData) do not carry it. Captured here, consumed and
        # popped in build_connector_meta below.
        if request.cache_salt:
            self._request_cache_salt[request.request_id] = request.cache_salt

    def _emit_store_chunk(self, meta, req_id: str, state: dict, num_computed: int, num_scheduled: int) -> None:
        """Queue the blocks this step actually finished computing.

        A block is storable only once every one of its tokens has been through
        a forward pass, i.e. once num_computed+num_scheduled has passed its end
        -- and only if its slot has been allocated (block_ids grows chunk by
        chunk). The last chunk carries session_id so the session row and the
        token log are written exactly once, with the full block count.

        in: meta -- QmlConnectorMetadata being built; a store ReqMeta is appended
        in: req_id -- vLLM request id
        in: state -- this request's _store_progress entry
            (chain/blocks/token_ids/session_id/saved_blocks)
        in: num_computed -- tokens already computed before this step
        in: num_scheduled -- tokens scheduled for computation in this step
        out: None; advances state["saved_blocks"] past the emitted block range
        """
        if req_id in self._save_dropped:
            return
        chain, blocks = state["chain"], state["blocks"]
        end = min((num_computed + num_scheduled) // self._block_size, len(chain), len(blocks))
        start = state["saved_blocks"]
        if end <= start:
            return
        state["saved_blocks"] = end
        final = end >= len(chain)
        meta.requests.append(
            ReqMeta(
                request_id=req_id,
                token_ids=state["token_ids"],
                block_ids=blocks[start:end],
                chain=chain[start:end],
                matched_blocks=0,
                is_store=True,
                session_id=state["session_id"] if final else None,
                total_blocks=len(chain),
            )
        )

    def build_connector_meta(self, scheduler_output) -> QmlConnectorMetadata:
        """Scheduler hook: turns this step's scheduler output into per-request load/store work items.

        in: scheduler_output -- vLLM SchedulerOutput (new/cached/finished
            requests and per-request scheduled token counts)
        out: QmlConnectorMetadata whose .requests holds a load ReqMeta for each
            request with matched blocks and a store ReqMeta for each chunk of
            newly computed blocks (chunked-prefill and preemption-resume aware)
        side effects: updates the per-request _store_progress / _match_cache /
            _request_cache_salt bookkeeping and clears _requests_need_load
        """
        meta = QmlConnectorMetadata()
        num_sched = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        # the engine's own token rate, measured between scheduler steps: this is
        # the alternative the restore gate is judged against
        now = time.perf_counter()
        prev = getattr(self, "_last_step_t", None)
        prev_tokens = getattr(self, "_last_step_tokens", 0)
        if prev is not None and prev_tokens > 0:
            dt = now - prev
            if dt > 0:
                _cost_update(engine_tps=prev_tokens / dt)
        self._last_step_t = now
        self._last_step_tokens = sum(num_sched.values()) if num_sched else 0

        # later chunks of a prompt already seen as a new request: pick up the
        # blocks allocated since, and store whatever finished computing
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None:
            req_ids = getattr(cached, "req_ids", []) or []
            new_block_ids = getattr(cached, "new_block_ids", None) or [None] * len(req_ids)
            computed = getattr(cached, "num_computed_tokens", None) or [0] * len(req_ids)
            # vLLM's own CachedRequestData docstring: for req_ids in
            # resumed_req_ids, new_block_ids REPLACES the request's block
            # list; for everyone else it appends. Appending for a resumed
            # request would leave stale freed-page ids at the front of the
            # list, and every block saved after that would be extracted from
            # whatever request reused those pages (bug #66's second half).
            resumed = getattr(cached, "resumed_req_ids", None) or set()
            for idx, req_id in enumerate(req_ids):
                state = self._store_progress.get(req_id)
                if state is None:
                    continue
                grown = new_block_ids[idx] if idx < len(new_block_ids) else None
                if grown:
                    flat = grown[0] if isinstance(grown, (tuple, list)) and grown and isinstance(grown[0], (tuple, list)) else grown
                    if req_id in resumed:
                        state["blocks"] = list(flat)
                    else:
                        state["blocks"].extend(flat)
                elif req_id in resumed:
                    state["blocks"] = []
                n_done = computed[idx] if idx < len(computed) else 0
                self._emit_store_chunk(meta, req_id, state, n_done, num_sched.get(req_id, 0))
                if state["saved_blocks"] >= len(state["chain"]):
                    self._store_progress.pop(req_id, None)

        for new_req in scheduler_output.scheduled_new_reqs:
            # same safety guard as get_num_new_matched_tokens above -- skip
            # this request entirely so save_kv_layer/start_load_kv/
            # wait_for_save (which only iterate meta.requests) never touch
            # its KV at all; it always falls through to native full compute
            if new_req.mm_features:
                self._request_cache_salt.pop(new_req.req_id, None)
                continue

            # get, NOT pop (bug #66): a request preempted mid-prefill comes
            # back through scheduled_new_reqs a second time, and a popped
            # salt made that second pass compute a session=None seed -- the
            # rest of the prompt was then stored as an orphan chain no later
            # request could ever match, and the session row/token log were
            # never written at all. The salt is released in the
            # finished_req_ids loop below instead.
            token_ids = new_req.prompt_token_ids or []
            session_id = self._request_cache_salt.get(new_req.req_id)
            num_tokens_to_check = _align_to_block_size(len(token_ids) - 1, self._block_size)

            t_meta = time.perf_counter()
            if new_req.req_id in self._requests_need_load:
                # reuse what get_num_new_matched_tokens already computed for
                # this very request; re-walking the store here cost another
                # 93 ms of TTFT for the same answer. Fall back to a fresh walk
                # only if the entry is missing (a request that reached here
                # without passing through the scheduler hook).
                cached = self._match_cache.pop(new_req.req_id, None)
                if cached is not None:
                    chain, matched_blocks = cached
                else:
                    chain, matched_blocks = _find_matched_blocks(
                        self._store,
                        _scoped_seed(self._cache_seed, self._share_scope, session_id, self._tenant_id),
                        token_ids[:num_tokens_to_check], self._block_size
                    )
                meta.requests.append(
                    ReqMeta(
                        request_id=new_req.req_id,
                        token_ids=token_ids,
                        block_ids=new_req.block_ids[0],
                        chain=chain,
                        matched_blocks=matched_blocks,
                        is_store=False,
                        session_id=session_id,
                    )
                )
                # A restoring request must ALSO store what the engine computes
                # past the restored prefix. Without this, load and store were
                # exclusive: turn 2 of a session matched turn 1, restored it,
                # and persisted nothing, so turn 3 still matched only turn 1 and
                # re-prefilled everything after it -- the recomputed tail grew
                # with every turn while the store stayed frozen at turn 1
                # (measured: multi-turn load, TEST-LEDGER row 89). Starting the
                # store at matched_blocks also guarantees the restored prefix is
                # never rewritten.
                state = self._store_progress.get(new_req.req_id)
                if state is None:
                    state = {
                        "chain": chain,
                        "blocks": list(new_req.block_ids[0]),
                        "token_ids": token_ids,
                        "session_id": session_id,
                        "saved_blocks": matched_blocks,
                    }
                else:
                    state["blocks"] = list(new_req.block_ids[0])
                    state["saved_blocks"] = max(state["saved_blocks"], matched_blocks)
                self._store_progress[new_req.req_id] = state
                self._emit_store_chunk(meta, new_req.req_id, state,
                                       getattr(new_req, "num_computed_tokens", 0) or 0,
                                       num_sched.get(new_req.req_id, 0))
                if state["saved_blocks"] >= len(state["chain"]):
                    self._store_progress.pop(new_req.req_id, None)
            else:
                prior = self._store_progress.get(new_req.req_id)
                if prior is not None:
                    # this req_id was already mid-store and is being scheduled
                    # as "new" again -- that is a preemption resume. Its old
                    # blocks were freed, so the allocation REPLACES the list;
                    # the chain is unchanged (same tokens, same salt now that
                    # the salt is no longer popped) and the blocks saved
                    # before the preemption are still valid, so saved_blocks
                    # carries over and those blocks are not rewritten.
                    state = prior
                    state["blocks"] = list(new_req.block_ids[0])
                else:
                    # reuse the walk the scheduler hook already did: it also
                    # tells us how many leading blocks the store already holds,
                    # which is where this request's save must begin. A request
                    # the restore gate turned down lands here with a non-zero
                    # match, and rewriting those blocks would be pure waste.
                    cached = self._match_cache.pop(new_req.req_id, None)
                    if cached is not None:
                        chain, already = cached
                    else:
                        chain = _compute_chain(
                            _scoped_seed(self._cache_seed, self._share_scope, session_id, self._tenant_id),
                            token_ids[:num_tokens_to_check], self._block_size)
                        already = 0
                    state = {
                        "chain": chain,
                        "blocks": list(new_req.block_ids[0]),
                        "token_ids": token_ids,
                        "session_id": session_id,
                        "saved_blocks": already,
                    }
                self._store_progress[new_req.req_id] = state
                self._emit_store_chunk(meta, new_req.req_id, state,
                                       getattr(new_req, "num_computed_tokens", 0) or 0,
                                       num_sched.get(new_req.req_id, 0))
                if state["saved_blocks"] >= len(state["chain"]):
                    self._store_progress.pop(new_req.req_id, None)

            if os.environ.get("QML_LOAD_TIMING") == "1":
                logger.warning("qml meta: %.0f ms (req %s)",
                               (time.perf_counter() - t_meta) * 1e3, new_req.req_id)

        for req_id in getattr(scheduler_output, "finished_req_ids", None) or ():
            self._save_dropped.discard(req_id)
            self._save_deadline.pop(req_id, None)
            self._store_progress.pop(req_id, None)
            self._match_cache.pop(req_id, None)
            self._request_cache_salt.pop(req_id, None)

        self._requests_need_load.clear()
        return meta

    # ==============================
    # Helper functions (tensor layout mirrors vLLM's own
    # example_connector.py -- MLA vs non-MLA branching copied from that
    # reference, NOT verified against a live tensor here)
    # ==============================

    def _quant_int8(self, kv: torch.Tensor, n_blocks: int) -> torch.Tensor:
        """symmetric absmax int8 over the last dim (head_dim groups), packed
        per block as [int8 data | fp16 scales]. Runs on the GPU so the
        device->host copy that follows already moves the HALVED bytes.

        in: kv -- extracted KV tensor covering n_blocks blocks (on the GPU)
        in: n_blocks -- number of blocks packed
        out: (n_blocks, per_block + per_block//head_dim*2) uint8 tensor, each
            row laid out as [int8 data | fp16 scales]
        """
        g = kv.shape[-1]
        x = kv.reshape(n_blocks, -1, g).float()
        scale = x.abs().amax(dim=2, keepdim=True) / 127.0
        scale = torch.clamp(scale, min=1e-8)
        q = torch.clamp((x / scale).round(), -127, 127).to(torch.int8)
        return torch.cat([
            q.reshape(n_blocks, -1).view(torch.uint8),
            scale.to(torch.float16).reshape(n_blocks, -1).view(torch.uint8),
        ], dim=1).contiguous()

    def _dequant_int8(self, raw_u8: torch.Tensor, n_blocks: int,
                                per_block: int, g: int, dtype) -> torch.Tensor:
        """inverse of _quant_int8; raw_u8 is (n_blocks*row,) uint8
        on the GPU, returns (n_blocks*per_block/(...)) fp tensor grouped as
        (n_blocks, per_block//g, g) flattened back to element order.

        in: raw_u8 -- packed uint8 rows as produced by _quant_int8
        in: n_blocks -- number of blocks contained in raw_u8
        in: per_block -- fp element count per block
        in: g -- group size (head_dim) the scales were computed over
        in: dtype -- target torch dtype of the dequantized values
        out: (n_blocks, per_block) tensor of dtype with dequantized values
        """
        row = per_block + (per_block // g) * 2
        m = raw_u8.reshape(n_blocks, row)
        q = m[:, :per_block].contiguous().view(torch.int8).float().reshape(n_blocks, -1, g)
        s = m[:, per_block:].contiguous().view(torch.float16).float().reshape(n_blocks, -1, 1)
        return (q * s).to(dtype).reshape(n_blocks, -1)

    def _inject_kv(self, dst_kv_cache_layer, src_kv_cache, slot_mapping, attn_metadata) -> None:
        """Scatters restored KV values into the GPU KV cache at the given token slots.

        in: dst_kv_cache_layer -- this layer's GPU KV cache tensor
        in: src_kv_cache -- restored KV values, one row per token slot
        in: slot_mapping -- absolute token slot index for each row
        in: attn_metadata -- selects MLA (flat) vs standard (paged) indexing
        out: None
        side effects: overwrites the addressed slots of the GPU KV cache
        """
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = dst_kv_cache_layer.shape[0], dst_kv_cache_layer.shape[1]
            dst = dst_kv_cache_layer.reshape(num_pages * page_size, -1)
            dst[slot_mapping, ...] = src_kv_cache
        else:
            block_idxs = slot_mapping // self._block_size
            offsets = slot_mapping % self._block_size
            dst_kv_cache_layer[block_idxs, :, offsets] = src_kv_cache

    def _extract_kv(self, layer: torch.Tensor, slot_mapping, attn_metadata) -> torch.Tensor:
        """Gathers KV values for the given token slots out of the GPU KV cache.

        in: layer -- this layer's GPU KV cache tensor
        in: slot_mapping -- absolute token slot index for each wanted row
        in: attn_metadata -- selects MLA (flat) vs standard (paged) indexing
        out: tensor of the addressed slots' KV values (one row per slot)
        """
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = layer.shape[0], layer.shape[1]
            return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
        block_idxs = slot_mapping // self._block_size
        offsets = slot_mapping % self._block_size
        return layer[block_idxs, :, offsets]
