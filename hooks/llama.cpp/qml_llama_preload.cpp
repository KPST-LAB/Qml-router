// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.

// hooks/llama.cpp/qml_llama_preload.cpp
//
// LD_PRELOAD symbol interceptor — no llama.cpp source file is modified.
// Public API only (llama.h); private headers are not needed here because
// state save/restore goes through llama_state_seq_get_data/set_data.
//
// Enabled only when QML_SESSION_ID is set in the environment (see
// test_llama_session.sh) — otherwise every hook is a transparent passthrough
// to the real llama.cpp symbol.
//
// Multi-slot design (llama-server): a single llama_memory_t can host many
// concurrent seq_id "slots", each potentially a different user/session.
// Every (mem, seq_id) pair gets its own qml_session_slot with its own
// mutex ("lock n free" — locked for the duration of one hook call touching
// that slot, released at the end of scope). Unrelated slots never block
// each other; only operations on the SAME seq_id serialize. Restore is
// applied lazily, inside the FIRST llama_decode() call for a given seq_id
// (deferred from context-creation time because the KV cache's backend
// buffers are not guaranteed allocated right after llama_init_from_model
// returns). "Session end" fires only on llama_free (whole-context/process
// end), which persists every slot still registered against that context.
//
// A full-range llama_memory_seq_rm is deliberately NOT treated as
// "the host is done with this slot" -- llama.cpp/llama-cli issue that
// exact call shape as part of their own routine internal cleanup even in
// a single-session run with no slot handoff involved (confirmed
// empirically; see docs/udf-control-plane-proposal.md). Persisting on
// eviction needs an external signal to distinguish "handing this slot to
// a new conversation" from "routine reset, llama_free still follows" --
// that is the deferred control-plane idea, not something this call alone
// can tell us.
//
// Session identity: QML_SESSION_ID is treated as a PREFIX; the actual
// per-slot session id is "{QML_SESSION_ID}#{seq_id}", so each concurrent
// slot in a multi-slot server gets its own persisted row. There is
// currently no external mechanism to tell this hook which real user/
// session a given seq_id belongs to (that is the deferred UDF control-
// plane idea in docs/udf-control-plane-proposal.md) — this prefix
// convention is a placeholder until that exists.
//
// Build:  libqml_llama_preload.so
// Run:    LD_PRELOAD=./libqml_llama_preload.so llama-cli / llama-server ...

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>

#include "llama.h"
#include "qml_router.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

//------------------------------------------------------------
// qml_slot_key / qml_session_slot
// one entry per (context-memory, seq_id) -- see file header comment
//------------------------------------------------------------
struct qml_slot_key
{
    llama_memory_t mem;
    llama_seq_id   seq_id;

    // key equality for the g_slots hash map
    //  in: o — other key
    // out: true when both the memory pointer and the seq_id match
    bool operator==(const qml_slot_key & o) const
    {
        return mem == o.mem && seq_id == o.seq_id;
    }
};

struct qml_slot_key_hash
{
    // hash for the g_slots map: mixes the memory pointer with the seq_id
    //  in: k — slot key to hash
    // out: combined hash value
    size_t operator()(const qml_slot_key & k) const
    {
        return std::hash<const void *>()(k.mem) ^ (std::hash<int32_t>()(k.seq_id) << 1);
    }
};

struct qml_session_slot
{
    std::mutex lock; // "lock n free": held only for the duration of one hook call touching this slot

    qml_llama_session_t * sess = nullptr;
    bool         restore_applied   = false;
    const void * pending_state_data = nullptr;
    size_t       pending_state_size = 0;
};

// structural lock: guards insert/erase into g_slots itself, NOT the
// per-slot critical sections (those use qml_session_slot::lock)
static std::mutex g_slots_mutex;
static std::unordered_map<qml_slot_key, std::unique_ptr<qml_session_slot>, qml_slot_key_hash> g_slots;

static std::once_flag g_init_flag;

//------------------------------------------------------------
// non-intercepted llama.cpp helper calls — resolved via dlsym like the
// intercepted ones, NOT linked directly. this .so carries no link-time
// dependency on any specific libllama.so build, so the same binary works
// whether LD_PRELOADed into the prebuilt llama-bin binaries or a freshly
// built third_party/llama.cpp checkout (their public headers can drift
// slightly, e.g. llama_model_params, without affecting this file).
//------------------------------------------------------------
static llama_memory_t (*p_llama_get_memory)(const struct llama_context *)                                    = nullptr;
static const struct llama_model * (*p_llama_get_model)(const struct llama_context *)                          = nullptr;
static size_t         (*p_llama_state_seq_get_size)(struct llama_context *, llama_seq_id)                     = nullptr;
static size_t         (*p_llama_state_seq_get_data)(struct llama_context *, uint8_t *, size_t, llama_seq_id)  = nullptr;
static size_t         (*p_llama_state_seq_set_data)(struct llama_context *, const uint8_t *, size_t, llama_seq_id) = nullptr;
static int32_t        (*p_llama_model_desc)(const struct llama_model *, char *, size_t)                       = nullptr;
static uint64_t       (*p_llama_model_n_params)(const struct llama_model *)                                    = nullptr;
static llama_pos      (*p_llama_memory_seq_pos_max)(llama_memory_t, llama_seq_id)                              = nullptr;

//------------------------------------------------------------
//Func : qml_ensure_init
//lazy one-time open of the sqlite/lmdb backing stores + dlsym resolution
//of the llama.cpp helper symbols this file calls directly
//  in: none
// out: none; side effects: opens llama_metadata.db + lmdb_mmap_store once per
//      process and fills the p_llama_* pointers via dlsym(RTLD_NEXT)
//------------------------------------------------------------
static void qml_ensure_init()
{
    std::call_once(g_init_flag, []() {
        if (qml_store_init("llama_metadata.db", "lmdb_mmap_store") != 0)
        {
            fprintf(stderr, "[qml] warning: failed to open session storage\n");
        }

        p_llama_get_memory         = reinterpret_cast<decltype(p_llama_get_memory)>(dlsym(RTLD_NEXT, "llama_get_memory"));
        p_llama_get_model          = reinterpret_cast<decltype(p_llama_get_model)>(dlsym(RTLD_NEXT, "llama_get_model"));
        p_llama_state_seq_get_size = reinterpret_cast<decltype(p_llama_state_seq_get_size)>(dlsym(RTLD_NEXT, "llama_state_seq_get_size"));
        p_llama_state_seq_get_data = reinterpret_cast<decltype(p_llama_state_seq_get_data)>(dlsym(RTLD_NEXT, "llama_state_seq_get_data"));
        p_llama_state_seq_set_data = reinterpret_cast<decltype(p_llama_state_seq_set_data)>(dlsym(RTLD_NEXT, "llama_state_seq_set_data"));
        p_llama_model_desc         = reinterpret_cast<decltype(p_llama_model_desc)>(dlsym(RTLD_NEXT, "llama_model_desc"));
        p_llama_model_n_params     = reinterpret_cast<decltype(p_llama_model_n_params)>(dlsym(RTLD_NEXT, "llama_model_n_params"));
        p_llama_memory_seq_pos_max = reinterpret_cast<decltype(p_llama_memory_seq_pos_max)>(dlsym(RTLD_NEXT, "llama_memory_seq_pos_max"));
    });
}

//------------------------------------------------------------
//Func : qml_session_enabled
//opt-in switch: everything in this file is a transparent passthrough
//unless QML_SESSION_ID is set
//  in: none (reads the QML_SESSION_ID environment variable)
// out: true = hooks are active, false = pure passthrough to real symbols
//------------------------------------------------------------
static bool qml_session_enabled()
{
    const char * session_id = getenv("QML_SESSION_ID");
    return session_id != nullptr && session_id[0] != '\0';
}

//------------------------------------------------------------
//Func : qml_slot_get
//looks up or inserts the qml_session_slot for one (memory, seq_id) pair
//  in: mem — the context's llama_memory_t; seq_id — sequence slot within it
// out: pointer to the slot (never NULL; owned by g_slots, address stable);
//      side effects: may insert a fresh slot under g_slots_mutex
//------------------------------------------------------------
static qml_session_slot * qml_slot_get(llama_memory_t mem, llama_seq_id seq_id)
{
    std::lock_guard<std::mutex> outer(g_slots_mutex);

    qml_slot_key key{mem, seq_id};
    auto it = g_slots.find(key);
    if (it != g_slots.end())
    {
        return it->second.get();
    }

    auto slot = std::make_unique<qml_session_slot>();
    qml_session_slot * raw = slot.get();
    g_slots.emplace(key, std::move(slot));
    return raw;
}

//------------------------------------------------------------
//Func : qml_model_derive_hash
//coarse fingerprint (arch/size) -- enough to refuse restoring a KV blob
//captured against a different model
//  in: model — loaded llama model; out / out_size — destination string buffer
// out: none; writes "desc:n_params" into out (truncated to out_size)
//------------------------------------------------------------
static void qml_model_derive_hash(const struct llama_model * model, char * out, size_t out_size)
{
    char desc[48] = {0};
    p_llama_model_desc(model, desc, sizeof(desc));
    snprintf(out, out_size, "%s:%llu", desc, (unsigned long long) p_llama_model_n_params(model));
}

//------------------------------------------------------------
//Func : qml_slot_ready
//lazily registers (first call) and restores (deferred here rather than
//at llama_init_from_model time -- see file header comment) the session
//for a seq_id the very first time it is seen. must be called with
//slot->lock already held.
//  in: slot — the (mem, seq_id) slot, lock HELD by the caller
//  in: ctx — live context; seq_id — sequence about to be decoded
// out: none; side effects: registers the session row (qml_session_begin,
//      id "{QML_SESSION_ID}#{seq_id}") and applies any pending KV blob into
//      the engine exactly once per slot lifetime
//------------------------------------------------------------
static void qml_slot_ready(qml_session_slot * slot, llama_context * ctx, llama_seq_id seq_id)
{
    if (slot->sess == nullptr)
    {
        const char * user_id     = getenv("QML_USER_ID");
        const char * session_pfx = getenv("QML_SESSION_ID");
        if (!user_id) user_id = "default_user";

        // one seq_id == one concurrent slot/conversation; suffixing lets
        // every slot in a multi-slot server get its own persisted row
        char session_id[300];
        snprintf(session_id, sizeof(session_id), "%s#%d", session_pfx, (int) seq_id);

        char model_hash[65] = {0}; // matches qml_llama_session's model_hash[65] (core/qml_session.c)
        qml_model_derive_hash(p_llama_get_model(ctx), model_hash, sizeof(model_hash));

        const void * state_data = nullptr;
        size_t       state_size = 0;
        int32_t      last_pos   = -1;

        slot->sess = qml_session_begin(user_id, session_id, seq_id, model_hash, &state_data, &state_size, &last_pos);
        slot->pending_state_data = state_data;
        slot->pending_state_size = state_size;
        slot->restore_applied    = false;
    }

    if (!slot->restore_applied)
    {
        if (slot->pending_state_data != nullptr && slot->pending_state_size > 0)
        {
            size_t applied = p_llama_state_seq_set_data(
                ctx, (const uint8_t *) slot->pending_state_data, slot->pending_state_size, seq_id);

            if (applied == 0)
            {
                fprintf(stderr, "[qml] warning: failed to restore KV state for seq_id=%d before first decode\n", (int) seq_id);
            }
            else
            {
                fprintf(stderr, "[qml] restored %zu bytes into llama KV memory for seq_id=%d before first decode\n",
                        applied, (int) seq_id);
            }

            if (p_llama_memory_seq_pos_max != nullptr)
            {
                llama_memory_t dbg_mem = p_llama_get_memory(ctx);
                fprintf(stderr, "[qml-debug] post-restore seq_pos_max(seq_id=%d) = %d\n",
                        (int) seq_id, (int) p_llama_memory_seq_pos_max(dbg_mem, seq_id));
            }
        }

        qml_session_restore_done(slot->sess);
        slot->restore_applied    = true;
        slot->pending_state_data = nullptr;
    }
}

extern "C" {

//------------------------------------------------------------
//Func : llama_init_from_model (interceptor)
//only lazily initializes storage (sqlite/lmdb + dlsym'd helper
//pointers) on first use. no session is created here -- that is
//deferred, per-seq_id, to the first llama_decode() touching that
//seq_id (see qml_slot_ready).
//  in: model / params — forwarded untouched to the real llama.cpp symbol
// out: the real function's context pointer (NULL passes through unchanged);
//      side effects: first enabled call opens the sqlite/lmdb stores
//------------------------------------------------------------
struct llama_context * llama_init_from_model(struct llama_model * model, struct llama_context_params params)
{
    static struct llama_context * (*real_init_from_model)(struct llama_model *, struct llama_context_params) = nullptr;

    if (!real_init_from_model)
    {
        real_init_from_model = reinterpret_cast<decltype(real_init_from_model)>(
            dlsym(RTLD_NEXT, "llama_init_from_model"));
    }

    struct llama_context * ctx = real_init_from_model(model, params);

    if (ctx != nullptr && qml_session_enabled())
    {
        qml_ensure_init();
    }

    return ctx;
}

//------------------------------------------------------------
//Func : llama_decode (interceptor)
//a server batch can mix tokens for many concurrent seq_id slots in one
//call. partitions the batch by seq_id, and for each: lazily registers +
//restores that slot (first time only), records the sub-batch into its
//session's token log, and rewrites its batch.pos[] entries in place,
//shifted by that slot's own restore offset.
//  in: ctx — live context; batch — decode batch (token/pos arrays read;
//      batch.pos[] rewritten IN PLACE via qml_session_track_decode)
// out: the real llama_decode's return code (0 = success)
//      side effects: lazy per-slot session registration + KV restore, and
//      token-log append into session storage for every sub-batch
//------------------------------------------------------------
int32_t llama_decode(struct llama_context * ctx, struct llama_batch batch)
{
    static int32_t (*real_decode)(struct llama_context *, struct llama_batch) = nullptr;

    if (!real_decode)
    {
        real_decode = reinterpret_cast<decltype(real_decode)>(dlsym(RTLD_NEXT, "llama_decode"));
    }

    llama_memory_t dbg_mem = (p_llama_get_memory != nullptr) ? p_llama_get_memory(ctx) : nullptr;
    fprintf(stderr, "[qml-debug] llama_decode called: ctx=%p mem=%p n_tokens=%d pos=%p first_pos=%d last_pos=%d\n",
            (void*)ctx, (void*)dbg_mem, (int) batch.n_tokens, (void*)batch.pos,
            (batch.pos != nullptr && batch.n_tokens > 0) ? batch.pos[0] : -999,
            (batch.pos != nullptr && batch.n_tokens > 0) ? batch.pos[batch.n_tokens-1] : -999);

    if (qml_session_enabled() && p_llama_get_memory != nullptr &&
        batch.token != nullptr && batch.pos != nullptr && batch.n_tokens > 0)
    {
        llama_memory_t mem = p_llama_get_memory(ctx);

        std::unordered_map<llama_seq_id, std::vector<int32_t>> idxs_by_seq;
        for (int32_t i = 0; i < batch.n_tokens; ++i)
        {
            llama_seq_id sid = (batch.seq_id != nullptr && batch.n_seq_id[i] > 0) ? batch.seq_id[i][0] : 0;
            idxs_by_seq[sid].push_back(i);
        }

        for (auto & kv : idxs_by_seq)
        {
            const llama_seq_id           seq_id = kv.first;
            const std::vector<int32_t> & idxs   = kv.second;

            qml_session_slot * slot = qml_slot_get(mem, seq_id);
            std::unique_lock<std::mutex> lk(slot->lock);

            qml_slot_ready(slot, ctx, seq_id);

            std::vector<int32_t> token_ids(idxs.size());
            std::vector<int32_t> pos(idxs.size());
            std::vector<int32_t> seq_ids(idxs.size(), seq_id);

            for (size_t j = 0; j < idxs.size(); ++j)
            {
                token_ids[j] = batch.token[idxs[j]];
                pos[j]       = batch.pos[idxs[j]];
            }

            qml_session_track_decode(slot->sess, token_ids.data(), pos.data(), seq_ids.data(), pos.size());

            for (size_t j = 0; j < idxs.size(); ++j)
            {
                batch.pos[idxs[j]] = pos[j];
            }

            if (p_llama_memory_seq_pos_max != nullptr)
            {
                fprintf(stderr, "[qml-debug] pre-real_decode seq_pos_max(seq_id=%d) = %d, batch.pos[first_idx]=%d\n",
                        (int) seq_id, (int) p_llama_memory_seq_pos_max(mem, seq_id), batch.pos[idxs[0]]);
            }
        }
    }

    return real_decode(ctx, batch);
}

//------------------------------------------------------------
//Func : llama_memory_seq_rm (interceptor)
//pure passthrough for now. A full-range wipe ([0, inf)) cannot be
//reliably read as "the host is reclaiming this slot for a different
//conversation" -- llama.cpp/llama-cli issue the exact same call shape
//as part of their OWN normal internal cleanup even in a single-session
//run (confirmed empirically: a p0=-1,p1=-1 seq_rm fires right before a
//completely ordinary llama_free, with no slot handoff involved at all;
//see docs/udf-control-plane-proposal.md). Treating every full wipe as
//an eviction caused real data loss (early persist + slot reset, so the
//real final state at llama_free never got saved). Persisting on
//eviction needs an external signal distinguishing "handing this slot to
//someone else" from "routine reset, llama_free will still follow" --
//that is the deferred control-plane mechanism, not something derivable
//from this call alone.
//  in: mem / seq_id / p0 / p1 — forwarded untouched to the real symbol
// out: the real function's result; no qml side effects (see rationale above)
//------------------------------------------------------------
bool llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1)
{
    static bool (*real_seq_rm)(llama_memory_t, llama_seq_id, llama_pos, llama_pos) = nullptr;

    if (!real_seq_rm)
    {
        real_seq_rm = reinterpret_cast<decltype(real_seq_rm)>(dlsym(RTLD_NEXT, "llama_memory_seq_rm"));
    }

    return real_seq_rm(mem, seq_id, p0, p1);
}

//------------------------------------------------------------
//Func : llama_free (interceptor)
//persists every slot still registered against this context's memory
//(a multi-slot server may have many), then forgets them.
//  in: ctx — context being destroyed (forwarded to the real llama_free)
// out: none; side effects: captures each registered slot's KV state, persists
//      it via qml_session_end, and erases those slots from g_slots
//------------------------------------------------------------
void llama_free(struct llama_context * ctx)
{
    static void (*real_free)(struct llama_context *) = nullptr;

    if (!real_free)
    {
        real_free = reinterpret_cast<decltype(real_free)>(dlsym(RTLD_NEXT, "llama_free"));
    }

    if (ctx != nullptr && qml_session_enabled() && p_llama_get_memory != nullptr)
    {
        llama_memory_t mem = p_llama_get_memory(ctx);

        std::vector<qml_slot_key> keys;
        {
            std::lock_guard<std::mutex> lock(g_slots_mutex);
            for (auto & kv : g_slots)
            {
                if (kv.first.mem == mem)
                {
                    keys.push_back(kv.first);
                }
            }
        }

        for (const auto & key : keys)
        {
            qml_session_slot * slot = qml_slot_get(key.mem, key.seq_id);
            std::unique_lock<std::mutex> lk(slot->lock);

            if (slot->sess != nullptr)
            {
                size_t sz = p_llama_state_seq_get_size(ctx, key.seq_id);
                std::vector<uint8_t> buf(sz);
                if (sz > 0)
                {
                    sz = p_llama_state_seq_get_data(ctx, buf.data(), buf.size(), key.seq_id);
                }

                qml_session_end(slot->sess, (sz > 0) ? buf.data() : nullptr, sz);
                slot->sess = nullptr;
            }
        }

        {
            std::lock_guard<std::mutex> lock(g_slots_mutex);
            for (const auto & key : keys)
            {
                g_slots.erase(key);
            }
        }
    }

    real_free(ctx);
}

} // extern "C"
