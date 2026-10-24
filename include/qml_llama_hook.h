// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
//
// qml_llama_hook.h
//
// llama.cpp-aware hook module for llama-server integration (layer 2 of 3):
//   1. core/qml_session.c + qml_router.h  -- pure engine-agnostic storage
//   2. THIS FILE                          -- llama.cpp API calls + the
//      persister thread, still isolated from llama-server's own C++
//      types (server_slot, server_tokens, etc.)
//   3. tools/server/server-context.cpp (llama.cpp source) -- thin call
//      sites into this file's functions only
//
// Unlike hooks/llama.cpp/qml_llama_preload.cpp (LD_PRELOAD, dlsym-resolved,
// zero link-time dependency -- needed there because it must run against
// whichever libllama.so happens to already be loaded), this file is
// compiled directly INTO llama-server's own build (see CMakeLists.txt's
// qml_llama_hook target, only added when built as part of the
// llama.cpp tree) and links normally against the `llama` target -- no
// ABI portability concern here, since it is always built against the
// exact same llama.cpp checkout it will run in.

#ifndef QML_LLAMA_HOOK_H
#define QML_LLAMA_HOOK_H

#include "llama.h"
#include "qml_router.h" // qml_control_cmd_t

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------
//Func : qml_llama_init
//opens the sqlite/lmdb backing stores and starts the internal
//persister thread. returns 0 on success, -1 on failure.
//  in: sqlite_path — session-metadata SQLite db path
//  in: lmdb_path — KV-blob LMDB store path
//------------------------------------------------------------
int qml_llama_init(const char * sqlite_path, const char * lmdb_path);

//------------------------------------------------------------
//Func : qml_llama_shutdown
//stops the persister thread (draining anything already queued) and
//closes the backing stores.
//  in: (none)
// out: none (void)
//------------------------------------------------------------
void qml_llama_shutdown(void);

//------------------------------------------------------------
//Func : qml_llama_capture
//Call right before BACKUP_NOW/MIGRATE: clears this session's capture-saving state so that the next
//enqueue_save always captures the full blob.
//  in: session_id — session whose capture-throttle state to clear
// out: none (void)
//------------------------------------------------------------
void qml_llama_capture(const char * session_id);

//------------------------------------------------------------
//Func : qml_llama_restore_known
//restore for a caller that already holds the token array (skips the second
//token-log SELECT -- 7,953 rows twice per restore at 8K tokens). Pass the
//number of tokens held; a mismatch against the stored metadata returns 0 so the
//caller can fall back to qml_llama_restore().
//returns 1 restored, 0 nothing usable, -1 state existed but restoring failed.
//  in: ctx_tgt — live llama_context to restore into
//  in: slot_id — server slot == destination sequence id
//  in: session_id — stored session key
//  in: model_hash — fingerprint to validate ("" = skip)
//  in: expect_n_tokens — token count the caller already holds
//------------------------------------------------------------
int qml_llama_restore_known(
    struct llama_context * ctx_tgt,
    int32_t                slot_id,
    const char *           session_id,
    const char *           model_hash,
    size_t                 expect_n_tokens
);

//------------------------------------------------------------
//Func : qml_llama_sync_geometry
//Tells the storage layer the live context's slot geometry (n_stream /
//n_ctx_seq) so a KV blob written under a different geometry is rejected
//before it is read. Call once after the context is created.
//  in: ctx — live llama_context whose n_stream / n_ctx_seq to record
// out: none (void)
//------------------------------------------------------------
void qml_llama_sync_geometry(struct llama_context * ctx);

//------------------------------------------------------------
//async restore, split in two: the blob fetch (disk + CPU, the slow part)
//runs on a background thread so the server's model loop keeps serving other
//slots; only set_data stays on the loop. Sequence:
//  launch:      qml_llama_fetch_begin(session, hash)
//  each loop:   rc = qml_llama_fetch_take(session, n_expect, &p, &n)
//               rc==1 -> still fetching: skip this slot, do NOT block
//               rc==0 -> qml_llama_state_apply(ctx, slot, p, n); free(p)
//               rc==2 -> fall back to the synchronous restore
//------------------------------------------------------------

// starts the background blob fetch (idempotent: a fetch already pending or
// ready for this session is left alone)
//  in: session_id — stored session key
//  in: model_hash — fingerprint to validate ("" = skip)
// out: none (void)
void qml_llama_fetch_begin(const char * session_id, const char * model_hash);

// non-consuming check for an in-flight fetch
//  in: session_id — session key previously passed to fetch_begin
// out: 1 fetch still running (defer this slot, do NOT block), 0 no fetch in
//      flight (ready, failed, or never started -- fetch_take disambiguates)
int qml_llama_fetch_pending(const char * session_id);

// consumes a finished fetch result
//  in: session_id — session key previously passed to fetch_begin
//  in: expect_n_tokens — token count the caller holds; a fetched blob whose
//      metadata claims MORE tokens is rejected (different lineage), FEWER is
//      accepted (capture-throttled prefix, tail recomputes)
//  in: out_state / out_size — (output) malloc-owned blob; caller free()s it
//      after state_apply
// out: 0 ready (outputs filled, ownership moves to the caller), 1 still
//      fetching (retry next loop tick), 2 nothing usable (fall back to the
//      synchronous restore)
int qml_llama_fetch_take(
    const char * session_id,
    size_t       expect_n_tokens,
    void **      out_state,
    size_t *     out_size
);

// loop-thread half of the async restore: llama_state_seq_set_data only.
// Must run on the thread that owns ctx_tgt (concurrent set_data vs decode
// is unsafe).
//  in: ctx_tgt — live llama_context
//  in: slot_id — server slot == destination sequence id
//  in: state_data / state_size — blob obtained from fetch_take
// out: 1 applied, -1 set_data failed
int qml_llama_state_apply(
    struct llama_context * ctx_tgt,
    int32_t                slot_id,
    const void *           state_data,
    size_t                 state_size
);

//------------------------------------------------------------
//Func : qml_llama_peek
//SQLite-only lookup of a stored session's token log -- does NOT touch the
//KV cache and does NOT load the LMDB blob's tensor bytes, so it is cheap
//enough to call while a slot is being launched. Used to prepend a stored
//session's history to an incoming request whose slot is still empty (the
//"lazy restore" path): without it, get_common_prefix() later compares the
//bare request against the just-restored token array, matches almost
//nothing, and the restored context is silently discarded.
//model_hash is validated exactly as qml_session_load does (pass
//"" to skip). Returns 1 and fills *out_tokens (malloc'd -- release with
//qml_llama_restore_done) / *out_n_tokens when a consistent stored
//session exists, 0 when there is nothing usable, -1 on error.
//  in: session_id — stored session key
//  in: model_hash — fingerprint to validate ("" = skip)
//  in: out_tokens / out_n_tokens — (output) malloc'd token array + count
//------------------------------------------------------------
int qml_llama_peek(
    const char *   session_id,
    const char *   model_hash,
    llama_token ** out_tokens,
    size_t *       out_n_tokens
);

//------------------------------------------------------------
//Func : qml_llama_restore
//looks up session_id in storage; if found (and model_hash matches,
//pass "" to skip the check), restores the KV state into ctx_tgt for
//seq_id == slot_id via llama_state_seq_set_data(), and returns the
//token id array that was resident in it.
//returns 1 if restored (*out_tokens/*out_n_tokens filled -- caller
//must eventually call qml_llama_restore_done), 0 if nothing to
//restore (out params are NULL/0, nothing to release), -1 if a prior
//state was found but restoring it failed (out params are NULL/0).
//  in: ctx_tgt — live llama_context to restore into
//  in: slot_id — server slot == destination sequence id
//  in: session_id — stored session key
//  in: model_hash — fingerprint to validate ("" = skip)
//  in: out_tokens / out_n_tokens — (output) malloc'd token array + count
//------------------------------------------------------------
int qml_llama_restore(
    struct llama_context * ctx_tgt,
    int32_t                slot_id,
    const char *           session_id,
    const char *           model_hash,
    llama_token **         out_tokens,
    size_t *               out_n_tokens
);

//------------------------------------------------------------
//Func : qml_llama_restore_done
//frees the token array returned by qml_llama_restore.
//safe to call with tokens == NULL.
//  in: tokens — array from try_restore/peek_tokens, or NULL
// out: none (void)
//------------------------------------------------------------
void qml_llama_restore_done(llama_token * tokens);

//------------------------------------------------------------
//Func : qml_llama_recent / _recent_sessions_free
//lists up to `limit` llama-engine session_ids, most recently
//updated first -- what --qml-restore-on-start auto restores.
//returns 0 on success (out array released with the _free call),
//-1 on failure.
//  in: limit — maximum number of session ids to return
//  in: out_ids / out_n — (output) malloc'd id array + count
//  in: ids / n — (_free) the array + count returned by recent_sessions
// out: _free — none (void)
//------------------------------------------------------------
int  qml_llama_recent(size_t limit, char *** out_ids, size_t * out_n);
void qml_llama_recent_free(char ** ids, size_t n);

//------------------------------------------------------------
//Func : qml_llama_save
//captures the current KV state for seq_id == slot_id (via
//llama_state_seq_get_size/get_data -- fast, in-process memcpy, safe to
//call from the model thread) together with the given token array, and
//hands both off to the background persister thread, which performs the
//actual SQLite/LMDB write off the model thread. Returns immediately;
//does not block on disk I/O. user_id is stored alongside the session for
//qml_session_save's per-user querying; pass "" if unknown.
//  in: ctx_tgt — live llama_context to capture from
//  in: slot_id — server slot == sequence id captured
//  in: session_id — session key to persist under
//  in: user_id — owner identity ("" if unknown)
//  in: model_hash — fingerprint stored with the session
//  in: tokens / n_tokens — full token array resident in the slot
// out: none (void; returns immediately, the write happens on the persister
//      thread)
//------------------------------------------------------------
void qml_llama_save(
    struct llama_context * ctx_tgt,
    int32_t                slot_id,
    const char *           session_id,
    const char *           user_id,
    const char *           model_hash,
    const llama_token *    tokens,
    size_t                 n_tokens
);

//------------------------------------------------------------
// QML Control Plane: live-KV command handoff
// (see docs/udf-control-plane-proposal.md)
//
// RESTORE/MIGRATE/BACKUP_NOW/PREWARM all touch live KV state, so unlike
// MONITOR_PING/COMPACT_NOW (which the persister thread executes directly
// -- see qml_llama_hook.cpp's dispatch_control_cmd), these four
// cannot run on the persister thread: they would race llama-server's own
// model thread. Instead, the persister thread only CLAIMS these commands
// and hands them to server-context.cpp via the queue below; the actual
// llama.cpp API calls happen on the model thread itself, at the same
// safe point task #17 already established (update_slots(), before the
// all-idle early return).
//------------------------------------------------------------

//------------------------------------------------------------
//Func : qml_llama_set_wake
//registers a callback the persister thread invokes (from ITS OWN thread,
//not the model thread) whenever it queues a new live-KV control op --
//queue_tasks only re-runs update_slots() in response to a posted task
//(see server-queue.cpp's start_loop: a 1s wait_for timeout alone does
//NOT trigger another update_slots() call), so without this wake-up a
//queued op could sit unexecuted until the next unrelated HTTP request.
//Expected implementation: post a no-op SERVER_TASK_TYPE_NEXT_RESPONSE
//task (the same "force one more update_slots() tick" idiom already used
//when !all_idle -- see update_slots()'s own self-post). Call once at
//server startup, right after qml_llama_init().
//  in: cb — callback invoked from the persister thread on each new live-KV op
//  in: userdata — opaque pointer passed back to cb unchanged
// out: none (void)
//------------------------------------------------------------
typedef void (*qml_wake_cb_t)(void * userdata);

void qml_llama_set_wake(qml_wake_cb_t cb, void * userdata);

//------------------------------------------------------------
//Func : qml_llama_control_poll
//pops one queued live-KV control op (RESTORE/MIGRATE/BACKUP_NOW/PREWARM),
//if any. Call from the model thread only (update_slots(), before the
//all-idle early return) -- never from the persister thread.
//returns 0 if a command was popped (*out_cmd filled), 1 if the queue is
//empty.
//  in: out_cmd — (output) popped command row
//------------------------------------------------------------
int qml_llama_control_poll(qml_control_cmd_t * out_cmd);

//------------------------------------------------------------
//Func : qml_llama_control_done
//reports the final DONE/FAILED status for a command popped via
//qml_llama_control_poll. Thin wrapper over
//qml_control_set_status (qml_router.h) -- kept here so
//server-context.cpp only ever calls into this header, not qml_router.h
//directly (same layering rule as the rest of this file).
//  in: cmd_id — cmd_id of the command popped via poll_control_op
//  in: status — "DONE" or "FAILED"
// out: none (void)
//------------------------------------------------------------
void qml_llama_control_done(int64_t cmd_id, const char * status);

//===========================================================================//
// [FTS-RECALL] BEGIN — automatic re-injection of archived turns.
// Additive-only feature block; removal instructions in qml_sqlite_store.h.
// Runtime default OFF: set QML_ARCHIVE_RECALL=1 to enable,
// QML_ARCHIVE_RECALL_BUDGET (tokens, default 1024) to cap the injection.
//===========================================================================//

//Func : qml_llama_recall_on
// in: none (reads env QML_ARCHIVE_RECALL once, cached)
//out: 1 when the feature is enabled, 0 otherwise
int qml_llama_recall_on(void);

//Func : qml_llama_recall
// in: session_id — session address (search is scoped to this session only)
// in: query_text — plain text of the NEW user turn (engine detokenizes it)
// in: eot_token — the vocab's end-of-turn token id (turn-boundary marker)
// in: detok — callback that detokenizes tokens to a malloc'd utf8 string
//     (special tokens skipped); ud is passed through to it
//out: number of recalled tokens written to *out_tokens (malloc'd, caller
//     frees) — complete archived turns matched by FTS, in original order,
//     capped by the budget env. 0 on: disabled, no archive, no FTS5, no
//     match, or any internal error (recall must never fail the request).
// in: n_ctx_avail — tokens of context headroom (n_ctx - history - new turn -
//     generation reserve); the dynamic budget is min(env budget, n_ctx_avail) and
//     any single turn larger than n_ctx_avail/2 is bypassed (RIG size rule)
size_t qml_llama_recall(const char * session_id,
                                       const char * query_text,
                                       int32_t eot_token,
                                       size_t n_ctx_avail,
                                       char * (*detok)(void * ud, const int32_t * toks, size_t n),
                                       void * ud,
                                       int32_t ** out_tokens);
//===========================================================================//
// [FTS-RECALL] END
//===========================================================================//

#ifdef __cplusplus
}
#endif

#endif // QML_LLAMA_HOOK_H
