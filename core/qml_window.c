// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
//
// qml_window.c — context-window lifetime policy (TAWT) in the core.
//
// TAWT (Turn-Aligned Window Trim): when a session's token log outgrows the
// restore window, keep [system prefix + the most recent window of turns],
// move the excised middle LOSSLESSLY to the archive table, and renumber the
// live log. The cut point snaps to a turn boundary (a mid-turn cut leaves an
// unclosed turn — measured to collapse generation into repetition loops), and
// the system prefix is exempt from positional trimming (instruction and
// chat-template integrity).
//
// History: this policy carried every published TAWT measurement from an
// earlier Python gateway over the same store. Moved here (2026-08-23) so an
// engine-only deployment gets TAWT
// without any gateway: the paper claims TAWT as a session-layer mechanism,
// and the session layer is this core, not a convenience web process.
// The numbers are ports, not new tuning: head <= 200 tokens, retrim head-room
// target = max(1000, window * 0.7).

#include <stdlib.h>
#include <string.h>

#include "qml_router.h"
#include "qml_sqlite_store.h"
#include "qml_lmdb_store.h"

//------------------------------------------------------------
//Func : qml_window_trim
//applies TAWT to one session if its token log exceeds `window` tokens.
//Safe call sites are the ones with no live KV for the session (the restore /
//admission path, before any slot state exists): the stale blob is dropped
//here, and the caller then rebuilds the trimmed window from the token log.
//  in: session_id — session to consider
//  in: window — restore-window token budget (0 disables; returns 0)
//  in: eot_token — end-of-turn token id of the serving model (turn boundary)
// out: number of tokens archived (0 = under budget / no boundary fits /
//      concurrent change made it a no-op), -1 on storage error
// side effects: archive insert + log rewrite + session-counter update in one
//      transaction (qml_sqlite_trim_apply); drops the session's KV blob and
//      ext row (stale after the rewrite); drops the session's FTS index rows
//      so the next recall re-indexes the grown archive
//------------------------------------------------------------
long qml_window_trim(const char * session_id, size_t window,
                           int32_t eot_token)
{
    if (session_id == NULL || window == 0)
    {
        return 0;
    }
    qml_token_rec_t * rows = NULL;
    size_t            n    = 0;
    if (qml_sqlite_get_tokens(session_id, &rows, &n) != 0)
    {
        return -1;
    }
    if (n <= window)
    {
        free(rows);
        return 0;
    }
    // head = the first turn (BOS + system, through the first eot). Longer
    // than 200 tokens means it is not a system turn -- no exemption.
    // [MEASUREMENT SWITCH 2026-08-31] QML_TRIM_HEAD=0 disables the head exemption.
    // It exists to compare which of head (system prefix) and tail (recent turns)
    // sustains accuracy; the default remains as before, exemption applied (=1).
    const char * head_env = getenv("QML_TRIM_HEAD");
    const int    head_on  = (head_env == NULL) || (atoi(head_env) != 0);
    // [TRIM-HEAD-STICKY 2026-09-01] The exemption decision is made once per session. Previously
    // it was re-decided on every trim, so an ordinary turn (<=200 tokens) that moved to the front
    // after renumbering was mistaken for system and stuck in permanent exemption (measured: the cause
    // of 9-15 rules disappearing from the index — TEST-LEDGER row 60). The first trim's decision is
    // stored and reused afterwards. The exempt range always stays at the front of the log, so the
    // stored value remains valid regardless of renumbering.
    size_t head_len = 0;
    {
        const long stored = qml_sqlite_trim_head_get(session_id);
        if (stored >= 0)
        {
            head_len = (size_t) stored;
            if (head_len >= n) { head_len = 0; }
        }
        else
        {
            for (size_t i = 0; head_on && i < n && i <= 200; i++)
            {
                if (rows[i].token_id == eot_token)
                {
                    head_len = i + 1;
                    break;
                }
            }
            qml_sqlite_trim_head_set(session_id, (long) head_len);
        }
    }
    // keep-target below the window so the next few turns do not re-trigger
    size_t target = (window * 7) / 10;
    if (target < 1000)
    {
        target = 1000;
    }
    // cut = earliest turn boundary whose kept tail fits the target
    size_t lo = head_len;
    if (target > head_len && n > (target - head_len) && n - (target - head_len) > lo)
    {
        lo = n - (target - head_len);
    }
    // cut only where a user turn starts, so a question and its answer stay on
    // the same side (qml_window_cut)
    size_t cut = 0;
    int32_t * ids = (int32_t *) malloc(n * sizeof(int32_t));
    if (ids != NULL)
    {
        for (size_t j = 0; j < n; j++)
        {
            ids[j] = rows[j].token_id;
        }
        cut = qml_window_cut(ids, n, lo, eot_token);
        free(ids);
    }
    free(rows);
    if (cut == 0 || cut <= head_len || cut >= n)
    {
        return 0;   // no turn boundary fits -- leave the log alone
    }
    const long archived = qml_sqlite_trim_apply(session_id, head_len, cut,
                                                eot_token);
    if (archived <= 0)
    {
        return archived;   // no-op (0) or storage error (-1)
    }
    // the saved KV blob predates the rewrite -- drop it so no restore path
    // can apply pre-trim state over the post-trim log
    {
        qml_llama_session_meta_t meta;
        if (qml_sqlite_get_session(session_id, &meta) == 0 &&
            meta.lmdb_key[0] != '\0')
        {
            qml_lmdb_delete_blob(meta.lmdb_key);
            qml_sqlite_delete_llama_ext(session_id);
        }
    }
    // KV blocks stored through the vLLM connector or the SGLang backend hold
    // the old positions (K after RoPE) and cannot match again: drop them now
    (void) qml_store_drop_kv(session_id);
    // the FTS index (if built) predates the new archive rows; the lazy
    // indexer runs once per session, so force a rebuild on next recall
    (void) qml_sqlite_fts_drop(session_id);
    return archived;
}
