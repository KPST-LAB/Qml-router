// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.

// hooks/llama.cpp/qml_llama_hook.cpp
//
// See qml_llama_hook.h for the layering rationale. This file owns
// every llama.cpp API call AND the persister thread, so that
// tools/server/server-context.cpp's own edits stay thin call sites.

#include "qml_llama_hook.h"
#include "qml_router.h"
#include "qml_lmdb_store.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <unordered_map>
#include <string>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h> // getpid()

// [2026-09-02] Set of sessions whose lazy vector backfill has finished (process-lifetime cache).
// Reason: the not-yet-embedded check query is a full FTS5 scan, so repeating it every turn is
// a bottleneck in itself. When trim indexes a new document, the session is removed from the set so the next recall
// backfills it again. Direct writes from outside the engine are not detected (see the comment at that point).
static std::mutex            g_vec_done_mtx;
static std::set<std::string> g_vec_done;

#ifdef __linux__
#include <fcntl.h>
#include <poll.h>
#include <sys/inotify.h>
#endif

namespace {

//------------------------------------------------------------
//control-cmd wait interval. This used to be the ONLY wake source, which
//made it the dominant term in observed control latency: the same RESTORE
//measured 17.2ms when the INSERT happened to land just before a poll and
//~350ms when it did not (2026-07-30). It is now only the safety-net
//timeout -- on Linux an inotify watch on the SQLite database and its -wal
//file wakes the loop as soon as another process writes a command, so the
//external contract (just INSERT a PENDING row) is unchanged (task #57).
//------------------------------------------------------------
constexpr auto kControlCmdPollInterval = std::chrono::milliseconds(200);

//------------------------------------------------------------
//inotify wake channel (Linux only; elsewhere the interval above is the
//sole wake source, exactly as before). Watching the -wal file matters
//more than the .db itself: in WAL mode a committed INSERT from another
//process appends to <db>-wal and may not touch the main file at all.
//------------------------------------------------------------
#ifdef __linux__
int         g_inotify_fd = -1;
std::string g_db_path;

// opens the inotify fd and arms watches on the DB file, its -wal file, and the
// containing directory so cross-process control-command INSERTs wake the loop
//  in: sqlite_path — path to the control-plane SQLite database (may be NULL)
// out: none; on any failure g_inotify_fd stays -1 and the persister loop
//      silently falls back to interval polling
void inotify_setup(const char * sqlite_path)
{
    g_db_path    = sqlite_path ? sqlite_path : "";
    g_inotify_fd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
    if (g_inotify_fd < 0)
    {
        return; // fall back to interval polling -- not fatal
    }

    // the -wal / -shm files are created lazily and recreated on checkpoint,
    // so watch the containing DIRECTORY as well; IN_MODIFY|IN_CREATE there
    // covers every path SQLite can take without us re-arming watches.
    inotify_add_watch(g_inotify_fd, g_db_path.c_str(), IN_MODIFY);

    const std::string wal = g_db_path + "-wal";
    inotify_add_watch(g_inotify_fd, wal.c_str(), IN_MODIFY);

    const size_t slash = g_db_path.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? std::string(".") : g_db_path.substr(0, slash);
    inotify_add_watch(g_inotify_fd, dir.c_str(), IN_MODIFY | IN_CREATE);
}

// closes the inotify fd opened by inotify_setup (watches go with the fd)
//  in: none
// out: none; safe to call when setup failed or never ran
void inotify_teardown()
{
    if (g_inotify_fd >= 0)
    {
        close(g_inotify_fd);
        g_inotify_fd = -1;
    }
}

//returns true if a filesystem event arrived within timeout (or immediately),
//false on timeout/error. Drains the queue so a burst collapses into one wake.
//  in: timeout — maximum time to block in poll() waiting for a write event
// out: true = filesystem event arrived, false = timeout, error, or fd not open
bool inotify_wait(std::chrono::milliseconds timeout)
{
    if (g_inotify_fd < 0)
    {
        return false;
    }

    struct pollfd pfd;
    pfd.fd      = g_inotify_fd;
    pfd.events  = POLLIN;
    pfd.revents = 0;

    const int rc = poll(&pfd, 1, (int) timeout.count());
    if (rc <= 0)
    {
        return false;
    }

    char buf[4096];
    while (read(g_inotify_fd, buf, sizeof(buf)) > 0)
    {
        // drain
    }
    return true;
}
#endif // __linux__

//------------------------------------------------------------
//live-KV control-op queue: filled by the persister thread (claim only,
//never executes these here), drained by the model thread via
//qml_llama_control_poll() at the update_slots() safe point.
//Separate mutex from g_mutex/g_cv below -- that pair is about the save
//queue's producer/consumer signaling, a different concern from this
//cross-thread handoff, and entangling them would make it harder to
//reason about which thread can block on what.
//------------------------------------------------------------
std::mutex                     g_ctrl_mutex;
std::deque<qml_control_cmd_t>  g_ctrl_ops;

std::mutex     g_wake_mutex;
qml_wake_cb_t  g_wake_cb = nullptr;
void *         g_wake_ud = nullptr;

//------------------------------------------------------------
//Func : is_live_kv_cmd_type
//tells whether a control command touches live KV state and therefore must run
//on the model thread (via g_ctrl_ops) instead of the persister thread
//  in: cmd_type — command name string from the claimed control-plane row
// out: true = route to the model thread, false = safe to dispatch here
//------------------------------------------------------------
bool is_live_kv_cmd_type(const char * cmd_type)
{
    return std::strcmp(cmd_type, "RESTORE")     == 0
        || std::strcmp(cmd_type, "MIGRATE")     == 0
        || std::strcmp(cmd_type, "BACKUP_NOW")  == 0
        || std::strcmp(cmd_type, "PREWARM")     == 0
        || std::strcmp(cmd_type, "PURGE_CACHE") == 0;
}

//------------------------------------------------------------
//Func : dispatch_control_cmd
//executes the cmd_type-specific action for an already-CLAIMED command
//and reports DONE/FAILED back. Only MONITOR_PING and COMPACT_NOW are
//handled here -- both are safe to run from this background thread
//because neither touches live KV state. RESTORE/MIGRATE/BACKUP_NOW/
//PREWARM are routed to g_ctrl_ops by persister_main BEFORE reaching this
//function (see is_live_kv_cmd_type's callers below) -- this function
//should never actually see one of those cmd_types; the fallback below
//is a genuine "unrecognized cmd_type" case, not a "not implemented yet"
//placeholder.
//  in: cmd — already-CLAIMED control command (cmd_id, cmd_type, session_id)
// out: none; side effects: writes DONE/FAILED to the command's SQLite row and,
//      per cmd_type, may prefetch page cache, compact storage, delete a
//      session's blob + ext row, or copy a session inside SQLite/LMDB
//------------------------------------------------------------
void dispatch_control_cmd(const qml_control_cmd_t & cmd)
{
    if (std::strcmp(cmd.cmd_type, "MONITOR_PING") == 0)
    {
        fprintf(stderr, "[qml][control] MONITOR_PING cmd_id=%lld session_id=%s handled by pid=%d\n",
                (long long) cmd.cmd_id, cmd.session_id, (int) getpid());
        qml_control_set_status(cmd.cmd_id, "DONE");
        return;
    }

    if (std::strcmp(cmd.cmd_type, "PREFETCH_RAM") == 0)
    {
        // deliberately handled HERE, on the persister thread, and NOT routed
        // through is_live_kv_cmd_type()/the model thread: this command never
        // touches live KV or a slot, so making the model thread wait for a
        // multi-hundred-MB page-cache walk would be pure latency for nothing.
        size_t    bytes = 0;
        const int rc    = qml_session_prefetch(cmd.session_id, &bytes);
        fprintf(stderr,
                "[qml][control] PREFETCH_RAM cmd_id=%lld session_id=%s pid=%d rc=%d bytes=%zu\n",
                (long long) cmd.cmd_id, cmd.session_id, (int) getpid(), rc, bytes);
        // rc == 1 (nothing stored) is reported FAILED on purpose: a caller
        // prefetching a session that has no saved state asked for something
        // that did not happen, and silently answering DONE would hide it.
        qml_control_set_status(cmd.cmd_id, rc == 0 ? "DONE" : "FAILED");
        return;
    }

    if (std::strcmp(cmd.cmd_type, "COMPACT_NOW") == 0)
    {
        const int rc = qml_store_compact();
        fprintf(stderr, "[qml][control] COMPACT_NOW cmd_id=%lld handled by pid=%d rc=%d\n",
                (long long) cmd.cmd_id, (int) getpid(), rc);
        qml_control_set_status(cmd.cmd_id, rc == 0 ? "DONE" : "FAILED");
        return;
    }

    if (std::strcmp(cmd.cmd_type, "DROP_BLOB") == 0)
    {
        // Cold storage: remove only the blob + ext row, keep the token log -- the session
        // is brought back on reconnect by replaying the token log (explicit form of the eviction = demotion principle).
        int rc = -1;
        qml_llama_session_meta_t meta;
        if (qml_sqlite_get_session(cmd.session_id, &meta) == 0 &&
            meta.lmdb_key[0] != '\0')
        {
            qml_lmdb_delete_blob(meta.lmdb_key);
            rc = qml_sqlite_delete_llama_ext(cmd.session_id);
        }
        else
        {
            rc = 1; // no blob -- already cold
        }
        fprintf(stderr, "[qml][control] DROP_BLOB cmd_id=%lld '%s' rc=%d\n",
                (long long) cmd.cmd_id, cmd.session_id, rc);
        // rc==1 (no blob -- already cold) is the requested end state, not a failure
        qml_control_set_status(cmd.cmd_id, rc >= 0 ? "DONE" : "FAILED");
        return;
    }

    if (std::strcmp(cmd.cmd_type, "SWEEP_BLOBS") == 0)
    {
        // [BLOB-SWEEP 2026-10-02] removes LMDB blobs no session refers to any
        // more. Session deletion used to remove only the SQLite rows, so every
        // deleted session left its blob behind (about 8.6 GB of 8.9 GB in use on
        // the H200 after a load-run cleanup). A blob is kept when a qml_session
        // row has its key as session id or a llama ext row has it as lmdb_key;
        // any lookup error also keeps it. Runs on this engine's own LMDB and
        // SQLite handles, in the control-command thread, after this engine's
        // pending saves. A brand-new session whose first blob is written but
        // whose rows are not committed yet could lose that blob; its token log
        // is kept and the session replays from it.
        char **    keys  = nullptr;
        uint64_t * sizes = nullptr;
        size_t     n     = 0;
        size_t     removed = 0, failed = 0;
        uint64_t   freed   = 0;
        int        rc      = qml_lmdb_list_blobs(&keys, &sizes, &n);
        if (rc == 0)
        {
            for (size_t i = 0; i < n; ++i)
            {
                if (qml_sqlite_uses_blob(keys[i]) != 0)
                {
                    continue;
                }
                if (qml_lmdb_delete_blob(keys[i]) == 0)
                {
                    ++removed;
                    freed += sizes[i];
                }
                else
                {
                    ++failed;
                }
            }
            qml_lmdb_free_list(keys, sizes, n);
        }
        fprintf(stderr, "[qml][control] SWEEP_BLOBS cmd_id=%lld scanned=%zu removed=%zu"
                " freed_bytes=%llu failed=%zu rc=%d\n",
                (long long) cmd.cmd_id, n, removed, (unsigned long long) freed, failed, rc);
        qml_control_set_status(cmd.cmd_id, (rc == 0 && failed == 0) ? "DONE" : "FAILED");
        return;
    }

    if (std::strcmp(cmd.cmd_type, "COPY_SESSION") == 0)
    {
        // session_id carries "src=>dst[=>owner]" -- the copy runs on the
        // ENGINE'S own SQLite connection (see qml_sqlite_copy_session's header
        // comment for why external writers are not equivalent on every host).
        // Field separator is '=>' for ALL fields: session ids legally contain
        // '@' (owui ids embed emails), so the old trailing-'@owner' parse
        // silently truncated the dst to everything before the email's '@'
        // (2026-08-11: copies landed in a phantom 'owui:big' session and all
        // retries then failed on the dst-exists precheck).
        const char * sep = std::strstr(cmd.session_id, "=>");
        int rc = -1;
        if (sep != nullptr && sep != cmd.session_id && sep[2] != '\0')
        {
            std::string src(cmd.session_id, sep - cmd.session_id);
            std::string dst(sep + 2);
            std::string owner; // "src=>dst=>user" -> owner override (sharing/handover)
            const size_t at = dst.find("=>");
            if (at != std::string::npos && at > 0 && at + 2 < dst.size())
            {
                owner = dst.substr(at + 2);
                dst.resize(at);
            }
            rc = qml_sqlite_copy_session(src.c_str(), dst.c_str(),
                                         owner.empty() ? NULL : owner.c_str());
        }
        fprintf(stderr, "[qml][control] COPY_SESSION cmd_id=%lld '%s' rc=%d\n",
                (long long) cmd.cmd_id, cmd.session_id, rc);
        qml_control_set_status(cmd.cmd_id, rc == 0 ? "DONE" : "FAILED");
        return;
    }

    fprintf(stderr,
            "[qml][control] cmd_id=%lld cmd_type=%s UNRECOGNIZED -- marking FAILED\n",
            (long long) cmd.cmd_id, cmd.cmd_type);
    qml_control_set_status(cmd.cmd_id, "FAILED");
}

//------------------------------------------------------------
//qml_save_job
//captured synchronously in qml_llama_save (fast,
//in-process memcpy only), consumed by persister_main on the background
//thread (the actual SQLite/LMDB write, off the model thread)
//------------------------------------------------------------
struct qml_save_job
{
    std::string              session_id;
    std::string              user_id;
    std::string              model_hash;
    int32_t                  seq_id;
    std::vector<uint8_t>     state_data;
    std::vector<llama_token> tokens;
    // capture throttle (2026-08-11): true = token log only, blob untouched.
    // The D2H state capture at release was profiled as the LARGEST per-turn
    // cost (~520ms @32K, comment in enqueue_save); when a turn adds fewer than
    // QML_CAPTURE_MIN_DELTA tokens since the last full capture, we skip it.
    // The stale blob still restores as a PREFIX (tail recomputed) -- see
    // qml_restore_covers in server-context.cpp.
    bool                     tokens_only = false;
};

std::mutex               g_mutex;
std::condition_variable  g_cv;
std::deque<qml_save_job>  g_queue;
std::thread               g_thread;
bool                      g_running = false;
bool                      g_stop    = false;

//the job the persister popped but has not committed yet. Restores must be able
//to see it: a queued/in-flight save is newer than anything in the store, and a
//restore that reads the store while a save for the same session is pending
//resurrects the PREVIOUS turn (observed as silent turn loss on hot sessions
//under slot churn -- max n_tokens 3,053 vs the 4,119 the client actually sent).
qml_save_job             g_inflight;
bool                     g_inflight_valid = false;

// capture throttle state: per-session token count at the last FULL capture.
// process-lifetime only -- after a restart the map is empty, so the first
// release captures fully (safe default).
std::mutex                              g_capture_mutex;
std::unordered_map<std::string, size_t> g_last_capture_n;
size_t                                  g_capture_min_delta = 512;

//------------------------------------------------------------
//Func : pending_save_lookup
//read-your-writes for the async persister: newest pending save wins over the
//store. Scans the queue back-to-front (newest first), then the in-flight job.
//Copies out under g_mutex; a ~90 MB copy (~15 ms) is cheap next to the
//full-history prefill that losing the turn would eventually force.
//  in: session_id — session to look up; model_hash — must match the saved job
//  in: out_state / out_tokens — filled with copies of the newest matching job
// out: true = a queued or in-flight save was copied out, false = none pending
//------------------------------------------------------------
bool pending_save_lookup(
    const char *               session_id,
    const char *               model_hash,
    std::vector<uint8_t> &     out_state,
    std::vector<llama_token> & out_tokens)
{
    std::lock_guard<std::mutex> lock(g_mutex);

    for (auto it = g_queue.rbegin(); it != g_queue.rend(); ++it)
    {
        if (it->session_id == session_id && it->model_hash == model_hash)
        {
            out_state  = it->state_data;
            out_tokens = it->tokens;
            return true;
        }
    }

    if (g_inflight_valid &&
        g_inflight.session_id == session_id && g_inflight.model_hash == model_hash)
    {
        out_state  = g_inflight.state_data;
        out_tokens = g_inflight.tokens;
        return true;
    }

    return false;
}

std::atomic<uint32_t> g_cur_n_stream{0};

// derives the live context's slot geometry (n_stream, per-sequence ctx size)
// from public llama.cpp API and publishes it (see the qml_sync_geometry doc
// block further down, above qml_llama_sync_geometry, for the rationale)
//  in: ctx — live llama_context; NULL is a no-op
// out: none; side effects: stores n_stream in g_cur_n_stream and hands the
//      geometry to the session layer's guard (qml_session_set_geometry)
static void qml_sync_geometry(struct llama_context * ctx)
{
    if (ctx == nullptr)
    {
        return;
    }

    const uint32_t n_ctx     = llama_n_ctx(ctx);
    const uint32_t n_ctx_seq = llama_n_ctx_seq(ctx);
    const uint32_t n_seq_max = llama_n_seq_max(ctx);
    const uint32_t n_stream  = (n_ctx_seq == n_ctx) ? 1u : n_seq_max;

    g_cur_n_stream.store(n_stream, std::memory_order_relaxed);
    qml_session_set_geometry(n_stream, n_ctx_seq);
}

//------------------------------------------------------------
//Func : reframe_state_blob
//Re-frames a per-sequence state blob written under one slot geometry so a
//context with a DIFFERENT one accepts it. Only the framing changes; not a
//single byte of KV data is touched or reinterpreted.
//
//Layout (llama-context.cpp state_seq_get_data + llama-kv-cache.cpp
//state_write):
//    [u32 magic][i32 seq_id][u32 n_stream]
//    then n_stream sections, each [u32 cell_count] + (data if non-zero)
//A per-sequence save fills exactly ONE section -- the stream that owns the
//sequence -- and writes a bare cell_count=0 for the others, so re-framing is
//"keep the one payload, rewrite the count, pad with empty sections". Placing
//the payload first is safe because state_read routes a per-seq section by
//seq_to_stream[seq_id], not by its position in the file.
//
//Returns a malloc'd copy, or nullptr when the blob does not have exactly
//that shape -- in which case the caller must fall back to recompute rather
//than guess. Deliberately conservative: a wrong transform would be caught by
//the engine's type/layer/capacity checks anyway (restore fails -> recompute),
//so this never risks silently wrong KV.
//  in: src / src_size — original per-sequence state blob bytes
//  in: target_n_stream — live context's stream count (only 1 is accepted)
//  in: out_size — receives the size of the re-framed copy on success
// out: malloc'd re-framed blob (caller frees), or nullptr = refuse and let the
//      caller recompute from the token log
//------------------------------------------------------------
static void * reframe_state_blob(const void * src, size_t src_size,
                                 uint32_t target_n_stream, size_t * out_size)
{
    constexpr size_t kPrefix = sizeof(uint32_t) + sizeof(int32_t); // magic + seq_id
    constexpr size_t kU32    = sizeof(uint32_t);

    if (src == nullptr || target_n_stream == 0 || src_size < kPrefix + kU32)
    {
        return nullptr;
    }

    // ONLY the "into a unified context" direction is enabled (2026-08-02).
    // N -> 1 was verified correct across repeated runs (3/3 sessions, real KV
    // restore, needle recall OK). The opposite direction (1 -> N) produced
    // intermittent wrong answers in the same harness; the harness also did not
    // wait for the async save to settle between phases, so the stored state
    // itself may have been inconsistent -- meaning the direction is UNPROVEN,
    // not proven broken. Until that is resolved, refuse it: the caller then
    // recomputes from the token log, which is correct and merely slower.
    // A wrong answer is never an acceptable trade for a faster restore.
    if (target_n_stream != 1)
    {
        return nullptr;
    }

    const uint8_t * p = (const uint8_t *) src;

    uint32_t src_n_stream = 0;
    memcpy(&src_n_stream, p + kPrefix, kU32);
    if (src_n_stream == 0 || src_n_stream == target_n_stream)
    {
        return nullptr; // nothing to do (caller uses the blob as-is)
    }

    size_t   cur        = kPrefix + kU32;
    size_t   pay_beg    = 0;
    size_t   pay_end    = 0;
    uint32_t n_nonempty = 0;

    for (uint32_t sidx = 0; sidx < src_n_stream; ++sidx)
    {
        if (cur + kU32 > src_size)
        {
            return nullptr;
        }

        uint32_t cell_count = 0;
        memcpy(&cell_count, p + cur, kU32);

        if (cell_count == 0)
        {
            cur += kU32;
            continue;
        }

        if (++n_nonempty > 1)
        {
            return nullptr; // not a single-sequence blob -- refuse
        }

        const size_t tail = (size_t) (src_n_stream - sidx - 1) * kU32;
        if (src_size < cur + kU32 + tail)
        {
            return nullptr;
        }

        pay_beg = cur;
        pay_end = src_size - tail;
        cur     = pay_end; // the loop's remaining turns walk the trailing zeros
    }

    if (n_nonempty != 1 || cur != src_size)
    {
        return nullptr;
    }

    const size_t pay_size = pay_end - pay_beg;
    const size_t out_sz   = kPrefix + kU32 + pay_size + (size_t) (target_n_stream - 1) * kU32;

    uint8_t * dst = (uint8_t *) malloc(out_sz);
    if (dst == nullptr)
    {
        return nullptr;
    }

    memcpy(dst, p, kPrefix);                                   // magic + seq_id
    memcpy(dst + kPrefix, &target_n_stream, kU32);             // new stream count
    memcpy(dst + kPrefix + kU32, p + pay_beg, pay_size);       // the one payload
    memset(dst + kPrefix + kU32 + pay_size, 0,
           (size_t) (target_n_stream - 1) * kU32);             // empty siblings

    *out_size = out_sz;
    return dst;
}

//------------------------------------------------------------
//async state fetch
//
//The mixed-load measurement (docs/engine-llama-cpp.md 7-7) showed WHY this
//exists: a synchronous restore blocks llama-server's single model loop for
//the whole disk read (~seconds at 32K tokens), stalling every other slot's
//decode -- short-request throughput came out 18% LOWER than the recompute
//configuration even though qml did 201x less prefill. The fix is to split
//the restore: the disk/CPU part (blob fetch) runs on this background thread,
//and only llama_state_seq_set_data (~0.17 s/GB, must touch llama_context on
//the loop thread) stays synchronous.
//------------------------------------------------------------
struct qml_fetch_entry
{
    void *  data   = nullptr; // malloc-owned state blob
    size_t  size   = 0;
    int64_t n_meta = 0;       // token count recorded by the save that wrote it
    int     status = 0;       // 0 = in progress, 1 = ready, 2 = failed/not found
};

std::mutex                                      g_fetch_mutex;
std::condition_variable                         g_fetch_cv;
std::deque<std::pair<std::string, std::string>> g_fetch_jobs;   // session_id, model_hash
std::map<std::string, qml_fetch_entry>          g_fetch_results;
std::thread                                     g_fetch_thread;
bool                                            g_fetch_stop = false;

// background fetch thread body: pops (session_id, model_hash) jobs, loads the
// state blob (newest pending save first, then the SQLite/LMDB store),
// re-frames it if the stored geometry differs, publishes the result into
// g_fetch_results, and nudges the server loop via the wake callback
//  in: none (consumes g_fetch_jobs under g_fetch_mutex)
// out: none; loops until g_fetch_stop; side effects: reads SQLite/LMDB,
//      mallocs blobs owned by g_fetch_results until fetch_take claims them
void fetch_main()
{
    for (;;)
    {
        std::pair<std::string, std::string> job;
        {
            std::unique_lock<std::mutex> lock(g_fetch_mutex);
            g_fetch_cv.wait(lock, [] { return g_fetch_stop || !g_fetch_jobs.empty(); });
            if (g_fetch_stop)
            {
                break;
            }
            job = std::move(g_fetch_jobs.front());
            g_fetch_jobs.pop_front();
        }

        void *  data   = nullptr;
        size_t  size   = 0;
        int64_t n_meta = 0;
        int     status = 2;

        const auto t_fetch0 = std::chrono::steady_clock::now();

        // read-your-writes: a queued/in-flight save is newer than the store
        {
            std::vector<uint8_t>     pstate;
            std::vector<llama_token> ptoks;
            if (pending_save_lookup(job.first.c_str(), job.second.c_str(), pstate, ptoks))
            {
                data = malloc(pstate.size());
                if (data != nullptr)
                {
                    memcpy(data, pstate.data(), pstate.size());
                    size   = pstate.size();
                    n_meta = (int64_t) ptoks.size();
                    status = 1;
                }
            }
        }

        if (status != 1)
        {
            uint32_t blob_nstream = 0;
            const int rc = qml_session_load_any(
                job.first.c_str(), job.second.c_str(), &data, &size, &n_meta, &blob_nstream);
            status = (rc == 0 && data != nullptr) ? 1 : 2;

            const uint32_t cur_nstream = g_cur_n_stream.load(std::memory_order_relaxed);
            if (status == 1 && blob_nstream != 0 && cur_nstream != 0 &&
                blob_nstream != cur_nstream)
            {
                size_t fixed_size = 0;
                void * fixed = reframe_state_blob(data, size, cur_nstream, &fixed_size);
                free(data);
                if (fixed == nullptr)
                {
                    data   = nullptr;
                    size   = 0;
                    status = 2; // caller recomputes from the token log
                }
                else
                {
                    fprintf(stderr,
                            "[qml] session '%s': re-framed blob n_stream %u -> %u (fetch)\n",
                            job.first.c_str(), blob_nstream, cur_nstream);
                    data = fixed;
                    size = fixed_size;
                }
            }
        }

        // one line per restore: where the CPU-side floor goes (fetch half)
        {
            const double fetch_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_fetch0).count();
            fprintf(stderr,
                    "[qml][timing] fetch session='%s' size=%zu status=%d ms=%.1f t=%.3f\n",
                    job.first.c_str(), size, status, fetch_ms,
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now().time_since_epoch()).count());
        }

        {
            std::lock_guard<std::mutex> lock(g_fetch_mutex);
            qml_fetch_entry & e = g_fetch_results[job.first];
            free(e.data); // paranoia: never leak a previous unclaimed blob
            e.data   = data;
            e.size   = size;
            e.n_meta = n_meta;
            e.status = status;
        }

        // nudge the server loop so a deferred slot retries its take() promptly
        qml_wake_cb_t wake_cb;
        void *        wake_ud;
        {
            std::lock_guard<std::mutex> wl(g_wake_mutex);
            wake_cb = g_wake_cb;
            wake_ud = g_wake_ud;
        }
        if (wake_cb != nullptr)
        {
            wake_cb(wake_ud);
        }
    }
}

//------------------------------------------------------------
//Func : persister_main
//alternates between draining the save queue (as before) and polling for
//control-plane commands targeting this process's own pid. The wait is
//now timed rather than purely condition-driven, since a pending control
//command produces no g_cv notification (it is written into SQLite by a
//DIFFERENT process entirely) -- polling on a fixed interval is how
//docs/udf-control-plane-proposal.md's blocker #1 (no cross-process push
//notification) gets resolved without inventing a new IPC channel.
//  in: none (consumes g_queue under g_mutex; claims control rows by own pid)
// out: none; loops until g_stop (drains the save queue first); side effects:
//      writes session snapshots to SQLite/LMDB, executes non-live-KV control
//      commands, routes live-KV ones to g_ctrl_ops, updates command status
//------------------------------------------------------------
void persister_main()
{
    for (;;)
    {
        std::unique_lock<std::mutex> lock(g_mutex);
#ifdef __linux__
        if (g_inotify_fd >= 0)
        {
            // Two wake sources have to be honoured here: g_cv (a save job was
            // just enqueued by the model thread, in-process) and inotify (a
            // control command was written by ANOTHER process). Waiting on both
            // at once would need an eventfd bridge; instead the condition wait
            // is kept short and the remaining budget is spent in poll(), which
            // is where control-command latency actually matters. Net effect:
            // a save job is picked up within kSaveSignalSlice, and a control
            // command within roughly the inotify wake (sub-ms) plus that
            // slice -- versus a flat 200ms average before.
            constexpr auto kSaveSignalSlice = std::chrono::milliseconds(5);
            g_cv.wait_for(lock, kSaveSignalSlice, [] { return g_stop || !g_queue.empty(); });
            if (!g_stop && g_queue.empty())
            {
                lock.unlock();
                inotify_wait(kControlCmdPollInterval - kSaveSignalSlice);
                lock.lock();
            }
        }
        else
        {
            g_cv.wait_for(lock, kControlCmdPollInterval, [] { return g_stop || !g_queue.empty(); });
        }
#else
        g_cv.wait_for(lock, kControlCmdPollInterval, [] { return g_stop || !g_queue.empty(); });
#endif

        const bool stopping = g_stop;

        while (!g_queue.empty())
        {
            // the job stays visible in g_inflight until its commit lands, so
            // pending_save_lookup() never has a window where a pending save is
            // in neither the queue nor the store. Only this thread writes
            // g_inflight; readers copy it out under g_mutex.
            g_inflight       = std::move(g_queue.front());
            g_inflight_valid = true;
            g_queue.pop_front();
            lock.unlock();

            int save_rc = 0;
            if (g_inflight.tokens_only)
            {
                save_rc = qml_session_save_tokens(
                    g_inflight.session_id.c_str(), g_inflight.user_id.c_str(),
                    g_inflight.model_hash.c_str(), g_inflight.seq_id,
                    g_inflight.tokens.data(), g_inflight.tokens.size());
            }
            else
            {
                save_rc = qml_session_save(
                    g_inflight.session_id.c_str(), g_inflight.user_id.c_str(),
                    g_inflight.model_hash.c_str(), g_inflight.seq_id,
                    g_inflight.state_data.data(), g_inflight.state_data.size(),
                    g_inflight.tokens.data(), g_inflight.tokens.size());
            }
            // [SAVE-LOG 2026-10-02] the return value used to be dropped, so a
            // failed save left no trace in the engine log
            if (save_rc != 0)
            {
                fprintf(stderr, "[qml][save] FAILED session=%s tokens=%zu mode=%s"
                        " -- this turn is not in the store\n",
                        g_inflight.session_id.c_str(), g_inflight.tokens.size(),
                        g_inflight.tokens_only ? "tokens-only" : "full");
                fflush(stderr);
            }

            lock.lock();
            g_inflight_valid = false;
            g_inflight       = qml_save_job();
        }

        lock.unlock();

        if (stopping)
        {
            break;
        }

        qml_control_cmd_t cmd;
        while (qml_control_claim((int32_t) getpid(), &cmd) == 0)
        {
            if (is_live_kv_cmd_type(cmd.cmd_type))
            {
                {
                    std::lock_guard<std::mutex> ctrl_lock(g_ctrl_mutex);
                    g_ctrl_ops.push_back(cmd);
                }

                qml_wake_cb_t wake_cb;
                void *        wake_ud;
                {
                    std::lock_guard<std::mutex> wl(g_wake_mutex);
                    wake_cb = g_wake_cb;
                    wake_ud = g_wake_ud;
                }
                if (wake_cb != nullptr)
                {
                    wake_cb(wake_ud);
                }
            }
            else
            {
                dispatch_control_cmd(cmd);
            }
        }
    }
}

} // namespace

extern "C" {

//------------------------------------------------------------
//Func : qml_llama_init
//opens session storage, arms the inotify wake channel, reads the capture
//throttle from QML_CAPTURE_MIN_DELTA, and starts the persister + fetch threads
//  in: sqlite_path — metadata/control-plane DB path; lmdb_path — blob store
// out: 0 = ready, -1 = storage init failed (no threads started)
//      side effects: opens SQLite/LMDB, spawns two background threads
//------------------------------------------------------------
int qml_llama_init(const char * sqlite_path, const char * lmdb_path)
{
    if (qml_store_init(sqlite_path, lmdb_path) != 0)
    {
        return -1;
    }

#ifdef __linux__
    inotify_setup(sqlite_path);
#endif

    if (const char * e = getenv("QML_CAPTURE_MIN_DELTA"))
    {
        const long v = atol(e);
        if (v >= 0)
        {
            g_capture_min_delta = (size_t) v;
        }
    }
    fprintf(stderr, "[qml] capture throttle: full capture only when >= %zu new tokens "
            "since the last blob (QML_CAPTURE_MIN_DELTA; 0 = capture every turn)\n",
            g_capture_min_delta);

    g_running      = true;
    g_thread       = std::thread(persister_main);
    g_fetch_thread = std::thread(fetch_main);

    return 0;
}

//------------------------------------------------------------
//Func : qml_llama_shutdown
//stops and joins both background threads (pending saves drain first), frees
//unclaimed fetch blobs, tears down inotify, and closes session storage
//  in: none
// out: none; no-op when init never ran; side effects: closes SQLite/LMDB
//------------------------------------------------------------
void qml_llama_shutdown(void)
{
    if (!g_running)
    {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_stop = true;
    }
    g_cv.notify_all();

    {
        std::lock_guard<std::mutex> lock(g_fetch_mutex);
        g_fetch_stop = true;
    }
    g_fetch_cv.notify_all();

    if (g_thread.joinable())
    {
        g_thread.join();
    }

    if (g_fetch_thread.joinable())
    {
        g_fetch_thread.join();
    }

    {
        std::lock_guard<std::mutex> lock(g_fetch_mutex);
        for (auto & kv : g_fetch_results)
        {
            free(kv.second.data);
        }
        g_fetch_results.clear();
        g_fetch_jobs.clear();
        g_fetch_stop = false;
    }

#ifdef __linux__
    inotify_teardown();
#endif

    g_running = false;
    g_stop    = false;

    qml_store_close();
}

//------------------------------------------------------------
//Func : qml_sync_geometry
//Derives the live context's slot geometry and hands it to the session layer
//so a blob written under a DIFFERENT geometry can be rejected before it is
//read (see qml_session_set_geometry).
//
//n_stream is not exposed directly, but it is derivable from public API:
//llama.cpp sets n_ctx_seq = n_ctx when the KV pool is unified (n_stream = 1)
//and n_ctx / n_seq_max otherwise (n_stream = n_seq_max). Verified against
//llama-context.cpp's cparams setup and llama-kv-cache.cpp:82.
//------------------------------------------------------------
//------------------------------------------------------------
//Func : qml_llama_sync_geometry
//Public one-shot form of qml_sync_geometry, called by the server once the
//context exists (server init()). Needed because the ASYNC FETCH path reads a
//blob from a background thread before any restore entry point has run: on a
//fresh process the very first session would otherwise be fetched with the
//geometry still unknown, handed to the engine, and rejected there (observed
//exactly once per restart before this existed).
//  in: ctx — the server's live llama_context
// out: none; side effects: publishes the geometry to g_cur_n_stream and the
//      session layer (qml_session_set_geometry)
//------------------------------------------------------------
void qml_llama_sync_geometry(struct llama_context * ctx)
{
    qml_sync_geometry(ctx);
}

//------------------------------------------------------------
//Func : qml_llama_restore
//synchronous paired restore: applies the newest saved KV state (pending save
//first, then the stored blob) into the slot and returns the matching token log
//  in: ctx_tgt — live context; slot_id — destination seq_id
//  in: session_id / model_hash — identity of the saved session
//  in: out_tokens / out_n_tokens — malloc'd token array + length on success
//      (free via qml_llama_restore_done)
// out: 1 = restored, 0 = nothing to restore, -1 = state existed but the engine
//      rejected it (or alloc failed); side effects: reads SQLite/LMDB and
//      writes KV state into the engine slot
//------------------------------------------------------------
int qml_llama_restore(
    struct llama_context * ctx_tgt,
    int32_t                slot_id,
    const char *           session_id,
    const char *           model_hash,
    llama_token **         out_tokens,
    size_t *               out_n_tokens)
{
    *out_tokens   = nullptr;
    *out_n_tokens = 0;

    qml_sync_geometry(ctx_tgt);

    {
        std::vector<uint8_t>     pstate;
        std::vector<llama_token> ptoks;
        // a tokens-only pending job (capture throttle) has no state to apply
        // -- fall through to the stored blob, which restores as a prefix
        if (pending_save_lookup(session_id, model_hash, pstate, ptoks) && !pstate.empty())
        {
            const size_t nset = llama_state_seq_set_data(
                ctx_tgt, pstate.data(), pstate.size(), slot_id);
            if (nset == 0)
            {
                return -1;
            }
            llama_token * out = (llama_token *) malloc(ptoks.size() * sizeof(llama_token));
            if (out == nullptr)
            {
                return -1;
            }
            memcpy(out, ptoks.data(), ptoks.size() * sizeof(llama_token));
            *out_tokens   = out;
            *out_n_tokens = ptoks.size();
            return 1;
        }
    }

    void *    state_data = nullptr;
    size_t    state_size = 0;
    int32_t * tok_ids    = nullptr;
    size_t    n_tok      = 0;

    const int rc = qml_session_load(
        session_id, model_hash, &state_data, &state_size, &tok_ids, &n_tok);

    if (rc != 0 || state_data == nullptr)
    {
        return 0; // nothing to restore
    }

    const size_t nset = llama_state_seq_set_data(
        ctx_tgt, (const uint8_t *) state_data, state_size, slot_id);

    if (nset == 0)
    {
        qml_session_load_done(state_data, tok_ids);
        return -1; // prior state existed but restoring it failed
    }

    // llama_token happens to be int32_t today, but convert explicitly
    // rather than assume the width never changes
    llama_token * out = (llama_token *) malloc(n_tok * sizeof(llama_token));
    for (size_t i = 0; i < n_tok; ++i)
    {
        out[i] = (llama_token) tok_ids[i];
    }

    qml_session_load_done(state_data, tok_ids);

    *out_tokens   = out;
    *out_n_tokens = n_tok;

    return 1;
}

//------------------------------------------------------------
//Func : qml_llama_peek
//tokens-only lookup: returns the session's saved token log (pending save
//first, then the store) without copying the potentially multi-GB state blob
//  in: session_id / model_hash — identity of the saved session
//  in: out_tokens / out_n_tokens — malloc'd token array + length on success
//      (free via qml_llama_restore_done)
// out: 1 = tokens returned, 0 = nothing stored, -1 = load or alloc error
//------------------------------------------------------------
int qml_llama_peek(
    const char *   session_id,
    const char *   model_hash,
    llama_token ** out_tokens,
    size_t *       out_n_tokens)
{
    *out_tokens   = nullptr;
    *out_n_tokens = 0;

    {
        std::vector<uint8_t>     pstate;
        std::vector<llama_token> ptoks;
        if (pending_save_lookup(session_id, model_hash, pstate, ptoks) && !ptoks.empty())
        {
            llama_token * out = (llama_token *) malloc(ptoks.size() * sizeof(llama_token));
            if (out == nullptr)
            {
                return -1;
            }
            memcpy(out, ptoks.data(), ptoks.size() * sizeof(llama_token));
            *out_tokens   = out;
            *out_n_tokens = ptoks.size();
            return 1;
        }
    }

    // [TAWT-CORE 2026-08-23] engine-standalone window trim. This peek is the
    // safe call site: it runs on the restore/admission path, BEFORE any KV
    // state exists for the session (a pending save returned above -- its log
    // is newer than the store and must not be trimmed under it; that session
    // gets trimmed on a later cold visit). Both envs required:
    //   QML_TRIM_WINDOW — restore-window token budget (0/unset = off)
    //   QML_TRIM_EOT    — the model's end-of-turn token id (turn boundary;
    //                     llama-3: 128009, ChatML/qwen3: 151645)
    // Gateway deployments keep their per-user trim; with both active the
    // gateway usually trims first and this call no-ops (same policy).
    {
        static const size_t trim_window = []() {
            const char * e = getenv("QML_TRIM_WINDOW");
            const long   v = (e != NULL) ? atol(e) : 0;
            return v > 0 ? (size_t) v : (size_t) 0;
        }();
        static const int32_t trim_eot = []() {
            const char * e = getenv("QML_TRIM_EOT");
            return (e != NULL) ? (int32_t) atol(e) : (int32_t) 0;
        }();
        if (trim_window > 0 && trim_eot != 0)
        {
            const long trimmed = qml_window_trim(session_id, trim_window,
                                                       trim_eot);
            if (trimmed > 0)
            {
                fprintf(stderr, "[qml][tawt] '%s': archived %ld token(s)"
                        " (window %zu)\n", session_id, trimmed, trim_window);
            }
            else if (trimmed < 0)
            {
                fprintf(stderr, "[qml][tawt] '%s': trim failed -- log left"
                        " intact\n", session_id);
            }
        }
    }

    int32_t * tok_ids = nullptr;
    size_t    n_tok   = 0;

    // tokens-only load: same model_hash and token-log/meta pairing checks as
    // the full load, plus a blob existence check (header read) -- but no
    // multi-GB blob copy. The copy was measured as a whole second of the
    // warm restore at 128K tokens, spent on data this peek never uses.
    const int rc = qml_session_load_tokens(
        session_id, model_hash, &tok_ids, &n_tok);

    if (rc != 0 || n_tok == 0)
    {
        if (rc == 0)
        {
            qml_session_load_done(nullptr, tok_ids);
        }
        return (rc < 0) ? -1 : 0;
    }

    llama_token * out = (llama_token *) malloc(n_tok * sizeof(llama_token));
    if (out == nullptr)
    {
        qml_session_load_done(nullptr, tok_ids);
        return -1;
    }

    for (size_t i = 0; i < n_tok; ++i)
    {
        out[i] = (llama_token) tok_ids[i];
    }

    qml_session_load_done(nullptr, tok_ids);

    *out_tokens   = out;
    *out_n_tokens = n_tok;

    return 1;
}

//------------------------------------------------------------
//Func : qml_llama_restore_known
//
//Restore variant for callers that ALREADY hold the session's token array (the
//llama-server request path peeks it in launch_slot_with_task() to prepend the
//history). Skipping the second token-log SELECT is the point: at 8K tokens the
//old path read 7,953 rows twice per restore.
//
//expect_n_tokens is checked against the count recorded by the save that wrote
//this blob. A mismatch means the caller's tokens and this blob are from
//different saves, so this returns 0 ("nothing usable") and the caller falls
//back to qml_llama_restore(), which re-reads both together.
//
//returns 1 restored, 0 nothing usable, -1 prior state existed but set failed.
//  in: ctx_tgt — live context; slot_id — destination seq_id
//  in: session_id / model_hash — identity of the saved session
//  in: expect_n_tokens — length of the token array the caller already holds
// out: see returns line above; side effects: reads SQLite/LMDB and writes KV
//      state into the engine slot (re-framing the blob if geometry differs)
//------------------------------------------------------------
int qml_llama_restore_known(
    struct llama_context * ctx_tgt,
    int32_t                slot_id,
    const char *           session_id,
    const char *           model_hash,
    size_t                 expect_n_tokens)
{
    qml_sync_geometry(ctx_tgt);

    {
        std::vector<uint8_t>     pstate;
        std::vector<llama_token> ptoks;
        if (pending_save_lookup(session_id, model_hash, pstate, ptoks) && !pstate.empty())
        {
            if (ptoks.size() > expect_n_tokens)
            {
                // the pending save holds MORE tokens than the caller's array --
                // different lineage; report "nothing usable" so the caller
                // falls back to qml_llama_restore(), which reads
                // tokens and state together (and itself checks pending first).
                // (fewer tokens = a prefix of the caller's log: apply it and
                // let the caller recompute the tail -- capture throttle case)
                return 0;
            }
            const size_t nset = llama_state_seq_set_data(
                ctx_tgt, pstate.data(), pstate.size(), slot_id);
            return (nset == 0) ? -1 : 1;
        }
    }

    void *   state_data   = nullptr;
    size_t   state_size   = 0;
    int64_t  n_meta       = 0;
    uint32_t blob_nstream = 0;

    // _any: no geometry guard here, because THIS path can repair a mismatch
    // by re-framing the blob instead of discarding it
    const int rc = qml_session_load_any(
        session_id, model_hash, &state_data, &state_size, &n_meta, &blob_nstream);

    if (rc != 0 || state_data == nullptr)
    {
        return 0;
    }

    const uint32_t cur_nstream = g_cur_n_stream.load(std::memory_order_relaxed);
    if (blob_nstream != 0 && cur_nstream != 0 && blob_nstream != cur_nstream)
    {
        size_t   fixed_size = 0;
        void *   fixed      = reframe_state_blob(state_data, state_size, cur_nstream, &fixed_size);
        qml_session_load_done(state_data, nullptr);
        if (fixed == nullptr)
        {
            fprintf(stderr,
                    "[qml] session '%s': blob written with n_stream=%u, context uses %u, "
                    "and it could not be re-framed -- recomputing from the token log\n",
                    session_id, blob_nstream, cur_nstream);
            return 0;
        }
        fprintf(stderr,
                "[qml] session '%s': re-framed blob n_stream %u -> %u (%zu -> %zu bytes)\n",
                session_id, blob_nstream, cur_nstream, state_size, fixed_size);
        state_data = fixed;
        state_size = fixed_size;
    }

    if ((size_t) n_meta > expect_n_tokens)
    {
        // blob holds MORE than the caller's token array -- different lineage.
        // (fewer = a prefix of the log, accepted: capture-throttled sessions
        // restore the prefix and recompute only the tail)
        fprintf(stderr,
                "[qml] restore_known: session '%s' meta has %lld tokens but caller holds %zu "
                "-- falling back to the paired load\n",
                session_id, (long long) n_meta, expect_n_tokens);
        qml_session_load_done(state_data, nullptr);
        return 0;
    }

    // NOTE: set_data itself is NOT the expensive part -- instrumented on
    // 2026-07-31 at 43 ms for a 280 MiB q4_0 blob on an 8K session. What used to
    // dominate the resume was doing this work inside the client's first request;
    // --qml-restore-on-start moves it to startup and the resume TTFT drops from
    // 248 ms to 43 ms.
    const size_t nset = llama_state_seq_set_data(
        ctx_tgt, (const uint8_t *) state_data, state_size, slot_id);

    qml_session_load_done(state_data, nullptr);

    return (nset == 0) ? -1 : 1;
}

//------------------------------------------------------------
//Func : qml_llama_fetch_begin
//starts a background fetch of the session's state blob. Idempotent: a
//fetch already pending or ready for this session is left alone.
//  in: session_id / model_hash — identity of the saved session to fetch
// out: none; no-op before init; side effects: enqueues a job for the fetch
//      thread, which reads SQLite/LMDB in the background
//------------------------------------------------------------
void qml_llama_fetch_begin(const char * session_id, const char * model_hash)
{
    if (!g_running)
    {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(g_fetch_mutex);
        if (g_fetch_results.count(session_id) != 0)
        {
            return; // pending or ready -- take() will consume it
        }
        g_fetch_results[session_id] = qml_fetch_entry(); // status 0 = in progress
        g_fetch_jobs.emplace_back(session_id, model_hash);
    }
    g_fetch_cv.notify_one();
}

//------------------------------------------------------------
//Func : qml_llama_fetch_pending
//non-consuming check: 1 = a fetch for this session is still running (the
//caller should defer, not block), 0 = no fetch in flight (ready, failed,
//or never started -- take() disambiguates).
//  in: session_id — session whose background fetch status is being checked
// out: 1 = fetch still running, 0 = no fetch in flight (status read only,
//      consumes nothing)
//------------------------------------------------------------
int qml_llama_fetch_pending(const char * session_id)
{
    std::lock_guard<std::mutex> lock(g_fetch_mutex);
    auto it = g_fetch_results.find(session_id);
    return (it != g_fetch_results.end() && it->second.status == 0) ? 1 : 0;
}

//------------------------------------------------------------
//Func : qml_llama_fetch_take
//returns 0 ready (*out_state malloc-owned by the caller, free() it after
//set_data), 1 fetch still in progress (retry next loop iteration -- do NOT
//block), 2 nothing usable (no fetch, fetch failed, or the fetched blob
//describes a different token count than the caller expects -- fall back to
//the synchronous paired load, which re-reads coherently).
//  in: session_id — session whose fetched blob to claim
//  in: expect_n_tokens — caller's token count; a blob recording MORE tokens is
//      a different lineage and is rejected (fewer = prefix, accepted)
//  in: out_state / out_size — the blob on return 0 (ownership moves to caller)
// out: 0 = ready, 1 = still in progress, 2 = nothing usable; side effects:
//      erases the fetch entry on 0 and 2
//------------------------------------------------------------
int qml_llama_fetch_take(
    const char * session_id,
    size_t       expect_n_tokens,
    void **      out_state,
    size_t *     out_size)
{
    *out_state = nullptr;
    *out_size  = 0;

    std::lock_guard<std::mutex> lock(g_fetch_mutex);

    auto it = g_fetch_results.find(session_id);
    if (it == g_fetch_results.end())
    {
        return 2;
    }
    if (it->second.status == 0)
    {
        return 1;
    }

    // n_meta > expect = different lineage (reject); n_meta < expect = a prefix
    // of the caller's log (accepted -- capture-throttled blob, tail recomputes)
    if (it->second.status != 1 || (size_t) it->second.n_meta > expect_n_tokens)
    {
        free(it->second.data);
        g_fetch_results.erase(it);
        return 2;
    }

    *out_state = it->second.data;
    *out_size  = it->second.size;
    g_fetch_results.erase(it); // ownership moves to the caller
    return 0;
}

//------------------------------------------------------------
//Func : qml_llama_state_apply
//the loop-thread half of an async restore: set_data only. Must run on the
//thread that owns llama_context (concurrent set_data vs decode is unsafe).
//returns 1 applied, -1 failed.
//  in: ctx_tgt — live context; slot_id — destination seq_id
//  in: state_data / state_size — blob bytes previously handed out by fetch_take
// out: 1 = applied, -1 = engine rejected the blob; side effects: writes KV
//      state into the engine slot and logs one [qml][timing] apply line
//------------------------------------------------------------
int qml_llama_state_apply(
    struct llama_context * ctx_tgt,
    int32_t                slot_id,
    const void *           state_data,
    size_t                 state_size)
{
    const auto   t0   = std::chrono::steady_clock::now();
    const size_t nset = llama_state_seq_set_data(
        ctx_tgt, (const uint8_t *) state_data, state_size, slot_id);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    // the other half of the restore floor: set_data on the loop thread
    fprintf(stderr, "[qml][timing] apply slot=%d size=%zu ms=%.1f t=%.3f\n",
            slot_id, state_size, ms,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    return (nset == 0) ? -1 : 1;
}

//------------------------------------------------------------
//Func : qml_llama_restore_done
//frees a token array returned by try_restore or peek_tokens
//  in: tokens — malloc'd array (NULL is safe)
// out: none
//------------------------------------------------------------
void qml_llama_restore_done(llama_token * tokens)
{
    free(tokens);
}

//------------------------------------------------------------
//Func : qml_llama_save
//loop-thread half of a save at slot release: captures the slot's KV state
//device-to-host (or, when the capture throttle applies, only the token log)
//and enqueues an async persist job for the persister thread
//  in: ctx_tgt — live context; slot_id — source seq_id
//  in: session_id / user_id / model_hash — identity to persist under
//  in: tokens / n_tokens — the session's full token log at this point
// out: none; no-op before init or when the engine reports no state
//      side effects: runs the profiled-largest per-turn D2H copy (unless
//      throttled), updates the capture-throttle map, pushes to g_queue
//------------------------------------------------------------
void qml_llama_save(
    struct llama_context * ctx_tgt,
    int32_t                slot_id,
    const char *           session_id,
    const char *           user_id,
    const char *           model_hash,
    const llama_token *    tokens,
    size_t                 n_tokens)
{
    if (!g_running)
    {
        return;
    }

    qml_sync_geometry(ctx_tgt);

    // capture throttle: if this turn added fewer than g_capture_min_delta new
    // tokens since the last FULL capture, persist the token log only and keep
    // the previous blob. Data safety is unchanged (the log is the source of
    // truth); the blob merely goes prefix-stale, and the restore path applies
    // a prefix blob + recomputes the tail.
    if (g_capture_min_delta > 0)
    {
        std::lock_guard<std::mutex> clock_(g_capture_mutex);
        auto it = g_last_capture_n.find(session_id);
        if (it != g_last_capture_n.end() &&
            n_tokens >= it->second &&
            n_tokens - it->second < g_capture_min_delta)
        {
            fprintf(stderr, "[qml][timing] capture SKIP slot=%d session=%s delta=%zu<%zu (tokens-only save)\n",
                    slot_id, session_id, n_tokens - it->second, g_capture_min_delta);
            qml_save_job job;
            job.session_id  = session_id;
            job.user_id     = user_id;
            job.model_hash  = model_hash;
            job.seq_id      = slot_id;
            job.tokens_only = true;
            job.tokens.assign(tokens, tokens + n_tokens);
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_queue.push_back(std::move(job));
            }
            g_cv.notify_one();
            return;
        }
    }

    const auto   t0 = std::chrono::steady_clock::now();
    const size_t sz = llama_state_seq_get_size(ctx_tgt, slot_id);
    if (sz == 0)
    {
        return;
    }

    std::vector<uint8_t> buf(sz);
    const size_t nget = llama_state_seq_get_data(ctx_tgt, buf.data(), buf.size(), slot_id);
    if (nget == 0)
    {
        return;
    }
    buf.resize(nget);

    // the per-turn cost paid on the loop thread at slot release: sizing,
    // allocation and the device-to-host state capture, before the persister
    // thread takes over. Profiled 2026-08-03: this was the LARGEST single
    // component of a 32K session turn (~520 ms), bigger than the restore
    // fetch+apply combined.
    // [WA-SKIP 2026-09-02] Measured write amplification of the store path -- records the
    // cumulative written/skipped bytes up to this capture (basis for the reviewer comment on edge TBW).
    {
        unsigned long long wa_w = 0, wa_s = 0;
        qml_lmdb_counters(&wa_w, &wa_s);
        fprintf(stderr, "[qml][wa] written=%llu skipped=%llu (skip %.1f%%)\n",
                wa_w, wa_s,
                (wa_w + wa_s) ? 100.0 * (double) wa_s / (double)(wa_w + wa_s)
                              : 0.0);
    }
    fprintf(stderr, "[qml][timing] capture slot=%d size=%zu ms=%.1f t=%.3f\n",
            slot_id, nget,
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count(),
            std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count());

    qml_save_job job;
    job.session_id = session_id;
    job.user_id    = user_id;
    job.model_hash = model_hash;
    job.seq_id     = slot_id;
    job.state_data = std::move(buf);
    job.tokens.assign(tokens, tokens + n_tokens);

    {
        std::lock_guard<std::mutex> clock_(g_capture_mutex);
        g_last_capture_n[session_id] = n_tokens;
    }

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_queue.push_back(std::move(job));
    }
    g_cv.notify_one();
}

//------------------------------------------------------------
//Func : qml_llama_capture
//BACKUP_NOW/MIGRATE must produce a CURRENT blob regardless of the capture
//throttle -- dropping the session's throttle entry makes the next
//enqueue_save capture fully.
//  in: session_id — session whose throttle entry to drop
// out: none; side effects: mutates only the in-process throttle map
//------------------------------------------------------------
void qml_llama_capture(const char * session_id)
{
    std::lock_guard<std::mutex> clock_(g_capture_mutex);
    g_last_capture_n.erase(session_id);
}

//------------------------------------------------------------
//Func : qml_llama_set_wake
//registers the callback the background threads use to nudge the server loop
//(after a live-KV control op is queued or an async fetch result is ready)
//  in: cb — function to invoke (NULL disables); userdata — passed through to cb
// out: none
//------------------------------------------------------------
void qml_llama_set_wake(qml_wake_cb_t cb, void * userdata)
{
    std::lock_guard<std::mutex> lock(g_wake_mutex);
    g_wake_cb = cb;
    g_wake_ud = userdata;
}

//------------------------------------------------------------
//Func : qml_llama_control_poll
//model-thread drain of the live-KV control-op queue (RESTORE/MIGRATE/...)
//claimed by the persister; called at the update_slots() safe point
//  in: out_cmd — filled with the oldest queued command on success
// out: 0 = command returned (popped from the queue), 1 = queue empty
//------------------------------------------------------------
int qml_llama_control_poll(qml_control_cmd_t * out_cmd)
{
    std::lock_guard<std::mutex> lock(g_ctrl_mutex);
    if (g_ctrl_ops.empty())
    {
        return 1;
    }

    *out_cmd = g_ctrl_ops.front();
    g_ctrl_ops.pop_front();
    return 0;
}

//------------------------------------------------------------
//Func : qml_llama_control_done
//reports a live-KV control op's outcome back to the control plane
//  in: cmd_id — row id of the command; status — "DONE" or "FAILED"
// out: none; side effects: updates the command's status row in SQLite
//------------------------------------------------------------
void qml_llama_control_done(int64_t cmd_id, const char * status)
{
    qml_control_set_status(cmd_id, status);
}

//------------------------------------------------------------
//Func : qml_llama_recent
//lists the most recently active llama.cpp sessions (--qml-restore-on-start)
//  in: limit — maximum number of session ids to return
//  in: out_ids / out_n — malloc'd id array + count on success
// out: 0 = ok (free via qml_llama_recent_free),
//      non-zero = SQLite error
//------------------------------------------------------------
int qml_llama_recent(size_t limit, char *** out_ids, size_t * out_n)
{
    return qml_sqlite_recent(QML_ENGINE_LLAMA_CPP, limit, out_ids, out_n);
}

//------------------------------------------------------------
//Func : qml_llama_recent_free
//frees the id array returned by qml_llama_recent
//  in: ids — array to free; n — its element count
// out: none
//------------------------------------------------------------
void qml_llama_recent_free(char ** ids, size_t n)
{
    qml_sqlite_recent_free(ids, n);
}

} // extern "C"

//===========================================================================//
// [FTS-RECALL] BEGIN — automatic re-injection of archived turns.
// Additive-only feature block (removal instructions in qml_sqlite_store.h).
#include <cctype>
#include <utility>
#include <vector>
// [HYBRID 2026-08-27] Semantic (embedding) search — uses the httplib (client) and
// nlohmann/json (response parser) that llama.cpp vendors, as is.
#include <httplib.h>
#include <nlohmann/json.hpp>
// Flow per request (only when QML_ARCHIVE_RECALL=1 and the session has
// archived rows): lazy-build the per-turn FTS index from the token archive
// (detokenizing each turn once via the engine-supplied callback), sanitize
// the new turn's text into an FTS5 OR-query, take the top-k turns, and
// return their ORIGINAL token ranges (template tokens included, so the
// injected block is turn-aligned by construction).
//===========================================================================//

//qml_llama_recall_on
// in: none (env read once, cached)
//out: 1 if QML_ARCHIVE_RECALL=1, else 0
extern "C" int qml_llama_recall_on(void)
{
    static const int enabled = []() {
        const char * e = getenv("QML_ARCHIVE_RECALL");
        return (e != NULL && strcmp(e, "1") == 0) ? 1 : 0;
    }();
    return enabled;
}

//rig_budget (static)
// in: none (env QML_ARCHIVE_RECALL_BUDGET, default 1024 tokens)
//out: max tokens the recall injection may occupy
//rig_topk (static)
// in: none (env QML_ARCHIVE_RECALL_TOPK, default 3, clamped to [1,64])
//out: max number of archived turns the FTS search may return per recall
// [Defect fix 2026-09-03, TEST-LEDGER row 85] The upper bound was fixed at 32, so
// the setting QML_ARCHIVE_RECALL_TOPK=50 was cut to 32 (checked against 266 server log entries:
// all 126 cases with more than 32 candidate rows injected 32 rows). The upper bound is raised to 64
// to match the FACTS path buffer (txts[64]).
static int rig_topk(void)
{
    static const int topk = []() {
        const char * e = getenv("QML_ARCHIVE_RECALL_TOPK");
        int v = (e != NULL) ? atoi(e) : 3;
        if (v < 1)  v = 1;
        if (v > 64) v = 64;
        return v;
    }();
    return topk;
}

//rig_spread (static)
// in: none (env QML_ARCHIVE_RECALL_SPREAD_W = position window, default 0 = off;
//           QML_ARCHIVE_RECALL_SPREAD_K = max picks per window, default 2)
//out: writes the two values through the pointers
// why: retrieval finds the evidence (recall@10 = 97% on multi-session) but the
//      end-to-end score stays low because a single strongly-matching region of
//      the archive takes every slot. The archive only gives us positions, so
//      "one source conversation" is approximated by a position window: at most
//      K picks may fall within W tokens of each other, which forces the budget
//      to spread over distant regions of the history.
static void rig_spread(int64_t * out_w, int * out_k)
{
    static const int64_t w = []() {
        const char * e = getenv("QML_ARCHIVE_RECALL_SPREAD_W");
        int64_t v = (e != NULL) ? atoll(e) : 0;     // 0 = feature off (v1.1 behavior)
        return v < 0 ? 0 : v;
    }();
    static const int k = []() {
        const char * e = getenv("QML_ARCHIVE_RECALL_SPREAD_K");
        int v = (e != NULL) ? atoi(e) : 2;
        return v < 1 ? 1 : v;
    }();
    *out_w = w;
    *out_k = k;
}

// ===========================================================================
// [RIG-GATE 2026-08-14] RIG conf gate (Recall Injection Gate, admission by confidence)
// ---------------------------------------------------------------------------
// Design basis: section 8-v2 of the engine backlog record of 2026-08-11.
//   rig_score = sim x min(1, n_eff / N_min)      (size_ok is handled by the existing turn_cap)
//   if rig_score < conf_threshold, that candidate is not injected (no-inject).
// Purpose: prevent forcing in weakly supported candidates that pollute the context, and when there is
//       no supporting evidence, let the model keep answering "I don't know".
// Meaning of each term:
//   sim    = this candidate's BM25 rank divided by the top rank in the same result set (0~1].
//            FTS5 rank is negative and smaller means higher, so the top result is 1.0 and
//            lower results approach 0. It is a relative value within the set, not an absolute score.
//   n_eff  = number of effective words in the question excluding stopwords (a question like "what was that?" is close to 0)
//   N_min  = minimum number of effective words considered reliable (default 4)
//   conf_threshold = injection threshold (default 0.0 = conf gate disabled → previous behavior unchanged)
// ---------------------------------------------------------------------------
// How to revert (if errors occur):
//   1) Disable only: env variable QML_ARCHIVE_RECALL_CONF=0 (default) — every candidate passes, so
//      behavior becomes the same as before 2026-08-14.
//   2) Remove the code: delete the 3 blocks marked [RIG-GATE] (this helper group, the decision
//      condition in the selection loop, the out_score argument in storage) and set the last argument of the search call
//      back to NULL to restore the previous state.
// ===========================================================================

//rig_conf (static)
// in: none (env QML_ARCHIVE_RECALL_CONF, default 0.2)
//out: injection threshold for rig_score
// [RIG-GATE] Basis for the default value (measured 2026-08-14, zero accuracy loss at both scales):
//   2,000 turn : conf 0 -> 21/21, 493 tokens injected / conf 0.2 -> 21/21, 220 tokens injected
//   20,000 turn: conf 0 -> 27/27, 458 tokens injected / conf 0.2 -> 27/27, 228 tokens injected
//   (conf 0.5 lost 1 at 2,000 turns -> not adopted)
// To revert to disabled, run with QML_ARCHIVE_RECALL_CONF=0.
static double rig_conf(void)
{
    static const double conf = []() {
        const char * e = getenv("QML_ARCHIVE_RECALL_CONF");
        double v = (e != NULL) ? atof(e) : 0.2;
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        return v;
    }();
    return conf;
}

//rig_n_min (static)
// in: none (env QML_ARCHIVE_RECALL_NMIN, default 4)
//out: lower bound on the number of effective words in the question (below this, rig_score is scaled down proportionally)
static int rig_n_min(void)
{
    static const int nmin = []() {
        const char * e = getenv("QML_ARCHIVE_RECALL_NMIN");
        int v = (e != NULL) ? atoi(e) : 4;
        return v < 1 ? 1 : v;
    }();
    return nmin;
}

//rig_n_eff (static)
// in: text — raw text of the new turn (question)
//out: number of effective words excluding stopwords. The stopword list is limited to English
//     function words that do not contribute to search (kept short on purpose so content words are not removed).
// [CORE-MOVE 2026-08-23] The implementation moved to qml_rig_n_words in core/qml_rig.c —
// a single implementation shared with the vLLM/SGLang paths. This wrapper is kept to
// minimize the diff at the call sites.
static int rig_n_eff(const char * text)
{
    return qml_rig_n_words(text);
}

// [RIG-DEFAULTS 2026-08-27] minimum match and the in-context duplicate guard went through a verification
// period (env switches) and were promoted to default behavior, and the env variables were removed
// (cleanup policy following the feedback about too many env switches).
//
// minimum match (QML_RIG_MIN_MATCH_SINGLE_HIT): only "when the search returns exactly 1 result"
// it requires at least 2 query words to actually match. Basis (2026-08-27 wx2k 21-question A/B):
// an unconditional floor (=2) blocked the correct candidate of the indirect category in multi-hit sets,
// losing 18/21 -> 17/21 (correct pos=11453 blocked, only wrong pos=110000 injected);
// the defect #11 failure occurs only in sets reduced to 1 candidate (sim is 1.0 by definition)
// (all 4090 measured cases were single-candidate), and in multi-hit sets the relative conf
// can discriminate. The conditional application satisfies both properties.
//
// in-context duplicate guard: if the candidate's token sequence already exists in the live log tail (8192 tokens)
// right before injection, skip it (TEST-LEDGER defect #13 — prevents at the source the failure of
// repeated injection of the same content accumulating). Since it only skips content already present, it structurally
// cannot lower accuracy; verified lossless on both the component 18 questions and the wx2k 21 questions.
// [RIG-FLOOR 2026-10-22] the floor is 1: only a single hit that shares no query word is
// refused. With 2, a single archived hit that shares one rare word with the question
// ("Which city was hit by the hailstorm?" -> "hailstorm") was refused although it was the
// answer (2,000-turn weather test).
static const int    QML_RIG_MIN_MATCH_SINGLE_HIT = 1;
static const size_t QML_RIG_INCONTEXT_TAIL       = 8192;

// [HYBRID 2026-08-27] Semantic search — embedding server address (QML_EMBED_URL,
// e.g. http://127.0.0.1:58072 — a llama-server --embedding instance).
// Unset = semantic search fully off (lexical search only, previous behavior unchanged).
// Adopted design (measurement basis: TEST-LEDGER A-2 items 21 and 22): the lexical path and gate
// stay as they are, and the top QML_VEC_TOPK semantic candidates are added as a union —
// the two methods failed on different questions, and only the union reached 27/27 (hit@10).
//
// [HYBRID defect fix 2026-08-27, measured in hybrid2] Injection adds only the "top 1"
// semantic result. Reason: across all 21 questions, gold was semantic rank 1 in 20/21
// (gold cos 0.50~0.83), and ranks 2~3 were all unrelated turns (cos 0.39~0.67).
// The lowest gold cos (0.50) is below the highest noise cos (0.67), so an absolute cos floor
// cannot separate them. In the hybrid2 run that injected the top-3 as is, the path "rank 1 = gold
// already duplicates the lexical pick -> ranks 2~3 noise are added instead" injected unrelated turns
// and the 3B reader was pushed to wrong answers (fact@50 indirect: wrong answer 'Suwon' even though gold
// was in the context; fact@1500 indirect same type). Therefore:
//  - search retrieves up to the top QML_VEC_TOPK (for logs/diagnostics), but the injection candidate is rank 1 only.
//  - if rank 1 is the same pos as the lexical pick, the two searches agree -> nothing is added
//    (it does not fall back to ranks 2~3).
//  - QML_VEC_MIN_COS: lower bound so that unrelated questions (small talk, etc.) do not pull in
//    archived turns. Fixed at 0.45, below the measured gold minimum of 0.50.
static const int    QML_VEC_TOPK    = 3;
static const double QML_VEC_MIN_COS = 0.45;
static std::string qml_embed_url(void)
{
    static const std::string url = []() {
        const char * e = getenv("QML_EMBED_URL");
        return std::string(e != NULL ? e : "");
    }();
    return url;
}

//qml_vec_clean (static)
// in: text — FTS content or raw query text
//out: a cleaned copy for embedding input — removes lines consisting only of a chat template role word
//     ("user"/"assistant"/"system"/"tool") and "<|...|>" markers.
// Why: measured (2026-08-27 hybrid1 first run) — embedding without cleanup pulls the vectors toward
// the format boilerplate shared by all documents, cosine collapses into a flat 0.54~0.55
// band, unrelated turns get injected in addition, and even a previously correct question (fact@50
// direct) became wrong. The lexical path does not have this problem because the query-building filter
// excludes role words — the embedding path needs the equivalent cleanup.
static std::string qml_vec_clean(const std::string & text)
{
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size())
    {
        size_t eol = text.find('\n', i);
        if (eol == std::string::npos)
        {
            eol = text.size();
        }
        std::string line = text.substr(i, eol - i);
        // remove "<|...|>" markers
        for (;;)
        {
            const size_t a = line.find("<|");
            if (a == std::string::npos)
            {
                break;
            }
            const size_t b = line.find("|>", a);
            if (b == std::string::npos)
            {
                line.erase(a);
                break;
            }
            line.erase(a, b + 2 - a);
        }
        // after normalizing whitespace, drop lines consisting only of a role word
        size_t s = line.find_first_not_of(" \t");
        size_t e = line.find_last_not_of(" \t");
        std::string t = (s == std::string::npos) ? "" : line.substr(s, e - s + 1);
        if (!t.empty() && t != "user" && t != "assistant" &&
            t != "system" && t != "tool")
        {
            if (!out.empty())
            {
                out += '\n';
            }
            out += t;
        }
        i = eol + 1;
    }
    return out;
}

//qml_embed_texts (static)
// in: texts — strings to embed (1 query or an index batch) — must be text
//     that went through qml_vec_clean before the call
//out: true on success, with float vectors in out (order preserved). No server response / parse failure returns
//     false — the caller skips only the semantic search and continues with the lexical results.
// [EMBED-C 2026-08-29] In-process embedding preferred path — when env QML_EMBED_MODEL
// is set, the C engine in the same process (router/src/qml_router.c) is used instead of
// a separate server (QML_EMBED_URL). Purpose: keep a single-process topology.
// QML_EMBED_NGL: number of GPU layers for the embedding model (default 0 = CPU only — does not
// touch the generation engine's VRAM).
// [ENTITY-INDEX 2026-08-29] Extracts identifiers from documents and queries with a deterministic rule.
// Rule: 1~4 uppercase English letters + '-' + 2 or more digits (P-107, C-87, MX-2071 …).
// Pattern/crew/asset numbers in the equipment domain take this form, and they do not overlap with
// natural-language words, so false detections are practically none. env QML_ENTITY_PATTERN can restrict the prefix.
// [COND-SLOTS 2026-08-31] Condition slot scheme. When the deployment gives env QML_COND_SLOTS
// a list of `field=value` entries (semicolon-separated), that order becomes the bit slot number.
// Reason for choosing a list instead of automatic collection: the author's instruction ("accuracy first
// even if the deployment grows"). Because the list is fixed there are no false detections, and slot numbers
// stay stable between load and query (if they shifted, the bit checks would be meaningless).
// Unit test measurement (tests/unit_rule_search.py): this method top-1 100%,
// lexical search 46~58.7%, embedding 41.3%.
static const std::vector<std::string> & qml_cond_slots(void)
{
    static const std::vector<std::string> slots = []() {
        std::vector<std::string> v;
        const char * e = getenv("QML_COND_SLOTS");
        if (e != NULL)
        {
            std::string cur;
            for (const char * p = e; ; ++p)
            {
                if (*p == ';' || *p == '\0')
                {
                    if (!cur.empty()) { v.push_back(cur); }
                    cur.clear();
                    if (*p == '\0') { break; }
                }
                else { cur += (char) tolower((unsigned char) *p); }
            }
        }
        return v;
    }();
    return slots;
}

// Sets condition bits from text. If a `field=value` in the slot list appears
// verbatim in the text, that slot is set to 1. out_n_cond receives the number of slots set
// (used for search order — more specific records first).
static std::vector<unsigned char> qml_cond_encode(const std::string & text,
                                                  int * out_n_cond)
{
    const std::vector<std::string> & slots = qml_cond_slots();
    const size_t nbytes = (slots.size() + 7) / 8;
    std::vector<unsigned char> code(nbytes, 0);
    int n = 0;
    if (nbytes == 0)
    {
        if (out_n_cond != NULL) { *out_n_cond = 0; }
        return code;
    }
    std::string low;
    low.reserve(text.size());
    for (char c : text) { low += (char) tolower((unsigned char) c); }
    // [Longest match first 2026-09-02] With substring matching, the prefix of a long value
    // falsely triggers a shorter slot (measured: event value `data_categories=Orange data`
    // turned on slot `data_categories=Orange`, so an unrelated rule satisfied the subset check —
    // the direct cause of 2 wrong answers, TEST-LEDGER row 61). When a longer slot matches at the
    // same start position, that position is treated as consumed and the shorter slot is suppressed.
    // Bit numbers keep the original slot order (sorting only changes the evaluation order).
    std::vector<size_t> order(slots.size());
    for (size_t i = 0; i < order.size(); ++i) { order[i] = i; }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return slots[a].size() > slots[b].size();
    });
    std::set<size_t> claimed;   // start positions where a longer slot matched
    for (size_t oi = 0; oi < order.size(); ++oi)
    {
        const size_t i = order[oi];
        bool hit = false;
        for (size_t p = low.find(slots[i]); p != std::string::npos;
             p = low.find(slots[i], p + 1))
        {
            // [Value boundary check 2026-09-02] If the character right after the match continues the value
            // (e.g. event value `orange data` turning on slot `...=orange` as a prefix)
            // it is a false bit. field=value lists are `;`-separated (common to event sentences and KEYS
            // lines -- the load side was also aligned to `; ` separation), so the boundary
            // accepts only ';' '.' newline or end of text.
            const size_t e2 = p + slots[i].size();
            if (e2 < low.size())
            {
                const char nx = low[e2];
                if (nx != ';' && nx != '.' && nx != '\n' && nx != '\r')
                {
                    continue;
                }
            }
            if (claimed.count(p) == 0)
            {
                hit = true;
                claimed.insert(p);
                break;
            }
        }
        if (hit)
        {
            code[i / 8] |= (unsigned char) (1u << (i % 8));
            n++;
        }
    }
    if (out_n_cond != NULL) { *out_n_cond = n; }
    return code;
}

// [ENTITY-NAMES 2026-08-30] Support for domains without identifier tokens. When the deployment's entity
// name list is given in env QML_ENTITY_NAMES (semicolon-separated), those names are looked up
// verbatim in queries and documents and used as entities. Reason: when the query already states which session
// to look at (e.g. "For electric motor, ..."), similarity
// search is not needed; deterministic addressing from name -> owning session is enough.
// The list is fixed at deployment time so there are no false detections; if unset, previous behavior unchanged.
static const std::vector<std::string> & qml_entity_names(void)
{
    static const std::vector<std::string> names = []() {
        std::vector<std::string> v;
        const char * e = getenv("QML_ENTITY_NAMES");
        if (e != NULL)
        {
            std::string s(e), cur;
            for (char c : s)
            {
                if (c == ';')
                {
                    if (!cur.empty()) { v.push_back(cur); }
                    cur.clear();
                }
                else { cur += (char) tolower((unsigned char) c); }
            }
            if (!cur.empty()) { v.push_back(cur); }
        }
        return v;
    }();
    return names;
}

static std::vector<std::string> qml_extract_entities(const std::string & text)
{
    std::vector<std::string> out;
    {
        std::string low;
        low.reserve(text.size());
        for (char c : text) { low += (char) tolower((unsigned char) c); }
        for (const auto & nm : qml_entity_names())
        {
            if (low.find(nm) != std::string::npos &&
                std::find(out.begin(), out.end(), nm) == out.end())
            {
                out.push_back(nm);
            }
        }
    }
    const size_t n = text.size();
    for (size_t i = 0; i < n; i++)
    {
        if (!isupper((unsigned char) text[i]))
        {
            continue;
        }
        size_t j = i;
        while (j < n && isupper((unsigned char) text[j]) && (j - i) < 4)
        {
            j++;
        }
        if (j >= n || text[j] != '-' || j == i)
        {
            continue;
        }
        size_t k = j + 1;
        while (k < n && isdigit((unsigned char) text[k]))
        {
            k++;
        }
        if (k - (j + 1) >= 2)
        {
            std::string e = text.substr(i, k - i);
            if (std::find(out.begin(), out.end(), e) == out.end())
            {
                out.push_back(e);
            }
            i = k - 1;
        }
    }
    return out;
}

static bool qml_embed_inproc(const std::vector<std::string> & texts,
                             std::vector<std::vector<float>> & out)
{
    const char * mp = getenv("QML_EMBED_MODEL");
    if (mp == NULL || mp[0] == '\0')
    {
        return false;
    }
    static const int ok = []() {
        const char * e = getenv("QML_EMBED_NGL");
        const int ngl = (e != NULL) ? atoi(e) : 0;
        return qml_embed_init(getenv("QML_EMBED_MODEL"), ngl) == 0;
    }();
    if (!ok)
    {
        return false;
    }
    const int dim = qml_embed_dim();
    out.clear();
    for (const auto & t : texts)
    {
        std::vector<float> v((size_t) dim);
        if (qml_embed_text(t.c_str(), v.data(), dim) != 0)
        {
            fprintf(stderr, "[qml][embed-c] embed failed (len=%zu)\n", t.size());
            return false;
        }
        out.push_back(std::move(v));
    }
    return true;
}

static bool qml_embed_texts(const std::vector<std::string> & texts,
                            std::vector<std::vector<float>> & out)
{
    if (texts.empty())
    {
        return false;
    }
    if (getenv("QML_EMBED_MODEL") != NULL)
    {
        return qml_embed_inproc(texts, out);
    }
    const std::string url = qml_embed_url();
    if (url.empty())
    {
        return false;
    }
    try
    {
        httplib::Client cli(url);
        cli.set_connection_timeout(2, 0);
        cli.set_read_timeout(60, 0);
        nlohmann::json body;
        body["content"] = texts;
        auto res = cli.Post("/embedding", body.dump(), "application/json");
        if (!res || res->status != 200)
        {
            fprintf(stderr, "[qml][vec] embed server unreachable (%s)\n", url.c_str());
            return false;
        }
        nlohmann::json j = nlohmann::json::parse(res->body);
        out.clear();
        for (const auto & item : j)
        {
            const auto & emb = item.at("embedding");
            // llama-server returns [[f,...]] (nested per pooling) or [f,...]
            const auto & row = emb.at(0).is_array() ? emb.at(0) : emb;
            out.push_back(row.get<std::vector<float>>());
        }
        return out.size() == texts.size();
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "[qml][vec] embed failed: %s\n", e.what());
        return false;
    }
}
// =================== end of [RIG-GATE] helper group =================================

static size_t rig_budget(void)
{
    static const size_t budget = []() {
        const char * e = getenv("QML_ARCHIVE_RECALL_BUDGET");
        const long v = (e != NULL) ? atol(e) : 0;
        return (v > 0) ? (size_t) v : (size_t) 1024;
    }();
    return budget;
}

//rig_query_text (static)
// in: text — raw new-turn text
//out: FTS5 MATCH expression: up to 12 quoted alphanumeric words (len >= 2)
//     joined with OR; empty string when the text has no usable words
//
// [Defect fix 2026-08-21] Exclude chat template role names from the query.
// The question text is a detokenize of new_part, so role words such as "user"/"assistant"
// come along from the format, and the first word of indexed documents is also "user" for the
// same reason — so even a question with completely unrelated content matched at least 1 word
// in every document. In a session with only 1 candidate (right after demoting 1 image), the conf gate's
// sim is a relative value within the set (sole candidate = 1.0), so this false match went all the way
// to injection (4090 measured: an image turn of 84 tokens injected for a kimchi-jjigae question). Role words
// are format, not question content, so removing them is justified.
static std::string rig_query_text(const char * text)
{
    // [CORE-MOVE 2026-08-23] The implementation moved to qml_rig_query in core/qml_rig.c
    // (same behavior — excludes role words / Korean function words / pure numbers, 12-word upper bound).
    char * q = qml_rig_query(text);
    if (q == NULL)
    {
        return std::string();
    }
    std::string out(q);
    free(q);
    return out;
}

//rig_index (static)
// in: session_id / eot_token / detok+ud — see the public function
//out: 0 when the FTS rows for this session exist afterwards, -1 on error
// side effects: inserts one qml_archive_fts row per archived turn
static int rig_index(const char * session_id, int32_t eot_token,
                              char * (*detok)(void *, const int32_t *, size_t),
                              void * ud)
{
    int32_t * pos = NULL;
    int32_t * tok = NULL;
    size_t    n   = 0;
    if (qml_sqlite_archive_fetch(session_id, &pos, &tok, &n) != 0)
    {
        return -1;
    }
    // [defect fix 2026-08-14] Indexing assistant responses as separate documents lets short confirmation
    // sentences ("Report from Daegu for day 12.") get the score boost from BM25 document length
    // normalization and take over the top-k (2 failures in the long-running weather
    // test came from this — 1,918 of 3,837 indexed documents were confirmation sentences, and 6 of the top
    // 10 for the failed questions were 12-token confirmation sentences while the correct document was out of the ranking).
    // Fix: merge the assistant segment into the preceding user segment and index them as one document
    // (content preserved, short-document boost removed, injection unit stays turn-aligned).
    // env QML_ARCHIVE_RECALL_PAIR_TURNS=0 restores the previous behavior.
    const bool pair_turns = []() {
        const char * e = getenv("QML_ARCHIVE_RECALL_PAIR_TURNS");
        return (e == NULL) || (atoi(e) != 0);          // default on
    }();
    // [FTS-BULK-FIX 2026-08-14] Wrap all documents in one transaction. Without this,
    // every document triggers a WAL commit (a network fsync on NFS), and the first index build of
    // 19,446 documents took 77.7 seconds. To revert, delete this call and the batch_end below
    // (same behavior, only speed differs).
    qml_sqlite_fts_begin();
    size_t s = 0;
    int    n_turns = 0;
    int64_t pend_lo = -1, pend_hi = -1;                // pending document range
    std::string pend_text;
    auto flush_pending = [&]() {
        if (pend_lo >= 0 && !pend_text.empty())
        {
            qml_sqlite_fts_add(session_id, pend_lo, pend_hi,
                                          pend_text.c_str());
            // when a new document is indexed, clear this session's vector-backfill-done cache
            {
                std::lock_guard<std::mutex> lk(g_vec_done_mtx);
                g_vec_done.erase(session_id);
            }
            // [ENTITY-INDEX] index identifiers of the same document for condition lookup
            for (const auto & e : qml_extract_entities(pend_text))
            {
                qml_sqlite_entity_add(session_id, e.c_str(),
                                              pend_lo, pend_hi);
            }
            // [COND-SLOTS 2026-08-31] Also record the condition bits. If the slot
            // If the list (QML_COND_SLOTS) is absent this does nothing, so it is the
            // same as the previous behavior. A turn with no condition at all is not
            // recorded and is excluded from condition search.
            {
                int n_cond = 0;
                const std::vector<unsigned char> cc =
                    qml_cond_encode(pend_text, &n_cond);
                if (n_cond > 0 && !cc.empty())
                {
                    qml_sqlite_cond_put(session_id, pend_lo, pend_hi,
                                                cc.data(), (int) cc.size(), n_cond);
                }
            }
            n_turns++;
        }
        pend_lo = pend_hi = -1;
        pend_text.clear();
    };
    for (size_t i = 0; i < n; i++)
    {
        const bool at_eot  = (tok[i] == eot_token);
        const bool at_tail = (i + 1 == n);
        if (!at_eot && !at_tail)
        {
            continue;
        }
        const size_t e = i;             // inclusive turn end index
        if (e - s + 1 >= 6)             // skip fragments too short to matter
        {
            char * text = detok(ud, tok + s, e - s + 1);
            if (text != NULL)
            {
                if (text[0] != '\0')
                {
                    const bool is_assistant =
                        pair_turns && strncmp(text, "assistant", 9) == 0;
                    if (is_assistant && pend_lo >= 0)
                    {
                        pend_hi = (int64_t) pos[e] + 1;   // merge into the previous document
                        pend_text += "\n";
                        pend_text += text;
                    }
                    else
                    {
                        flush_pending();
                        pend_lo   = (int64_t) pos[s];
                        pend_hi   = (int64_t) pos[e] + 1;
                        pend_text = text;
                    }
                }
                free(text);
            }
        }
        s = i + 1;
    }
    flush_pending();
    qml_sqlite_fts_end(1);   // [FTS-BULK-FIX] commit only once
    fprintf(stderr, "[qml][fts-recall] indexed %d archived turn(s) for '%s'\n",
            n_turns, session_id);
    free(pos);
    free(tok);
    return 0;
}

//------------------------------------------------------------
//Func : rig_idents (static)
//identifiers of a text: its words holding a digit and something else
//("EGR82Z", "P-107"); a trailing non-ASCII particle after the last ASCII
//letter or digit is cut. Pure numbers are no identifiers (the FTS query
//leaves them out too)
//  in: text — question or row text
// out: the identifiers, each once
//------------------------------------------------------------
static std::vector<std::string> rig_idents(const char * text)
{
    std::vector<std::string> out;
    const std::string t = text != NULL ? text : "";
    size_t i = 0;
    while (i < t.size())
    {
        while (i < t.size() && (isspace((unsigned char) t[i]) ||
                                (ispunct((unsigned char) t[i]) && t[i] != '-')))
        {
            i++;
        }
        size_t j = i;
        while (j < t.size() && !isspace((unsigned char) t[j]) &&
               !(ispunct((unsigned char) t[j]) && t[j] != '-'))
        {
            j++;
        }
        std::string w = t.substr(i, j - i);
        i = j;
        size_t last = std::string::npos;               // last ASCII letter or digit
        bool digit = false, other = false;
        for (size_t k = 0; k < w.size(); k++)
        {
            const unsigned char c = (unsigned char) w[k];
            if (isdigit(c))
            {
                digit = true;
                last  = k;
            }
            else if (isalpha(c))
            {
                other = true;
                last  = k;
            }
            else if (c >= 0x80 && last == std::string::npos)
            {
                other = true;                          // a name before the number
            }
        }
        if (!digit || !other || last == std::string::npos)
        {
            continue;
        }
        w.resize(last + 1);
        if (std::find(out.begin(), out.end(), w) == out.end())
        {
            out.push_back(w);
        }
    }
    return out;
}

//------------------------------------------------------------
//Func : rig_words (static)
//lower-case words of two or more characters (ASCII letters and digits, or
//multibyte characters)
//------------------------------------------------------------
static std::set<std::string> rig_words(const char * text)
{
    std::set<std::string> out;
    std::string w;
    for (const char * c = text != NULL ? text : ""; ; c++)
    {
        const unsigned char u = (unsigned char) *c;
        if (u != '\0' && (isalnum(u) || u >= 0x80))
        {
            w += (char) (u < 0x80 ? tolower(u) : u);
            continue;
        }
        if (w.size() >= 2)
        {
            out.insert(w);
        }
        w.clear();
        if (u == '\0')
        {
            break;
        }
    }
    return out;
}

//------------------------------------------------------------
//Func : rig_same_question (static)
//[RIG-SAMEQ 2026-10-22] an archived row that is only the question asked
//before, without its answer: injecting it adds the question, not the fact.
//A row is that question when it has no assistant part and either equals the
//new question (ignoring spaces) or holds only the new question's identifiers
//and at most two words the new question does not have
//  in: row_text — indexed text of the archived row ("user\n\n...")
//  in: query_text — the new question
// out: true when the row is the question asked before
//------------------------------------------------------------
static bool rig_same_question(const char * row_text, const char * query_text)
{
    if (row_text == NULL || query_text == NULL || strstr(row_text, "\nassistant") != NULL)
    {
        return false;
    }
    const char * u = row_text;
    if (strncmp(u, "user", 4) == 0)
    {
        u += 4;
    }
    auto squeeze = [](const char * text) {
        std::string o;
        for (const char * c = text; *c != '\0'; c++)
        {
            if (!isspace((unsigned char) *c))
            {
                o += *c;
            }
        }
        return o;
    };
    if (squeeze(u) == squeeze(query_text))
    {
        return true;
    }
    const std::vector<std::string> qids = rig_idents(query_text);
    for (const std::string & w : rig_idents(u))
    {
        if (std::find(qids.begin(), qids.end(), w) == qids.end())
        {
            return false;
        }
    }
    const std::set<std::string> rw = rig_words(u);
    size_t shared = 0;
    for (const std::string & w : rig_words(query_text))
    {
        shared += rw.count(w);
    }
    return rw.size() <= shared + 2;
}

//qml_llama_recall — see header for the in/out contract
extern "C" size_t qml_llama_recall(const char * session_id,
                                                  const char * query_text,
                                                  int32_t eot_token,
                                                  size_t n_ctx_avail,
                                                  char * (*detok)(void * ud, const int32_t * toks, size_t n),
                                                  void * ud,
                                                  int32_t ** out_tokens)
{
    if (out_tokens == NULL)
    {
        return 0;
    }
    *out_tokens = NULL;
    if (!qml_llama_recall_on() ||
        session_id == NULL || query_text == NULL || detok == NULL)
    {
        return 0;
    }
    // [RIG-TRIGGER 2026-08-29, user-approved] deterministic control of retrieval triggering.
    // When QML_ARCHIVE_RECALL_TRIGGER is set, search runs only on turns that
    // contain that string. Basis (TEST-LEDGER rows 34 and 34-1, measured): in an agent workload where
    // most turns are observations, triggering on every turn, lexical or hybrid alike, lets irrelevant
    // injection contamination outweigh the recall gain (all 177 injections were false triggers; with hybrid,
    // escalation collapsed 35 -> 7). This applies the principle of this experiment, where the deterministic layer decides WHEN and the model
    // answers WHAT, to search triggering as well.
    // When unset, the previous behavior is unchanged (question-driven workload — gateway deployment).
    {
        static const char * trig = getenv("QML_ARCHIVE_RECALL_TRIGGER");
        if (trig != NULL && trig[0] != '\0' &&
            strstr(query_text, trig) == NULL)
        {
            return 0;
        }
    }
    //=======================================================================//
    // [RIG-TIMER 2026-08-14] per-stage elapsed time measurement.
    // Why: enabling recall increases the time per question (at 20,000-turn scale, 1,339ms ->
    // 2,919ms), but there was no measurement of whether the increase is in index / FTS search / archive read / prefill after injection.
    // The project rule is to measure before stating a cause.
    // What is measured: the 3 stages inside this function (index build, FTS search, archive token read) and
    // the whole function. Prefill and generation after injection are already reported by the engine in the timings of the
    // /completion response (prompt_ms / predicted_ms), so they are not measured here.
    // Revert: deleting the 2 blocks marked [RIG-TIMER] (this block and the 1-line log at the end of the function)
    // restores the previous behavior. Measurement is only 4 steady_clock calls, a negligible cost.
    //=======================================================================//
    using qml_clk = std::chrono::steady_clock;
    const auto t_enter = qml_clk::now();
    auto ms_since = [](const qml_clk::time_point & a, const qml_clk::time_point & b)
    {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    double ms_index = 0.0;
    // [RIG-TIMER] the status check is also measured separately — in the 2026-08-14 measurement this single call
    // spent 904ms on a full table scan (see FTS-SCAN-FIX). It is still monitored after the fix.
    const auto t_status = qml_clk::now();
    int status = qml_sqlite_fts_status(session_id);
    const double ms_status = ms_since(t_status, qml_clk::now());
    if (status == 0)
    {
        const auto t_ix = qml_clk::now();
        if (rig_index(session_id, eot_token, detok, ud) != 0)
        {
            return 0;
        }
        ms_index = ms_since(t_ix, qml_clk::now());
        status = 1;
    }
    if (status != 1)
    {
        // [COND-SLOTS 2026-08-31] Condition search is possible even when the own session's
        // archive is empty -- the reference data is in another session (the reference session) and the condition check
        // looks at the owner scope. This return was blocking condition search (measured:
        // the checking session had an empty archive because the reference was discarded, so status=2, and all 7
        // condition-check triggers happened during visits to the reference session while loading).
        // If the query has condition bits, the flow continues so that the condition search
        // block below handles it (that block skips lexical search on success).
        bool cond_possible = false;
        if (!qml_cond_slots().empty())
        {
            int n_cq = 0;
            const std::vector<unsigned char> qc0 = qml_cond_encode(query_text, &n_cq);
            cond_possible = (n_cq > 0) && !qc0.empty();
        }
        if (!cond_possible)
        {
            return 0;   // 2 = nothing archived; -1 = no FTS5 — silently off
        }
    }
    // [HYBRID] lazy vector index — turns that are in the FTS index but have no embedding
    // are embedded in batches (512 per call) and stored. Skipped if the embedding server is unset or does not respond.
    double ms_embed_ix = 0.0;
    if (!qml_embed_url().empty() || getenv("QML_EMBED_MODEL") != NULL)
    {
        // [2026-09-02 SQL audit applied, TEST-LEDGER row pending]
        //  (1) batch 24 -> 512: qml_archive_fts is FTS5, so an equality comparison on session_id is
        //      a full scan (1.3s measured on a 1.48M-row store) -- the number of calls is the number of scans.
        //  (2) use vec_missing_txt, which reads coordinates and body in one scan: previously
        //      fts_text (itself a full scan) was called again for each document, so a 485-document session
        //      took 486 scans = around 10 minutes.
        //  (3) sessions whose backfill is finished are remembered for the process lifetime (g_vec_done) --
        //      the re-check query is also a scan, so it is not repeated every turn. Cleared when trim indexes a new
        //      document. (Caution: a path that writes to fts directly from outside the engine
        //      is not detected -- for that session the backfill is delayed until the engine restarts.)
        bool skip = false;
        {
            std::lock_guard<std::mutex> lk(g_vec_done_mtx);
            skip = g_vec_done.count(session_id) != 0;
        }
        if (!skip)
        {
            const auto t_ve = qml_clk::now();
            int n_embedded = 0;
            bool complete = false;
            for (;;)
            {
                int64_t mp[512], me[512];
                char *  mt[512];
                const int nm = qml_sqlite_vec_missing_text(session_id, 512,
                                                                  mp, me, mt);
                if (nm <= 0)
                {
                    complete = (nm == 0);
                    break;
                }
                std::vector<std::string> texts;
                texts.reserve((size_t) nm);
                for (int i = 0; i < nm; i++)
                {
                    texts.push_back(qml_vec_clean(mt[i]));
                    free(mt[i]);
                }
                std::vector<std::vector<float>> vecs;
                if (!qml_embed_texts(texts, vecs))
                {
                    break;   // embedding failed — this request proceeds with lexical search only
                }
                for (size_t i = 0; i < vecs.size(); i++)
                {
                    qml_sqlite_vec_add(session_id, mp[i], me[i],
                                               vecs[i].data(), vecs[i].size());
                    n_embedded++;
                }
                if (nm < 512)
                {
                    complete = true;
                    break;
                }
            }
            if (complete)
            {
                std::lock_guard<std::mutex> lk(g_vec_done_mtx);
                g_vec_done.insert(session_id);
            }
            ms_embed_ix = ms_since(t_ve, qml_clk::now());
            if (n_embedded > 0)
            {
                fprintf(stderr, "[qml][vec] embedded %d archived turn(s) in %.0fms\n",
                        n_embedded, ms_embed_ix);
            }
        }
    }
    // [query hygiene 2026-09-03] The "Today is <date>." context sentence at the start of a new turn
    // is excluded from the search input — date and weekday words consume the query term
    // limit (12 words), and the date token matches every summary row that carries a date, so search
    // becomes meaningless (measured: with it, temporal reasoning 3/50 on 50 questions; without it, about 32/50).
    // No effect on the context used for injection and generation — removed only when building the search query.
    std::string q_src(query_text);
    {
        const size_t tpos = q_src.find("Today is ");
        if (tpos != std::string::npos)
        {
            const size_t dot = q_src.find(". ", tpos);
            if (dot != std::string::npos && dot - tpos < 64)
            {
                q_src.erase(tpos, dot + 2 - tpos);
            }
        }
    }
    const std::string query = rig_query_text(q_src.c_str());
    if (query.empty())
    {
        return 0;
    }
    const int topk = rig_topk();
    // [FACTS 2026-09-02] context summary injection — when env QML_ARCHIVE_RECALL_FACTS=1,
    // tag='fact' rows (summary sentences produced by LLM extraction at build time) are selected by BM25 similarity
    // and the sentences themselves are injected (instead of the original turns). Basis: on the same 50 questions,
    // equal or better accuracy than original-text injection with about 1/17 of the injected tokens
    // (TEST-LEDGER row 66-1, prototype measurement — this branch is its engine integration).
    // Default 0 = existing behavior unchanged. The tokenizer relies on the detok userdata (ud) being the slot's
    // llama_context in the server wrapper (see rec_detok in the patch).
    {
        static const bool facts_mode = []() {
            const char * e = getenv("QML_ARCHIVE_RECALL_FACTS");
            return (e != NULL) && (atoi(e) != 0);
        }();
        if (facts_mode)
        {
            const std::string fq = "tag:fact AND (" + query + ")";
            const int fk = (topk < 64) ? topk : 64;
            // [dedup 2026-09-03, TEST-LEDGER row 75] Summary rows extracted from the same turn
            // take about 30% of top-k (measured), causing wrong count aggregation answers and wasted injection
            // slots. Select 3x as many, keep only the top 1 row per turn coordinate,
            // and truncate to k.
            const int fetch_n = (fk * 3 < 192) ? fk * 3 : 192;
            char *  txts_all[192];
            int64_t pos_all[192];
            const int nf_all = qml_sqlite_fts_search_text(
                session_id, fq.c_str(), fetch_n, txts_all, pos_all);
            if (nf_all <= 0)
            {
                return 0;   // no summary rows / no match — proceed without injection
            }
            char * txts[64];
            int    nf = 0;
            {
                // [dedup criterion correction 2026-09-03] Removal by turn coordinate also deleted complementary
                // facts of the same turn (date/period/item spread across rows), dropping multi-session
                // end-to-end from 17->12 (measured). Only rows whose normalized sentence is identical (lowercase,
                // alphanumeric only) are removed — no information loss (search curve identical, measured).
                std::set<std::string> seen_txt;
                for (int i = 0; i < nf_all; i++)
                {
                    std::string key;
                    for (const char * c = txts_all[i]; *c != '\0'; c++)
                    {
                        const unsigned char uc = (unsigned char) *c;
                        if (isalnum(uc))
                        {
                            key += (char) tolower(uc);
                        }
                    }
                    if (nf < fk && seen_txt.insert(key).second)
                    {
                        txts[nf++] = txts_all[i];
                    }
                    else
                    {
                        free(txts_all[i]);
                    }
                }
            }
            if (nf <= 0)
            {
                return 0;
            }
            // [FACTS format 2026-09-03] Wrap the block as a complete user turn.
            // Inserting it as bare text between turns violates the chat template, and
            // temporal reasoning questions lose heavily (measured: prototype 33/50 vs bare-text
            // injection 21/50). Original-text injection had no problem because the stored tokens already include the role
            // template — summary injection is made to match the same form.
            std::string block = "<|start_header_id|>user<|end_header_id|>\n\n"
                                "Known facts from my earlier conversations"
                                " (with dates):\n";
            for (int i = 0; i < nf; i++)
            {
                block += "- ";
                block += txts[i];
                block += "\n";
                free(txts[i]);
            }
            block += "<|eot_id|>";
            const llama_vocab * vb =
                llama_model_get_vocab(llama_get_model((llama_context *) ud));
            std::vector<llama_token> tt(block.size() + 16);
            int nt = llama_tokenize(vb, block.c_str(), (int32_t) block.size(),
                                    tt.data(), (int32_t) tt.size(),
                                    /*add_special=*/false, /*parse_special=*/true);
            if (nt <= 0)
            {
                return 0;
            }
            if (n_ctx_avail > 0 && (size_t) nt > n_ctx_avail)
            {
                nt = (int) n_ctx_avail;
            }
            int32_t * out = (int32_t *) malloc((size_t) nt * sizeof(int32_t));
            if (out == NULL)
            {
                return 0;
            }
            memcpy(out, tt.data(), (size_t) nt * sizeof(int32_t));
            *out_tokens = out;
            fprintf(stderr, "[qml][fts-recall] session '%s': facts mode,"
                    " %d summary line(s), %d token(s) injected\n",
                    session_id, nf, nt);
            return (size_t) nt;
        }
    }
    std::vector<int64_t> hit_pos((size_t) topk);
    std::vector<int64_t> hit_end((size_t) topk);
    std::vector<double>  hit_score((size_t) topk, 0.0);   // [RIG-GATE] BM25 rank
    const auto t_search = qml_clk::now();                 // [RIG-TIMER]
    // [COND-SLOTS 2026-08-31] If the condition slot scheme is configured, **the condition bit
    // check is tried first**. It is a containment check, so there is no ranking competition, and if matching
    // data exists it is the answer (unit test top-1 100%). If there is a result,
    // lexical search is not run -- mixing in lexical search pushes the answer down, from 100% to
    // 88.9% (measured). If there is no slot list or no candidate, the previous path is used.
    // Recall passes only top-1: adding candidates lowers accuracy because the model cannot
    // choose (1 rule 69.6% / 3 rules 65.2% / 5 rules 65.2%, experiment 47).
    int n_hits = 0;
    bool cond_used = false;
    std::string cond_sid;
    if (!qml_cond_slots().empty())
    {
        int n_cond_q = 0;
        // [correction] query is a string converted to FTS5 MATCH format, so `field=value`
        // does not survive (measured: 60 bit records but 0 checks). Condition encoding
        // must use the original query_text before conversion.
        const std::vector<unsigned char> qc = qml_cond_encode(query_text, &n_cond_q);
        if (n_cond_q > 0 && !qc.empty())
        {
            char csid[1][256];
            csid[0][0] = '\0';
            const int nc = qml_sqlite_cond_search(
                session_id, qc.data(), (int) qc.size(), 1,
                hit_pos.data(), hit_end.data(), csid);
            if (nc > 0)
            {
                n_hits    = nc;
                cond_used = true;
                hit_score[0] = -1.0;      // fixed value since there is no ranking
                // The candidate may come from another session (the reference session), so
                // remember that session and read the tokens from it.
                cond_sid.assign(csid[0]);
                fprintf(stderr, "[qml][cond] '%s': %d condition bit(s) in query"
                        " -> matched turn pos=%lld in '%s'\n",
                        session_id, n_cond_q, (long long) hit_pos[0], csid[0]);
            }
        }
    }
    if (!cond_used)
    {
        n_hits = qml_sqlite_fts_search(session_id, query.c_str(),
                                               topk, hit_pos.data(), hit_end.data(),
                                               hit_score.data());
    }
    const double ms_search = ms_since(t_search, qml_clk::now());   // [RIG-TIMER]
    // [PEER-REACH 2026-08-30] Even if lexical search in the own session returns 0 hits, continue when the query
    // has an identifier and cross-session lookup is enabled.
    // Previously it returned here and never reached the other-session lookup — this was the cause of 0 lookup
    // triggers in the multi-step approval measurement (measured). When the own session has no evidence
    // is exactly when the session of an agent with another role needs to be checked.
    {
        const char * pe = getenv("QML_ENTITY_PEER");
        const bool peer_possible = (pe != NULL) && (atoi(pe) != 0) &&
                                   !qml_extract_entities(query_text).empty();
        if (n_hits <= 0 && !peer_possible)
        {
            return 0;
        }
    }
    // [v1.1 RIG structural selection]
    // greedy in FTS rank order (best match first) under a DYNAMIC budget
    // min(env budget, n_ctx_avail); a candidate turn larger than beta*n_ctx_avail is
    // bypassed wholesale (turn alignment forbids partial injection). The
    // SELECTED turns are then re-ordered chronologically for injection.
    const size_t budget = std::min(rig_budget(),
                                   n_ctx_avail > 0 ? n_ctx_avail : rig_budget());
    const size_t turn_cap = n_ctx_avail > 0 ? n_ctx_avail / 2 : budget;   // beta = 0.5
    int64_t spread_w = 0;
    int     spread_k = 2;
    rig_spread(&spread_w, &spread_k);
    // [RIG-GATE] conf gate preparation: the sim reference (top rank) and the number of valid words in the question.
    // If conf_threshold is 0 (default), the check below always passes, so behavior is the same as before.
    const double conf   = rig_conf();
    const double best   = (n_hits > 0 && hit_score[0] != 0.0) ? hit_score[0] : -1.0;
    const int    n_eff  = rig_n_eff(query_text);
    const int    n_min  = rig_n_min();
    const double info   = n_eff >= n_min ? 1.0 : (double) n_eff / (double) n_min;
    if (conf > 0.0)
    {
        fprintf(stderr, "[qml][rig] conf=%.2f n_eff=%d N_min=%d info=%.2f\n",
                conf, n_eff, n_min, info);
    }
    struct qml_rec_pick { int64_t pos; int64_t end; std::string sid; };
    std::vector<qml_rec_pick> picks;
    size_t picked_tokens = 0;
    // [ENTITY-INDEX 2026-08-29] If the query has an identifier, the condition lookup results are inserted
    // first — they skip ranking competition, so they are not pushed out no matter how many similar documents
    // exist (direct fix for large-scale recurrence recall 3/53). For natural-language
    // questions without an identifier this block is empty and the previous path is used.
    // Disabled with env QML_ENTITY_LOOKUP=0.
    {
        static const bool ent_on = []() {
            const char * e = getenv("QML_ENTITY_LOOKUP");
            return (e == NULL) || (atoi(e) != 0);
        }();
        if (ent_on)
        {
            for (const auto & ent : qml_extract_entities(query_text))
            {
                int64_t ep[4], ee[4];
                const int ne = qml_sqlite_entity_find(
                    session_id, ent.c_str(), 4, ep, ee);
                for (int i = 0; i < ne; i++)
                {
                    const size_t len = (size_t) (ee[i] - ep[i]);
                    if (len == 0 || len > turn_cap ||
                        picked_tokens + len > budget)
                    {
                        continue;
                    }
                    // [dedup check correction 2026-08-31] Comparing pos only mistakes documents
                    // at the same position in different sessions for the same one. Archive
                    // positions are independent per session, so the (session, pos) pair is used.
                    bool dup = false;
                    for (const auto & p : picks)
                    {
                        dup = dup || (p.pos == ep[i] && p.sid == session_id);
                    }
                    if (dup)
                    {
                        continue;
                    }
                    picks.push_back({ ep[i], ee[i], session_id });
                    picked_tokens += len;
                    fprintf(stderr, "[qml][entity] '%s' -> pos=%lld len=%zu\n",
                            ent.c_str(), (long long) ep[i], len);
                }
                // [ENTITY-PEER] If not found in the own session and peer lookup is enabled,
                // also look up other agent sessions of the same owner ("owner::" prefix).
                // This path is for cases where the answer exists only in another agent's observations; sessions
                // of a different owner are not crossed (identity isolation is kept).
                // Enabled with env QML_ENTITY_PEER=1 (off by default — the isolation default takes priority).
                static const bool peer_on = []() {
                    const char * e = getenv("QML_ENTITY_PEER");
                    return (e != NULL) && (atoi(e) != 0);
                }();
                // [PEER trigger condition correction 2026-08-29] If set to "when not found in the own session",
                // it never triggers — the query text itself (e.g. "DIAGNOSE
                // pattern=P-107") is later moved into the archive and indexed under the same identifier,
                // so the own-session lookup always returns its own question
                // (measured: 214 P- entities in the analyst session, 0 peer triggers).
                // Correction: peer session observations are also recalled even when the own session
                // has results. Budget and dedup checks below still apply as before.
                if (peer_on)
                {
                    const char * sep = strstr(session_id, "::");
                    if (sep != NULL)
                    {
                        char owner[128];
                        const size_t olen = (size_t) (sep - session_id) + 2;
                        snprintf(owner, sizeof(owner), "%.*s",
                                 (int) (olen < sizeof(owner) ? olen : sizeof(owner) - 1),
                                 session_id);
                        char psid[4][128];
                        int64_t pp[4], pe[4];
                        const int np = qml_sqlite_entity_find_peers(
                            owner, ent.c_str(), 4, psid, pp, pe);
                        for (int i = 0; i < np; i++)
                        {
                            const size_t len = (size_t) (pe[i] - pp[i]);
                            if (len == 0 || len > turn_cap ||
                                picked_tokens + len > budget)
                            {
                                continue;
                            }
                            // [duplicate prevention 2026-08-31] The peer path had no duplicate
                            // check, so the same document was injected twice
                            // (measured: picked 2 turn(s) pos=[0,0], 367 -> 734
                            // token). Own-session results and peer results often point to the same
                            // document.
                            bool pdup = false;
                            for (const auto & p : picks)
                            {
                                pdup = pdup || (p.pos == pp[i] &&
                                                p.sid == std::string(psid[i]));
                            }
                            if (pdup)
                            {
                                continue;
                            }
                            picks.push_back({ pp[i], pe[i], std::string(psid[i]) });
                            picked_tokens += len;
                            fprintf(stderr, "[qml][entity-peer] '%s' from '%s'"
                                    " pos=%lld len=%zu\n", ent.c_str(), psid[i],
                                    (long long) pp[i], len);
                        }
                    }
                }
            }
        }
    }
    // [RIG-SAMEQ 2026-10-22] the question asked before, without its answer, counts
    // for nothing (rig_same_question). Rows of the asking session only: a
    // condition match from a reference session is a rule, not a question
    auto same_question = [&](int64_t pos) {
        char * t = qml_sqlite_fts_text(session_id, pos);
        const bool sq = rig_same_question(t, q_src.c_str());
        free(t);
        if (sq)
        {
            fprintf(stderr, "[qml][rig] skip pos=%lld: the question asked before, without its answer\n",
                    (long long) pos);
        }
        return sq;
    };
    for (int i = 0; i < n_hits; i++)   // rank order = score-descending greedy
    {
        const size_t turn_len = (size_t) (hit_end[i] - hit_pos[i]);
        if (turn_len > turn_cap)
        {
            fprintf(stderr, "[qml][fts-recall] bypass oversized turn pos=%lld len=%zu (cap %zu)\n",
                    (long long) hit_pos[i], turn_len, turn_cap);
            continue;                   // size_ok = 0 -> bypass, never truncate
        }
        if (!cond_used && same_question(hit_pos[i]))
        {
            continue;
        }
        // [RIG-GATE] injection check: rig_score = sim x info
        // sim is relative within the same result set (top = 1.0). Passes if conf=0.
        // [CORE-MOVE 2026-08-23] the check itself is core's qml_rig_gate
        // (rig_score = sim x info >= conf; always passes if conf=0 — same behavior)
        // [COND-GATE 2026-09-01] Candidates selected by the condition bit check skip the two
        // gates below. Both gates measure **the evidence strength of lexical search**,
        // while the condition check does not look at words but verifies containment of conditions,
        // so it is already stronger evidence (if containment holds, all conditions of that data
        // are satisfied). Measured (2026-09-01): condition search found the correct rule in
        // 10 of 13 cases, but all were discarded at this gate, so accuracy tied with lexical
        // search (12/23) -- the log line "no-inject ... abs-match=0
        // single-hit (defect #11 floor)" is that point. In addition, the lexical text that gate
        // looks up is based on the searching session, so for a candidate in another session (the reference session)
        // it is 0 by definition.
        if (!cond_used && conf > 0.0 &&
            !qml_rig_gate(hit_score[i], best, n_eff, n_min, conf))
        {
            const double sim = (hit_score[i] != 0.0 && best != 0.0)
                             ? (hit_score[i] / best) : 0.0;   // for log display
            fprintf(stderr, "[qml][rig] no-inject pos=%lld sim=%.3f info=%.2f"
                    " rig=%.3f < conf=%.2f\n",
                    (long long) hit_pos[i], sim, info, sim * info, conf);
            continue;                   // evidence is weak -> do not inject
        }
        // [RIG-ABS 2026-08-26, made conditional 2026-08-27] minimum match —
        // applied only to a degenerate set with a single search result. With 1 candidate, the sim of the
        // conf gate above is 1.0 by definition and has no effect (TEST-LEDGER
        // defect #11), so an actual match of query words is required; multi-hit sets
        // are discriminated by relative conf, so the floor is not applied (applying it
        // unconditionally blocked indirect correct candidates on the 21 wx2k questions, -1 loss measured).
        if (!cond_used && n_hits == 1)
        {
            char * cand_text = qml_sqlite_fts_text(session_id, hit_pos[i]);
            const int n_match = qml_rig_match_count(query_text, cand_text);
            free(cand_text);
            if (n_match < QML_RIG_MIN_MATCH_SINGLE_HIT)
            {
                fprintf(stderr, "[qml][rig] no-inject pos=%lld abs-match=%d"
                        " < min=%d, single-hit (defect #11 floor)\n",
                        (long long) hit_pos[i], n_match,
                        QML_RIG_MIN_MATCH_SINGLE_HIT);
                continue;               // degenerate set + insufficient absolute evidence -> no injection
            }
        }
        if (spread_w > 0)               // positional spread constraint (off by default)
        {
            int near = 0;
            for (size_t j = 0; j < picks.size(); j++)
            {
                const int64_t d = picks[j].pos > hit_pos[i]
                                ? picks[j].pos - hit_pos[i] : hit_pos[i] - picks[j].pos;
                if (d <= spread_w)
                {
                    near++;
                }
            }
            if (near >= spread_k)
            {
                fprintf(stderr, "[qml][fts-recall] spread skip pos=%lld (%d already within %lld)\n",
                        (long long) hit_pos[i], near, (long long) spread_w);
                continue;               // let a distant region use this slot instead
            }
        }
        if (picked_tokens + turn_len > budget)
        {
            continue;
        }
        // [duplicate prevention 2026-08-31] The lexical search path also had no duplicate check, so a document
        // already inserted by entity lookup (own session or other session) was inserted again
        // (measured: picked 2 turn(s) pos=[0,0], 367 -> 734 token). The three paths accumulate into
        // the same picks, so the check is done in one place on the (session, pos) pair.
        {
            bool fdup = false;
            for (const auto & p : picks)
            {
                fdup = fdup || (p.pos == hit_pos[i] && p.sid == session_id);
            }
            if (fdup)
            {
                continue;
            }
        }
        picks.push_back({hit_pos[i], hit_end[i],
                         (cond_used && !cond_sid.empty()) ? cond_sid : std::string(session_id)});
        picked_tokens += turn_len;
    }
    // [HYBRID] Add the semantic top QML_VEC_TOPK candidates as a union — covers paraphrased
    // questions that lexical search misses (measured: the two methods fail on different questions, and
    // only the union hits every question, TEST-LEDGER A-2 item 22). Lexical picks use the budget
    // first and semantic picks only get the remaining budget (verified path first).
    // Duplicates (same pos) are excluded; the in-context duplicate guard is applied in the materialization step below
    // to semantic picks as well.
    if (!qml_embed_url().empty() || getenv("QML_EMBED_MODEL") != NULL)
    {
        std::vector<std::string> qtxt(1, qml_vec_clean(query_text));
        std::vector<std::vector<float>> qv;
        if (qml_embed_texts(qtxt, qv) && !qv.empty())
        {
            fprintf(stderr, "[qml][vec][dbg] query='%.100s' qdim=%zu\n",
                    qtxt[0].c_str(), qv[0].size());
            int64_t vp[QML_VEC_TOPK], ve[QML_VEC_TOPK];
            double  vs[QML_VEC_TOPK];
            const int nv = qml_sqlite_vec_search(
                session_id, qv[0].data(), qv[0].size(), QML_VEC_TOPK, vp, ve, vs);
            for (int i = 0; i < nv; i++)
            {
                fprintf(stderr, "[qml][vec][dbg] raw#%d pos=%lld cos=%.3f\n",
                        i, (long long) vp[i], vs[i]);
            }
            // Only the top-1 is an injection candidate — the basis is in the [HYBRID defect fix] comment above.
            if (nv > 0 && vs[0] >= QML_VEC_MIN_COS)
            {
                bool dup = false;
                for (const qml_rec_pick & pk : picks)
                {
                    if (pk.pos == vp[0])
                    {
                        dup = true;
                        break;
                    }
                }
                const size_t turn_len = (size_t) (ve[0] - vp[0]);
                if (!dup && same_question(vp[0]))
                {
                    // [RIG-SAMEQ] logged by same_question; nothing added
                }
                else if (dup)
                {
                    fprintf(stderr, "[qml][vec] top1 pos=%lld cos=%.3f already a"
                            " lexical pick -- no semantic add\n",
                            (long long) vp[0], vs[0]);
                }
                else if (turn_len <= turn_cap && picked_tokens + turn_len <= budget)
                {
                    picks.push_back({vp[0], ve[0], session_id});
                    picked_tokens += turn_len;
                    fprintf(stderr, "[qml][vec] semantic add pos=%lld cos=%.3f len=%zu\n",
                            (long long) vp[0], vs[0], turn_len);
                }
            }
            else if (nv > 0)
            {
                fprintf(stderr, "[qml][vec] top1 pos=%lld cos=%.3f < min=%.2f"
                        " -- no semantic add\n",
                        (long long) vp[0], vs[0], QML_VEC_MIN_COS);
            }
        }
    }
    // chronological order for the injected block, except the best pick
    // [RIG-ORDER 2026-10-22] (picks[0]: the first one chosen -- entity lookup,
    // then word rank), which goes last, next to the question. In time order an
    // unrelated later row stood next to the question and Llama-3.2-3B answered
    // "Not available." with the fact injected; with only the order changed it
    // answered the fact
    const qml_rec_pick best_pick = picks.empty() ? qml_rec_pick{ 0, 0, std::string() } : picks[0];
    for (size_t i = 0; i < picks.size(); i++)
    {
        for (size_t j = i + 1; j < picks.size(); j++)
        {
            if (picks[j].pos < picks[i].pos)
            {
                std::swap(picks[i], picks[j]);
            }
        }
    }
    for (size_t i = 0; i + 1 < picks.size(); i++)
    {
        if (picks[i].pos == best_pick.pos && picks[i].sid == best_pick.sid)
        {
            std::rotate(picks.begin() + (long) i, picks.begin() + (long) i + 1, picks.end());
            break;
        }
    }
    // [SPAN withdrawn 2026-09-03] pick neighbor expansion (QML_ARCHIVE_RECALL_SPAN) was removed after
    // measurement rejected it — injection grew to around 20,000 tokens per question and selection precision
    // collapsed (TEST-LEDGER row 66-1: original-text top-24 injection reached equal accuracy with 6,632 tokens,
    // context summary reached it with 396 tokens — expansion fell short of both).
    std::vector<int32_t> recalled;
    std::string pos_list;
    // [RIG-INCTX 2026-08-27] in-context duplicate guard preparation — read the live log tail
    // once, and run a substring check per candidate in the materialization step below.
    // (default behavior after the verification period — see the [RIG-DEFAULTS] comment)
    int32_t * tail_ids = NULL;
    size_t    tail_n   = 0;
    if (!picks.empty())
    {
        qml_sqlite_get_tail(session_id, QML_RIG_INCONTEXT_TAIL,
                                      &tail_ids, &tail_n);
    }
    const auto t_fetch = qml_clk::now();      // [RIG-TIMER] archive token read stage
    for (const qml_rec_pick & pk : picks)
    {
        int32_t * rt = NULL;
        size_t    rn = 0;
        if (qml_sqlite_archive_range(pk.sid.c_str(), pk.pos, pk.end, &rt, &rn) != 0)
        {
            continue;
        }
        if (tail_ids != NULL && qml_rig_contains(tail_ids, tail_n, rt, rn))
        {
            fprintf(stderr, "[qml][rig] skip in-context pos=%lld len=%zu"
                    " (already present in live tail)\n",
                    (long long) pk.pos, rn);
            free(rt);
            continue;                   // the same content is already in context -> prevent duplicate injection
        }
        recalled.insert(recalled.end(), rt, rt + rn);
        free(rt);
        pos_list += (pos_list.empty() ? "" : ",") + std::to_string(pk.pos);
    }
    free(tail_ids);
    const double ms_fetch = ms_since(t_fetch, qml_clk::now());      // [RIG-TIMER]
    if (recalled.empty())
    {
        return 0;
    }
    int32_t * out = (int32_t *) malloc(recalled.size() * sizeof(int32_t));
    if (out == NULL)
    {
        return 0;
    }
    memcpy(out, recalled.data(), recalled.size() * sizeof(int32_t));
    *out_tokens = out;
    // [v1.1] injected turn positions logged -> retrieval recall@k can be
    // computed against ground-truth answer ranges without a reader model
    fprintf(stderr, "[qml][fts-recall] session '%s': %d hit(s), picked %zu turn(s) pos=[%s], %zu token(s), n_ctx_avail=%zu\n",
            session_id, n_hits, picks.size(), pos_list.c_str(), recalled.size(), n_ctx_avail);
    // [RIG-TIMER] per-stage elapsed time. index=0.0 means warm (index reused).
    fprintf(stderr, "[qml][timer] status=%.1fms index=%.1fms search=%.1fms fetch=%.1fms"
            " hook_total=%.1fms injected=%zu token(s)\n",
            ms_status, ms_index, ms_search, ms_fetch,
            ms_since(t_enter, qml_clk::now()), recalled.size());
    return recalled.size();
}
//===========================================================================//
// [FTS-RECALL] END
//===========================================================================//
