// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
//
// unit_sqlite_save_tx.c -- tests for the one-transaction session save and the
// busy timeout in storage/sqlite/qml_sqlite_store.c (2026-10-02).
// No GPU, engine or LMDB needed.
//
//   cc -I include -I storage/sqlite tests/unit/unit_sqlite_save_tx.c
//      -L <build dir> -lqml_session_store -lsqlite3 -lpthread -o /tmp/t   (one line)
//   LD_LIBRARY_PATH=<build dir> /tmp/t [directory for the test database]
//
// What is checked:
//   1. a save writes the session row and the token log together
//   2. a save waits for another connection's write lock (held 1.5 s) and succeeds
//   3. a lock held longer than the busy timeout (7 s > 5 s) fails the save, the
//      previous row and log stay together (no row pointing at an empty log), and
//      the next save on the same connection works (no transaction left open)
//   4. a context-shift rewrite archives the old log inside the same transaction
//   5. n_tokens == 0 leaves or clears the log as requested
//   6. qml_session_save_tokens goes through the same path

#include <signal.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "qml_router.h"
#include "qml_sqlite_store.h"

static int g_fail = 0;

#define CHECK(cond, name)                                        \
    do {                                                         \
        if (cond) { printf("ok   %s\n", (name)); }               \
        else      { printf("FAIL %s\n", (name)); g_fail = 1; }   \
    } while (0)

static char g_path[256];

//------------------------------------------------------------
//Func : now_s
//------------------------------------------------------------
//out: monotonic time in seconds
static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + ts.tv_nsec / 1e9;
}

//------------------------------------------------------------
//Func : count_q
//------------------------------------------------------------
//in: sql -- one-value query with one text parameter; sid -- that parameter
//out: the integer result read on a separate connection, -1 on error
static long count_q(const char * sql, const char * sid)
{
    sqlite3 * db = NULL;
    long      v  = -1;
    if (sqlite3_open(g_path, &db) == SQLITE_OK)
    {
        sqlite3_stmt * st = NULL;
        if (sqlite3_prepare_v2(db, sql, -1, &st, NULL) == SQLITE_OK)
        {
            sqlite3_bind_text(st, 1, sid, -1, SQLITE_TRANSIENT);
            if (sqlite3_step(st) == SQLITE_ROW)
            {
                v = (long) sqlite3_column_int64(st, 0);
            }
            sqlite3_finalize(st);
        }
    }
    sqlite3_close(db);
    return v;
}

static long log_rows(const char * sid)
{
    return count_q("SELECT COUNT(*) FROM qml_token_log WHERE session_id=?", sid);
}

static long row_n_tokens(const char * sid)
{
    return count_q("SELECT n_tokens FROM qml_session WHERE session_id=?", sid);
}

static long archive_rows(const char * sid)
{
    return count_q("SELECT COUNT(*) FROM qml_token_archive WHERE session_id=?", sid);
}

//------------------------------------------------------------
//Func : hold_lock
//------------------------------------------------------------
//in: seconds -- how long a second process holds BEGIN IMMEDIATE on the file
//out: child pid; returns once the child holds the lock
static pid_t hold_lock(double seconds)
{
    int fd[2];
    if (pipe(fd) != 0)
    {
        return -1;
    }
    pid_t pid = fork();
    if (pid == 0)
    {
        close(fd[0]);
        sqlite3 * db = NULL;
        sqlite3_open(g_path, &db);
        const int rc = sqlite3_exec(db, "BEGIN IMMEDIATE;", NULL, NULL, NULL);
        char c = (rc == SQLITE_OK) ? 'y' : 'n';
        if (write(fd[1], &c, 1) != 1)
        {
            _exit(2);
        }
        usleep((useconds_t) (seconds * 1e6));
        sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
        sqlite3_close(db);
        _exit(0);
    }
    close(fd[1]);
    char c = 0;
    if (read(fd[0], &c, 1) != 1 || c != 'y')
    {
        fprintf(stderr, "helper could not take the lock\n");
    }
    close(fd[0]);
    return pid;
}

//------------------------------------------------------------
//Func : base_for
//------------------------------------------------------------
//in: sid, n -- session id and token count
//out: base row for a base-only save
static qml_session_meta_t base_for(const char * sid, size_t n, const int32_t * toks)
{
    qml_session_meta_t b;
    memset(&b, 0, sizeof(b));
    strncpy(b.user_id, "unit", sizeof(b.user_id) - 1);
    strncpy(b.session_id, sid, sizeof(b.session_id) - 1);
    b.engine_type = QML_ENGINE_LLAMA_CPP;
    b.n_tokens    = (int64_t) n;
    b.last_token  = n ? toks[n - 1] : -1;
    b.last_pos    = (int32_t) n - 1;
    return b;
}

static int save(const char * sid, const int32_t * toks, size_t n, int clear)
{
    qml_session_meta_t b = base_for(sid, n, toks);
    return qml_sqlite_save_session(sid, NULL, &b, toks, n, 0, clear);
}

int main(int argc, char ** argv)
{
    // optional argv[1]: directory for the test database (e.g. the NFS volume the
    // service store lives on, where locking can behave differently from /tmp)
    snprintf(g_path, sizeof(g_path), "%s/unit_save_tx_%d.db",
             (argc > 1) ? argv[1] : "/tmp", (int) getpid());
    if (qml_sqlite_open(g_path) != 0)
    {
        printf("FAIL open %s\n", g_path);
        return 1;
    }

    int32_t t[32];
    for (int i = 0; i < 32; i++)
    {
        t[i] = 1000 + i;
    }

    // 1. basic save
    CHECK(save("s1", t, 10, 1) == 0, "save 10 tokens");
    CHECK(log_rows("s1") == 10 && row_n_tokens("s1") == 10, "row and log both 10");

    // 2. lock held shorter than the busy timeout -> waits, then succeeds
    pid_t p = hold_lock(1.5);
    double t0 = now_s();
    int rc = save("s1", t, 12, 1);
    double waited = now_s() - t0;
    waitpid(p, NULL, 0);
    CHECK(rc == 0, "save succeeds while another writer holds the lock for 1.5 s");
    CHECK(waited > 1.2 && waited < 4.5, "save waited for the lock (about 1.5 s)");
    CHECK(log_rows("s1") == 12 && row_n_tokens("s1") == 12, "row and log both 12");

    // 3. lock held longer than the busy timeout -> fails, nothing half-written
    p = hold_lock(7.0);
    t0 = now_s();
    rc = save("s1", t, 15, 1);
    waited = now_s() - t0;
    CHECK(rc == -1, "save fails when the lock is held 7 s (busy timeout 5 s)");
    CHECK(waited > 4.5 && waited < 6.9, "save gave up after about 5 s");
    waitpid(p, NULL, 0);
    CHECK(log_rows("s1") == 12 && row_n_tokens("s1") == 12,
          "after the failure, row and log are still both 12");
    CHECK(save("s1", t, 15, 1) == 0, "next save on the same connection works");
    CHECK(log_rows("s1") == 15 && row_n_tokens("s1") == 15, "row and log both 15");

    // 4. context-shift rewrite: stored log is not a prefix -> archived in the same save
    int32_t a[10], b[8];
    for (int i = 0; i < 10; i++)
    {
        a[i] = 100 + i;
    }
    b[0] = 100; b[1] = 101;
    for (int i = 2; i < 7; i++)
    {
        b[i] = 105 + (i - 2);
    }
    b[7] = 200;
    CHECK(save("s2", a, 10, 1) == 0, "save 10 tokens for the shift case");
    CHECK(save("s2", a, 10, 1) == 0 && archive_rows("s2") == 0, "same window archives nothing");
    CHECK(save("s2", b, 8, 1) == 0, "save the rewritten window");
    CHECK(archive_rows("s2") == 10 && log_rows("s2") == 8,
          "old 10 tokens archived, log holds the new 8");

    // 5. n_tokens == 0
    CHECK(save("s1", t, 0, 0) == 0 && log_rows("s1") == 15, "empty save with clear=0 keeps the log");
    CHECK(save("s1", t, 0, 1) == 0 && log_rows("s1") == 0, "empty save with clear=1 clears the log");

    // 6. public tokens-only save goes through the same transaction
    CHECK(qml_session_save_tokens("s3", "unit", "", 0, t, 20) == 0,
          "qml_session_save_tokens saves");
    CHECK(log_rows("s3") == 20 && row_n_tokens("s3") == 20, "row and log both 20");

    qml_sqlite_close();
    char side[300];
    unlink(g_path);
    snprintf(side, sizeof(side), "%s-wal", g_path);
    unlink(side);
    snprintf(side, sizeof(side), "%s-shm", g_path);
    unlink(side);
    printf(g_fail ? "SOME TESTS FAILED\n" : "ALL PASSED\n");
    return g_fail;
}
