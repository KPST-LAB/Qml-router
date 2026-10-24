// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
//
// unit_blob_fallback.c -- when the LMDB blob write fails, the full save must
// still store the session row and the token log (2026-10-02, BLOB-FALLBACK).
// No GPU or engine needed. The LMDB map is set to 1 MiB so a 4 MiB blob fails
// with MDB_MAP_FULL, which is what happened on the H200 at 99.7% map use.
//
//   cc -I include -I storage/sqlite -I storage/lmdb tests/unit/unit_blob_fallback.c
//      -L <build dir> -lqml_session_store -llmdb -lsqlite3 -lcrypto -lpthread -o /tmp/t   (one line)
//   LD_LIBRARY_PATH=<build dir> /tmp/t
//
// Expected stderr: one "[qml][lmdb] put_blob ... MDB_MAP_FULL" line and one
// "[qml][save] blob write failed ..." line.

#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "qml_router.h"
#include "qml_sqlite_store.h"

static int g_fail = 0;

#define CHECK(cond, name)                                        \
    do {                                                         \
        if (cond) { printf("ok   %s\n", (name)); }               \
        else      { printf("FAIL %s\n", (name)); g_fail = 1; }   \
    } while (0)

static char g_db[256];

static long count_q(const char * sql, const char * sid)
{
    sqlite3 * db = NULL;
    long      v  = -1;
    if (sqlite3_open(g_db, &db) == SQLITE_OK)
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

int main(void)
{
    char dir[200], lmdb[256];
    snprintf(dir, sizeof(dir), "/tmp/unit_blob_fallback_%d", (int) getpid());
    mkdir(dir, 0700);
    snprintf(g_db, sizeof(g_db), "%s/qml.db", dir);
    snprintf(lmdb, sizeof(lmdb), "%s/lmdb", dir);
    mkdir(lmdb, 0700);
    setenv("QML_LMDB_MAP_SIZE_MB", "1", 1);   // existing switch: 1 MiB map

    if (qml_store_init(g_db, lmdb) != 0)
    {
        printf("FAIL storage init\n");
        return 1;
    }

    int32_t toks[40];
    for (int i = 0; i < 40; i++)
    {
        toks[i] = 2000 + i;
    }
    const size_t small = 4096, big = 4u << 20;
    unsigned char * blob = calloc(1, big);

    // 1. a blob that fits: normal full save
    CHECK(qml_session_save("f1", "unit", "m", 0, blob, small, toks, 20) == 0,
          "full save with a 4 KiB blob succeeds");
    CHECK(count_q("SELECT COUNT(*) FROM qml_token_log WHERE session_id=?", "f1") == 20 &&
          count_q("SELECT n_tokens FROM qml_session WHERE session_id=?", "f1") == 20,
          "row and log both 20");

    // 2. a blob that cannot fit: the save must still store row + log
    CHECK(qml_session_save("f1", "unit", "m", 0, blob, big, toks, 40) == 0,
          "full save with a 4 MiB blob in a 1 MiB map still returns 0 (fallback)");
    CHECK(count_q("SELECT COUNT(*) FROM qml_token_log WHERE session_id=?", "f1") == 40 &&
          count_q("SELECT n_tokens FROM qml_session WHERE session_id=?", "f1") == 40,
          "row and log both 40 after the blob write failed");
    CHECK(count_q("SELECT state_size FROM qml_llama_session_ext WHERE session_id=?", "f1") == (long) small,
          "ext row still describes the earlier 4 KiB blob (kept as a prefix)");

    // 3. a brand-new session whose first blob cannot fit: row + log, no ext row
    CHECK(qml_session_save("f2", "unit", "m", 0, blob, big, toks, 30) == 0,
          "first save of a new session with an oversized blob returns 0");
    CHECK(count_q("SELECT COUNT(*) FROM qml_token_log WHERE session_id=?", "f2") == 30 &&
          count_q("SELECT COUNT(*) FROM qml_llama_session_ext WHERE session_id=?", "f2") == 0,
          "new session has its 30-token log and no ext row (replays from the log)");

    free(blob);
    printf(g_fail ? "SOME TESTS FAILED\n" : "ALL PASSED\n");
    printf("test files left in %s\n", dir);
    return g_fail;
}
