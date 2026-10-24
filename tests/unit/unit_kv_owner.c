// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
//
// unit_kv_owner.c -- block ownership (qml_kv_owner) and qml_store_drop_kv, which a
// window trim calls: the blocks only one session stored go (records and LMDB values),
// a block another session also recorded stays until its last owner drops it, and a
// session that stored nothing drops nothing.
// No GPU or engine needed.
//
//   cc -I include -I storage/sqlite -I storage/lmdb tests/unit/unit_kv_owner.c
//      -L <build dir> -lqml_session_store -llmdb -lsqlite3 -lcrypto -lpthread -o /tmp/t   (one line)
//   LD_LIBRARY_PATH=<build dir> /tmp/t [work dir]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "qml_router.h"
#include "qml_sqlite_store.h"
#include "qml_lmdb_store.h"

static int g_fail = 0;

#define CHECK(cond, name)                                        \
    do {                                                         \
        if (cond) { printf("ok   %s\n", (name)); }               \
        else      { printf("FAIL %s\n", (name)); g_fail = 1; }   \
    } while (0)

//------------------------------------------------------------
//Func : present
// 1 when key has an LMDB value
//------------------------------------------------------------
static int present(const char * key)
{
    const void *     data = NULL;
    size_t           size = 0;
    qml_lmdb_txn_t * h    = NULL;
    if (qml_lmdb_get(key, &data, &size, &h) != 0)
    {
        return 0;
    }
    qml_lmdb_get_done(h);
    return 1;
}

int main(int argc, char ** argv)
{
    char dir[256], db[300], lmdb[300];
    snprintf(dir, sizeof(dir), "%s/unit_kv_owner_%d", argc > 1 ? argv[1] : "/tmp", (int) getpid());
    mkdir(dir, 0700);
    snprintf(db, sizeof(db), "%s/qml.db", dir);
    snprintf(lmdb, sizeof(lmdb), "%s/lmdb", dir);
    mkdir(lmdb, 0700);
    setenv("QML_LMDB_MAP_SIZE_MB", "64", 1);
    if (qml_store_init(db, lmdb) != 0)
    {
        printf("FAIL storage init\n");
        return 1;
    }

    const char * a_keys[] = { "blk-a1", "blk-a2", "blk-shared" };
    const char * b_keys[] = { "blk-shared", "blk-b1" };
    const char * all[]    = { "blk-a1", "blk-a2", "blk-shared", "blk-b1" };
    const char   payload[64] = { 7 };
    for (int i = 0; i < 4; i++)
    {
        qml_lmdb_put(all[i], payload, sizeof payload);
        qml_sqlite_block_touch(all[i], (int64_t) sizeof payload);
    }
    CHECK(qml_sqlite_owner_add("user::a", a_keys, 3) == 0, "owner_add session a");
    CHECK(qml_sqlite_owner_add("user::b", b_keys, 2) == 0, "owner_add session b");
    CHECK(qml_sqlite_owner_add("user::a", a_keys, 3) == 0, "owner_add again is a no-op");

    long n = qml_store_drop_kv("user::a");
    CHECK(n == 2, "drop a deletes its two own blocks");
    CHECK(!present("blk-a1") && !present("blk-a2"), "a's own values are gone");
    CHECK(present("blk-shared"), "shared block stays while b owns it");
    CHECK(present("blk-b1"), "b's block stays");

    int64_t total = 0;
    qml_sqlite_block_total_bytes(&total);
    CHECK(total == 2 * (int64_t) sizeof payload, "accounting rows of a's blocks are gone");

    CHECK(qml_store_drop_kv("user::a") == 0, "second drop of a deletes nothing");
    CHECK(qml_store_drop_kv("user::none") == 0, "session without blocks deletes nothing");

    n = qml_store_drop_kv("user::b");
    CHECK(n == 2, "drop b deletes the shared block and its own");
    CHECK(!present("blk-shared") && !present("blk-b1"), "b's values are gone");
    qml_sqlite_block_total_bytes(&total);
    CHECK(total == 0, "no accounting rows left");

    qml_store_close();
    printf("%s\n", g_fail ? "SOME FAILED" : "ALL PASSED");
    printf("test files left in %s\n", dir);
    return g_fail;
}
