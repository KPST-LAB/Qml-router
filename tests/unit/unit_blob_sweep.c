// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
//
// unit_blob_sweep.c -- store functions behind the SWEEP_BLOBS control command
// (2026-10-02, BLOB-SWEEP): qml_lmdb_list_blobs lists only blob headers, and
// qml_sqlite_uses_blob tells live blobs from orphans.
// No GPU or engine needed.
//
//   cc -I include -I storage/sqlite -I storage/lmdb tests/unit/unit_blob_sweep.c
//      -L <build dir> -lqml_session_store -llmdb -lsqlite3 -lcrypto -lpthread -o /tmp/t   (one line)
//   LD_LIBRARY_PATH=<build dir> /tmp/t

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

static int has_key(char ** keys, size_t n, const char * k)
{
    for (size_t i = 0; i < n; ++i)
    {
        if (strcmp(keys[i], k) == 0)
        {
            return 1;
        }
    }
    return 0;
}

int main(void)
{
    char dir[200], db[256], lmdb[256];
    snprintf(dir, sizeof(dir), "/tmp/unit_blob_sweep_%d", (int) getpid());
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

    int32_t toks[16];
    for (int i = 0; i < 16; i++)
    {
        toks[i] = 3000 + i;
    }
    const size_t sz = (9u << 20);   // 9 MiB -> 3 chunks of 4 MiB each
    unsigned char * blob = calloc(1, sz);

    CHECK(qml_session_save("live", "unit", "m", 0, blob, sz, toks, 16) == 0, "save session 'live'");
    CHECK(qml_session_save("gone", "unit", "m", 0, blob, sz, toks, 16) == 0, "save session 'gone'");
    CHECK(qml_sqlite_delete_session("gone") == 0, "delete only the SQLite rows of 'gone' (the old delete path)");

    char ** keys = NULL; uint64_t * sizes = NULL; size_t n = 0;
    CHECK(qml_lmdb_list_blobs(&keys, &sizes, &n) == 0, "list blobs");
    CHECK(n == 2 && has_key(keys, n, "live") && has_key(keys, n, "gone"),
          "exactly the two blob headers are listed, no chunk entries");
    CHECK(n == 2 && sizes[0] == sz && sizes[1] == sz, "listed sizes are the blob sizes");
    CHECK(qml_sqlite_uses_blob("live") == 1, "'live' is referenced");
    CHECK(qml_sqlite_uses_blob("gone") == 0, "'gone' is an orphan");

    // the SWEEP_BLOBS loop
    size_t removed = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (qml_sqlite_uses_blob(keys[i]) == 0 && qml_lmdb_delete_blob(keys[i]) == 0)
        {
            ++removed;
        }
    }
    qml_lmdb_free_list(keys, sizes, n);
    CHECK(removed == 1, "the sweep removes one blob");

    size_t bsz = 0;
    CHECK(qml_lmdb_blob_stat("gone", &bsz) != 0, "orphan blob is gone");
    CHECK(qml_lmdb_blob_stat("live", &bsz) == 0 && bsz == sz, "live blob is untouched");
    CHECK(qml_lmdb_list_blobs(&keys, &sizes, &n) == 0 && n == 1, "one blob left");
    qml_lmdb_free_list(keys, sizes, n);

    free(blob);
    printf(g_fail ? "SOME TESTS FAILED\n" : "ALL PASSED\n");
    printf("test files left in %s\n", dir);
    return g_fail;
}
