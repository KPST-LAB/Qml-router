// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
//
// lmdb_usage.c -- read-only LMDB space report for a qml store (2026-10-02).
// Prints the map size, the page high-water mark and the free-list pages, so the
// space actually in use is (high-water pages - free pages) * page size. Opening
// read-only is safe while engines have the store open.
//
//   cc -O2 -o lmdb_usage tests/unit/lmdb_usage.c -llmdb
//   ./lmdb_usage <store>/qml_lmdb

#include <lmdb.h>
#include <stdio.h>

int main(int argc, char ** argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "usage: %s <lmdb dir>\n", argv[0]);
        return 2;
    }
    MDB_env * env = NULL;
    mdb_env_create(&env);
    int rc = mdb_env_open(env, argv[1], MDB_RDONLY, 0644);
    if (rc != 0)
    {
        printf("open rc=%d (%s)\n", rc, mdb_strerror(rc));
        return 1;
    }
    MDB_envinfo info;
    MDB_stat    st;
    mdb_env_info(env, &info);
    mdb_env_stat(env, &st);

    // the free list is DBI 0 inside LMDB; each record is an ID list whose first
    // word is the number of page ids in it
    size_t       free_pages = 0;
    MDB_txn *    txn = NULL;
    MDB_cursor * cur = NULL;
    MDB_val      k, v;
    mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);
    if (mdb_cursor_open(txn, 0, &cur) == 0)
    {
        while (mdb_cursor_get(cur, &k, &v, MDB_NEXT) == 0)
        {
            free_pages += *(const size_t *) v.mv_data;
        }
        mdb_cursor_close(cur);
    }
    mdb_txn_abort(txn);

    const size_t hw   = info.me_last_pgno + 1;
    const size_t used = hw - free_pages;
    printf("map_bytes=%zu page=%u high_water_pages=%zu free_pages=%zu in_use_pages=%zu"
           " in_use_GB=%.2f free_reusable_GB=%.2f\n",
           info.me_mapsize, st.ms_psize, hw, free_pages, used,
           used * (double) st.ms_psize / 1e9, free_pages * (double) st.ms_psize / 1e9);
    mdb_env_close(env);
    return 0;
}
