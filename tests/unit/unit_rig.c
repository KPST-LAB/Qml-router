// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
//
// unit_qml_rig.c — unit tests for core/qml_rig.c (no GPU, engine or DB needed).
//
//   cc -I include -I storage/sqlite tests/unit/unit_rig.c core/qml_rig.c -o /tmp/t
//   /tmp/t
//
// Covers: qml_rig_query (query-term filter), qml_rig_n_words,
//       qml_rig_gate (relative decision), qml_rig_match_count (minimum match --
//       follow-up to TEST-LEDGER defect #11, 2026-08-26),
//       qml_rig_contains (in-context duplicate guard, 2026-08-27).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qml_router.h"

static int g_fail = 0;

#define CHECK(cond, name)                                        \
    do {                                                         \
        if (cond) { printf("ok   %s\n", (name)); }               \
        else      { printf("FAIL %s\n", (name)); g_fail = 1; }   \
    } while (0)

static void test_build_fts_query(void)
{
    // role words, Korean function words and all-digit terms are dropped; mixed alphanumerics are kept
    char * q = qml_rig_query("user 오늘 2026 MX-207 서울 날씨");
    CHECK(q != NULL, "build: usable words survive");
    CHECK(q != NULL && strstr(q, "\"MX\"") != NULL, "build: mixed alnum kept (MX)");
    CHECK(q != NULL && strstr(q, "\"207\"") == NULL, "build: all-digit excluded (207)");
    CHECK(q != NULL && strstr(q, "\"서울\"") != NULL, "build: Korean content word kept");
    CHECK(q != NULL && strstr(q, "\"오늘\"") == NULL, "build: Korean function word excluded");
    CHECK(q != NULL && strstr(q, "\"user\"") == NULL, "build: role word excluded");
    CHECK(q != NULL && strstr(q, "\"2026\"") == NULL, "build: all-digit year excluded");
    free(q);
    // NULL when no usable term is left
    q = qml_rig_query("user 오늘 2026");
    CHECK(q == NULL, "build: no usable word -> NULL");
    q = qml_rig_query(NULL);
    CHECK(q == NULL, "build: NULL in -> NULL out");
    // upper limit of 12 terms
    q = qml_rig_query("w01 w02 w03 w04 w05 w06 w07 w08 w09 w10 w11 w12 w13 w14");
    int n_or = 0;
    for (const char * p = q; (p = strstr(p, " OR ")) != NULL; p += 4) { n_or++; }
    CHECK(q != NULL && n_or == 11, "build: 12-word cap (11 ORs)");
    free(q);
}

static void test_n_eff(void)
{
    CHECK(qml_rig_n_words("what is the weather in Seoul") == 2, "n_eff: stopwords excluded");
    CHECK(qml_rig_n_words(NULL) == 0, "n_eff: NULL -> 0");
}

static void test_rig_gate(void)
{
    // conf <= 0 = gate off (always inject)
    CHECK(qml_rig_gate(-1.0, -5.0, 1, 4, 0.0) == 1, "gate: conf 0 always inject");
    // top candidate: sim = 1.0
    CHECK(qml_rig_gate(-5.0, -5.0, 4, 4, 0.9) == 1, "gate: top hit passes");
    // lower-ranked candidate: sim = 0.2 < conf 0.5
    CHECK(qml_rig_gate(-1.0, -5.0, 4, 4, 0.5) == 0, "gate: weak hit blocked");
    // info damping: n_eff 1/4 -> even with sim 1.0, 0.25 < 0.5
    CHECK(qml_rig_gate(-5.0, -5.0, 1, 4, 0.5) == 0, "gate: low-content damped");
}

static void test_query_match_count(void)
{
    // basic case: the query terms really occur in the candidate
    CHECK(qml_rig_match_count("서울 날씨 예보", "서울의 내일 날씨는 맑음") == 2,
          "match: two words found (agglutinated substring ok)");
    // defect #11 scenario: relative sim can be 1.0 while the absolute match is 0
    CHECK(qml_rig_match_count("납품 단가 계약", "오늘 점심 메뉴는 김치찌개") == 0,
          "match: unrelated candidate -> 0 (defect #11 floor)");
    // case-insensitive (ASCII)
    CHECK(qml_rig_match_count("LMCache 성능", "lmcache is a kv layer") == 1,
          "match: ASCII case-insensitive");
    // a repeated query term counts once
    CHECK(qml_rig_match_count("서울 서울 서울", "서울 시내") == 1,
          "match: duplicate query words counted once");
    // function words and all-digit terms are not query terms, so they are not counted
    CHECK(qml_rig_match_count("오늘 2026", "오늘은 2026년이다") == 0,
          "match: function word + all-digit contribute nothing");
    // NULL-safe
    CHECK(qml_rig_match_count(NULL, "abc") == 0, "match: NULL question -> 0");
    CHECK(qml_rig_match_count("abc", NULL) == 0, "match: NULL candidate -> 0");
}

static void test_tokens_contain(void)
{
    const int32_t hay[]    = { 5, 9, 12, 7, 3, 12, 7, 8 };
    const int32_t mid[]    = { 12, 7, 3 };
    const int32_t tail[]   = { 12, 7, 8 };
    const int32_t absent[] = { 7, 12 };
    const int32_t whole[]  = { 5, 9, 12, 7, 3, 12, 7, 8 };
    const int32_t longer[] = { 5, 9, 12, 7, 3, 12, 7, 8, 1 };
    CHECK(qml_rig_contains(hay, 8, mid, 3) == 1, "contain: middle subsequence");
    CHECK(qml_rig_contains(hay, 8, tail, 3) == 1, "contain: tail subsequence");
    CHECK(qml_rig_contains(hay, 8, absent, 2) == 0, "contain: order matters (7,12 absent)");
    CHECK(qml_rig_contains(hay, 8, whole, 8) == 1, "contain: whole == hay");
    CHECK(qml_rig_contains(hay, 8, longer, 9) == 0, "contain: needle longer than hay");
    CHECK(qml_rig_contains(hay, 8, NULL, 3) == 0, "contain: NULL needle");
    CHECK(qml_rig_contains(NULL, 8, mid, 3) == 0, "contain: NULL hay");
    CHECK(qml_rig_contains(hay, 8, mid, 0) == 0, "contain: empty needle");
}

int main(void)
{
    test_build_fts_query();
    test_n_eff();
    test_rig_gate();
    test_query_match_count();
    test_tokens_contain();
    printf(g_fail ? "\nRESULT: FAIL\n" : "\nRESULT: ALL PASS\n");
    return g_fail;
}
