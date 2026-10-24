// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
//
// qml_rig.c — retrieval/ranking policy shared by every engine hook.
//
// These functions were born inside hooks/llama.cpp/qml_llama_hook.cpp and
// moved here verbatim (2026-08-23) so that (a) the C core, not a gateway or a
// single hook, owns the measured RIG policy, and (b) the llama.cpp / vLLM /
// SGLang paths can share one implementation instead of per-hook ports.
// Behavior is intentionally identical to the C++ originals; the measured
// tunings (stopword lists, 12-word cap, sim*info gate) are the ones recorded
// in the project's measurement records (recall series, traps 21-22).

#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "qml_router.h"

// query-word filter shared by qml_rig_query and
// qml_rig_match_count — the exclusion rules are measured tunings
// (TEST-LEDGER recall series, traps 21-22) and MUST stay identical between
// the FTS query that ranks candidates and the absolute match check that
// verifies them.
static const char * k_role_words[] = { "user", "assistant", "system", "tool",
    "현재", "날짜", "요일", "오늘", "내일", "어제",
    "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일",
    "알려줘", "알려줘라", "알려주세요", "자세하게", "자세히", "간단히",
    "무엇", "무엇인가", "뭐야", "뭔가", "어디서", "어떻게", "그리고",
    "그러면", "그런데", "해줘", "주세요", "합니다", "있는가", "인가",
    "말해줘", "정리해", "설명해", "근거로", "자료를", "답하라", "위",
    "아래", "관련", "내용", "정보",
    // [2026-09-03, TEST-LEDGER row 75] English common words (stopwords) — if not removed,
    // they consume the 12-word query term limit and content words drop out (measured: "camping ... year"
    // question lost "year"). Even with a low BM25 weight they still take a slot under the limit, so
    // they are excluded by a list. Same kind of rule as excluding Korean function words.
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "from",
    "and", "or", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "have", "has", "had", "will", "would", "can", "could", "should",
    "i", "me", "my", "you", "your", "we", "our", "it", "its", "this",
    "that", "these", "those", "there", "what", "when", "where", "which",
    "who", "how", "why", "many", "much", "any", "all", "some", "about",
    "please", "tell", "also", "than", "then", "so", "as", "by", "up",
    NULL };

//------------------------------------------------------------
//Func : next_query_word (static)
//advance *pp to the next query-eligible word: length >= 2, not a
//role/function word, not all-digit. Word bytes are isalnum or >= 0x80
//(multibyte passthrough, matching FTS5 unicode61).
//  in: pp — cursor into the text (advanced past the returned word)
// out: 1 with *s_out/*len_out set when an eligible word was found; 0 at end
//------------------------------------------------------------
static int next_query_word(const char ** pp, const char ** s_out, size_t * len_out)
{
    const char * p = *pp;
    while (*p != '\0')
    {
        while (*p != '\0' && !(isalnum((unsigned char) *p) || (unsigned char) *p >= 0x80))
        {
            p++;
        }
        const char * s = p;
        while (*p != '\0' && (isalnum((unsigned char) *p) || (unsigned char) *p >= 0x80))
        {
            p++;
        }
        const size_t wlen = (size_t) (p - s);
        int is_role = 0;
        for (int r = 0; k_role_words[r] != NULL; r++)
        {
            if (wlen == strlen(k_role_words[r]) && strncasecmp(s, k_role_words[r], wlen) == 0)
            {
                is_role = 1;
                break;
            }
        }
        int all_digit = 1;
        for (const char * d = s; d < p; d++)
        {
            if (*d < '0' || *d > '9')
            {
                all_digit = 0;
                break;
            }
        }
        if (wlen >= 2 && !is_role && !all_digit)
        {
            *pp      = p;
            *s_out   = s;
            *len_out = wlen;
            return 1;
        }
        if (*p == '\0')
        {
            break;
        }
    }
    *pp = p;
    return 0;
}

//------------------------------------------------------------
//Func : qml_rig_query
//FTS5 MATCH expression from raw new-turn text: up to 12 quoted words
//(len >= 2) joined with OR. Excluded, by measurement: chat-template role
//names, Korean function/date words (the gateway's per-turn date line made
//content-free follow-ups match everything), and all-digit tokens ("2026").
//Mixed alphanumerics ("MX-207") survive.
//  in: text — raw new-turn text (UTF-8; multibyte runs pass through, which
//      matches FTS5's default unicode61 tokenizer)
// out: malloc'd query string, or NULL when no usable word remains (callers
//      should then skip the search entirely). Free with free().
//------------------------------------------------------------
char * qml_rig_query(const char * text)
{
    if (text == NULL)
    {
        return NULL;
    }
    size_t cap = strlen(text) * 2 + 32;   // worst case: every word quoted + OR
    char * q   = (char *) malloc(cap);
    if (q == NULL)
    {
        return NULL;
    }
    size_t       qlen    = 0;
    int          n_words = 0;
    const char * p       = text;
    const char * s       = NULL;
    size_t       wlen    = 0;
    while (n_words < 12 && next_query_word(&p, &s, &wlen))
    {
        if (qlen + wlen + 8 > cap)
        {
            break;   // cap generously sized; never truncate mid-word
        }
        if (qlen > 0)
        {
            memcpy(q + qlen, " OR ", 4);
            qlen += 4;
        }
        q[qlen++] = '"';
        memcpy(q + qlen, s, wlen);
        qlen += wlen;
        q[qlen++] = '"';
        n_words++;
    }
    if (n_words == 0)
    {
        free(q);
        return NULL;
    }
    q[qlen] = '\0';
    return q;
}

//------------------------------------------------------------
//Func : qml_rig_match_count
//absolute overlap between a question and one retrieval candidate: how many
//DISTINCT query-eligible words of the question (same filter and 12-word cap
//as qml_rig_query) literally occur in the candidate text
//(ASCII case-insensitive substring; multibyte bytes compared exactly).
//This is the absolute half of the RIG verdict — the sim term of
//qml_rig_gate is relative to the best hit of the same result set
//(TEST-LEDGER defect #11: with one candidate sim is 1.0 by definition), so
//callers use this count as a literal-match floor after the gate passes.
//  in: question — raw new-turn text the FTS query was built from
//  in: candidate — retrieval candidate's stored text
// out: number of distinct query words found in candidate (0 when either
//      input is NULL or no eligible word exists)
//------------------------------------------------------------
//------------------------------------------------------------
//Func : qml_rig_contains
//exact token-subsequence search — does `needle` appear contiguously inside
//`hay`? Used by the RIG in-context duplicate guard: an archived span whose
//token ids already sit in the live log tail is a leftover of a previous
//injection (injection writes the same ids through the slot save path,
//TEST-LEDGER defect #13), so injecting it again only duplicates context.
//  in: hay / n_hay — the live log tail token ids
//  in: needle / n_needle — one injection candidate's token ids
// out: 1 when the exact contiguous subsequence exists, else 0 (empty or
//      oversized needle returns 0)
//------------------------------------------------------------
int qml_rig_contains(const int32_t * hay, size_t n_hay,
                          const int32_t * needle, size_t n_needle)
{
    if (hay == NULL || needle == NULL || n_needle == 0 || n_needle > n_hay)
    {
        return 0;
    }
    for (size_t i = 0; i + n_needle <= n_hay; i++)
    {
        if (hay[i] == needle[0] &&
            memcmp(hay + i, needle, n_needle * sizeof(int32_t)) == 0)
        {
            return 1;
        }
    }
    return 0;
}

int qml_rig_match_count(const char * question, const char * candidate)
{
    if (question == NULL || candidate == NULL)
    {
        return 0;
    }
    const size_t cand_len = strlen(candidate);
    const char * words[12];
    size_t       lens[12];
    int          n_words = 0;
    const char * p       = question;
    const char * s       = NULL;
    size_t       wlen    = 0;
    while (n_words < 12 && next_query_word(&p, &s, &wlen))
    {
        int dup = 0;
        for (int i = 0; i < n_words; i++)   // count duplicate words only once
        {
            if (lens[i] == wlen && strncasecmp(words[i], s, wlen) == 0)
            {
                dup = 1;
                break;
            }
        }
        if (!dup)
        {
            words[n_words] = s;
            lens[n_words]  = wlen;
            n_words++;
        }
    }
    int matched = 0;
    for (int i = 0; i < n_words; i++)
    {
        if (lens[i] > cand_len)
        {
            continue;
        }
        for (size_t off = 0; off + lens[i] <= cand_len; off++)
        {
            if (strncasecmp(candidate + off, words[i], lens[i]) == 0)
            {
                matched++;
                break;
            }
        }
    }
    return matched;
}

//------------------------------------------------------------
//Func : qml_rig_n_words
//effective (non-stopword) word count of a question; feeds the RIG gate's
//info term. The stopword list is deliberately short: English function words
//only, so content words are never discarded.
//  in: text — new-turn (question) text
// out: number of words of length >= 2 that are not stopwords
//------------------------------------------------------------
int qml_rig_n_words(const char * text)
{
    static const char * kStop[] = {
        "the","a","an","and","or","but","if","then","of","to","in","on","at","by",
        "for","with","from","as","is","are","was","were","be","been","being","do",
        "does","did","have","has","had","it","its","this","that","these","those",
        "i","you","he","she","we","they","me","my","your","his","her","our","their",
        "what","when","where","who","whom","which","how","why","can","could","would",
        "should","will","shall","may","might","must","about","again","just","also",
        "there","here","not","no","yes","so","than","too","very","much","many", NULL };
    if (text == NULL)
    {
        return 0;
    }
    int          n_eff = 0;
    const char * p     = text;
    while (*p != '\0')
    {
        while (*p != '\0' && !(isalnum((unsigned char) *p) || (unsigned char) *p >= 0x80))
        {
            p++;
        }
        const char * st = p;
        while (*p != '\0' && (isalnum((unsigned char) *p) || (unsigned char) *p >= 0x80))
        {
            p++;
        }
        const size_t len = (size_t) (p - st);
        if (len < 2)
        {
            continue;
        }
        char         w[64];
        const size_t cp = len < sizeof(w) - 1 ? len : sizeof(w) - 1;
        for (size_t i = 0; i < cp; i++)
        {
            w[i] = (char) tolower((unsigned char) st[i]);
        }
        w[cp] = '\0';
        int stop = 0;
        for (int i = 0; kStop[i] != NULL; i++)
        {
            if (strcmp(w, kStop[i]) == 0)
            {
                stop = 1;
                break;
            }
        }
        if (!stop)
        {
            n_eff++;
        }
    }
    return n_eff;
}

//------------------------------------------------------------
//Func : qml_rig_gate
//RIG injection verdict for one retrieval candidate: rig_score = sim * info,
//where sim is the candidate's BM25 score relative to the best of the SAME
//result set (top hit = 1.0; scores are negative) and info damps low-content
//questions (n_eff below n_min scales proportionally). conf <= 0 disables the
//gate (always inject), preserving pre-gate behavior.
//Known limit, stated in the paper: with a single candidate sim is 1.0 by
//definition; an absolute criterion is an open item.
//  in: score / best_score — this candidate's and the top candidate's BM25
//  in: n_eff / n_min — effective word count of the question and its floor
//  in: conf — gate threshold in (0,1]; <= 0 = gate off
// out: 1 = inject, 0 = skip
//------------------------------------------------------------
int qml_rig_gate(double score, double best_score,
                    int n_eff, int n_min, double conf)
{
    if (conf <= 0.0)
    {
        return 1;
    }
    const double sim  = (score != 0.0 && best_score != 0.0)
                      ? (score / best_score) : 0.0;
    const double info = (n_min <= 0 || n_eff >= n_min)
                      ? 1.0 : (double) n_eff / (double) n_min;
    return (sim * info >= conf) ? 1 : 0;
}

//------------------------------------------------------------
//Func : qml_window_cut
//[TAWT-PAIR 2026-10-22] trim cut that keeps a user turn and its answer
//together. The earlier rule cut after any end-of-turn token, and the eot also
//ends the user turn: the kept tail could start with an answer whose question
//went to the archive (measured on the 2,000-turn weather test: the live log
//began with an assistant "Confirmed." alone). The user turn's role token is read from the log itself:
//a saved log ends with the model's answer, so the message before the last one
//is a user turn; its role token is the one after the header token
//("<|start_header_id|> user", "<|im_start|> user")
//  in: toks / n — the session's token log
//  in: lo — earliest allowed cut position
//  in: eot — end-of-turn token id
// out: earliest cut j >= lo (0 < j < n) where toks[j-1] is an eot and the
//      message starting at j is a user turn; 0 when none fits. With fewer than
//      two messages the role is unknown and the old rule (any eot) applies
//------------------------------------------------------------
size_t qml_window_cut(const int32_t * toks, size_t n, size_t lo, int32_t eot)
{
    if (toks == NULL || n < 2)
    {
        return 0;
    }
    // starts of the last two messages
    size_t last = 0, prev = 0;
    int    n_msg = 0;
    for (size_t j = 0; j < n; j++)
    {
        if (j == 0 || toks[j - 1] == eot)
        {
            prev = last;
            last = j;
            n_msg++;
        }
    }
    const int have_role = (n_msg >= 2 && prev + 1 < n);
    const int32_t user_role = have_role ? toks[prev + 1] : 0;
    for (size_t j = lo > 0 ? lo : 1; j < n; j++)
    {
        if (toks[j - 1] != eot)
        {
            continue;
        }
        if (!have_role || (j + 1 < n && toks[j + 1] == user_role))
        {
            return j;
        }
    }
    return 0;
}
