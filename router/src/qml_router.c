// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
// router/src/qml_router.c
//
// The routing layer's C implementation — first resident: the in-process
// embedding engine (2026-08-29, author direction: no separate embedding
// server; embed inside the same process in C). The hybrid retrieval path
// previously required a second llama-server process serving bge-m3 over
// HTTP; on single-engine deployments (one process + N agent sessions,
// e.g. Jetson AGX Orin) that second process is not acceptable. This module
// loads the embedding model with the llama C API inside the SAME process,
// so hybrid retrieval keeps the one-process topology.
//
// Threading: one context guarded by a mutex — embedding calls are short
// (tens of ms for turn-sized texts) and come from the hook's retrieval
// path only, so serialization is acceptable and keeps this file simple.

#include "qml_router.h"

#include "llama.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct llama_model *   g_emb_model = NULL;
static struct llama_context * g_emb_ctx   = NULL;
static int                    g_emb_dim   = 0;
static pthread_mutex_t        g_emb_lock  = PTHREAD_MUTEX_INITIALIZER;

// Embedding documents/queries are short per-turn texts — if longer than this, only the first part is used
// (harmless loss for retrieval purposes; bge-m3's own limit is 8192).
#define QML_EMBED_MAX_TOKENS 1024

//------------------------------------------------------------
//Func : qml_embed_init
//  in: model_path — embedding model gguf (e.g. bge-m3); n_gpu_layers —
//      0 = CPU only (recommended: does not touch the generation engine's VRAM), 99 = all on GPU
// out: 0 success / -1 failure (0 if already initialized — does not re-initialize)
//------------------------------------------------------------
int qml_embed_init(const char * model_path, int n_gpu_layers)
{
    if (model_path == NULL || model_path[0] == '\0')
    {
        return -1;
    }
    pthread_mutex_lock(&g_emb_lock);
    if (g_emb_ctx != NULL)
    {
        pthread_mutex_unlock(&g_emb_lock);
        return 0;
    }

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    g_emb_model = llama_model_load_from_file(model_path, mp);
    if (g_emb_model == NULL)
    {
        fprintf(stderr, "[qml][embed-c] model load failed: %s\n", model_path);
        pthread_mutex_unlock(&g_emb_lock);
        return -1;
    }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = QML_EMBED_MAX_TOKENS;
    cp.n_batch      = QML_EMBED_MAX_TOKENS;
    cp.n_ubatch     = QML_EMBED_MAX_TOKENS;
    cp.embeddings   = true;
    // pooling follows the default in the model metadata (bge-m3 = CLS)
    g_emb_ctx = llama_init_from_model(g_emb_model, cp);
    if (g_emb_ctx == NULL)
    {
        fprintf(stderr, "[qml][embed-c] context init failed\n");
        llama_model_free(g_emb_model);
        g_emb_model = NULL;
        pthread_mutex_unlock(&g_emb_lock);
        return -1;
    }
    g_emb_dim = llama_model_n_embd(g_emb_model);
    fprintf(stderr, "[qml][embed-c] in-process embedding up: %s dim=%d ngl=%d\n",
            model_path, g_emb_dim, n_gpu_layers);
    pthread_mutex_unlock(&g_emb_lock);
    return 0;
}

//------------------------------------------------------------
//Func : qml_embed_dim
// out: embedding dimension (0 if not initialized)
//------------------------------------------------------------
int qml_embed_dim(void)
{
    return g_emb_dim;
}

//------------------------------------------------------------
//Func : qml_embed_text
//  in: text — one UTF-8 text (per turn; truncated beyond 1,024 tokens)
//  in: out — float array (size at least qml_embed_dim())
//  in: out_cap — capacity of out (number of floats)
// out: 0 success (L2-normalized vector in out) / -1 failure
//------------------------------------------------------------
int qml_embed_text(const char * text, float * out, int out_cap)
{
    if (text == NULL || out == NULL || g_emb_ctx == NULL ||
        out_cap < g_emb_dim || g_emb_dim <= 0)
    {
        return -1;
    }
    pthread_mutex_lock(&g_emb_lock);

    const struct llama_vocab * vocab = llama_model_get_vocab(g_emb_model);
    llama_token toks[QML_EMBED_MAX_TOKENS];
    int n = llama_tokenize(vocab, text, (int32_t) strlen(text),
                           toks, QML_EMBED_MAX_TOKENS,
                           /*add_special=*/true, /*parse_special=*/false);
    if (n < 0)
    {
        // Truncation. llama_tokenize returns -(required count) when the buffer is too small,
        // so calling again with a buffer of the same size fails again (measured 2026-09-02:
        // on a len=15981 document the whole backfill stopped -- hybrid was silently disabled).
        // Tokenize the whole text into a heap buffer of the required size, then use only the first 1,024.
        const int need = -n;
        llama_token * big = (llama_token *) malloc((size_t) need * sizeof(llama_token));
        if (big == NULL)
        {
            pthread_mutex_unlock(&g_emb_lock);
            return -1;
        }
        const int m = llama_tokenize(vocab, text, (int32_t) strlen(text),
                                     big, need, true, false);
        if (m <= 0)
        {
            free(big);
            pthread_mutex_unlock(&g_emb_lock);
            return -1;
        }
        n = (m < QML_EMBED_MAX_TOKENS) ? m : QML_EMBED_MAX_TOKENS;
        memcpy(toks, big, (size_t) n * sizeof(llama_token));
        free(big);
    }
    if (n == 0)
    {
        pthread_mutex_unlock(&g_emb_lock);
        return -1;
    }

    // Encoder-type models keep no state, but memory is cleared on every call so that leftovers
    // of pooled seq results do not mix into the next call (same as server /embedding).
    llama_memory_clear(llama_get_memory(g_emb_ctx), true);

    struct llama_batch batch = llama_batch_get_one(toks, n);
    if (llama_encode(g_emb_ctx, batch) != 0)
    {
        fprintf(stderr, "[qml][embed-c] encode failed (n=%d)\n", n);
        pthread_mutex_unlock(&g_emb_lock);
        return -1;
    }

    const float * emb = llama_get_embeddings_seq(g_emb_ctx, 0);
    if (emb == NULL)
    {
        emb = llama_get_embeddings(g_emb_ctx);
    }
    if (emb == NULL)
    {
        pthread_mutex_unlock(&g_emb_lock);
        return -1;
    }

    // L2 normalization — makes cosine comparison stable as a dot product (existing vectors are also stored normalized)
    double ss = 0.0;
    for (int i = 0; i < g_emb_dim; i++)
    {
        ss += (double) emb[i] * (double) emb[i];
    }
    const float inv = (ss > 0.0) ? (float) (1.0 / sqrt(ss)) : 0.0f;
    for (int i = 0; i < g_emb_dim; i++)
    {
        out[i] = emb[i] * inv;
    }
    pthread_mutex_unlock(&g_emb_lock);
    return 0;
}
