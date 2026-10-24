// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
//
// qml_format.h
//
// Engine-agnostic base formats shared across every hook backend
// (hooks/llama.cpp, hooks/vllm, hooks/sglang) and every storage backend
// (storage/sqlite, storage/lmdb, ...).
//
// C has no inheritance, so the parent/child relationship is done the
// classic C way: each engine-specific struct embeds one of these as its
// FIRST member (e.g. storage/sqlite/qml_sqlite_store.h's
// qml_llama_session_meta_t embeds qml_session_meta_t as `.base`). That
// keeps a `qml_session_meta_t*` reinterpret-castable from any child type,
// the same trick C APIs like GObject use for single inheritance.

#ifndef QML_FORMAT_H
#define QML_FORMAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------
// engine identifiers -- shared with qml_router_update_kv_mesh()'s
// engine_type parameter
//------------------------------------------------------------
#define QML_ENGINE_LLAMA_CPP 0
#define QML_ENGINE_VLLM      1
#define QML_ENGINE_SGLANG    2

//------------------------------------------------------------
//qml_token_t
//identity of a single token within a sequence, independent of which
//engine produced it
//------------------------------------------------------------
typedef struct
{
    char    user_id[128];
    char    session_id[256];
    int32_t engine_type; // QML_ENGINE_* value (defines above)
    int32_t seq_id;      // engine-native sequence id the token lives in
    int32_t pos;         // token position within the sequence
    int32_t token_id;    // vocabulary id of the token
} qml_token_t;

//------------------------------------------------------------
//qml_op_type_t
//kinds of KV-cache-affecting operations tracked by qml_op_record_t.
//mirrors the llama_memory_* verbs, but the set is engine-agnostic --
//vLLM/SGLang backends map their own block-table operations onto the
//same verbs.
//------------------------------------------------------------
typedef enum
{
    QML_OP_DECODE   = 0, // tokens appended (decode/encode)
    QML_OP_SEQ_ADD  = 1, // position shift
    QML_OP_SEQ_RM   = 2, // range removal
    QML_OP_SEQ_CP   = 3, // sequence copy
    QML_OP_SEQ_KEEP = 4, // keep-only
    QML_OP_SEQ_DIV  = 5, // position divide
    QML_OP_CLEAR    = 6, // full memory clear
} qml_op_type_t;

//------------------------------------------------------------
//qml_op_record_t (OpRE -- Operation Record/Entry)
//one replayable operation, ordered by seq_no within a session. an
//append-only log of these is the actual recovery path for reconstructing
//a session's KV-cache state (replayed directly against the engine's
//native KV storage), not merely an audit trail.
//------------------------------------------------------------
typedef struct
{
    char          session_id[256];
    int32_t       engine_type; // QML_ENGINE_* value (defines above)
    int64_t       seq_no;      // monotonically increasing per session
    qml_op_type_t op_type;
    int32_t       seq_id;      // source sequence id the operation applies to
    int32_t       seq_id_dst;  // QML_OP_SEQ_CP only
    int32_t       p0;          // range start position (llama_memory_seq_* convention)
    int32_t       p1;          // range end position
    int32_t       delta;       // QML_OP_SEQ_ADD shift, or QML_OP_SEQ_DIV divisor
    int64_t       timestamp;
} qml_op_record_t;

//------------------------------------------------------------
//qml_session_meta_t (base)
//universal session identity + position bookkeeping common to every
//engine. engine-specific storage layers embed this as their first
//member and add their own fields after it (e.g. llama.cpp's
//model_hash/lmdb_key/state_size in storage/sqlite/qml_sqlite_store.h).
//------------------------------------------------------------
typedef struct
{
    char    user_id[128];
    char    session_id[256];
    int32_t engine_type; // QML_ENGINE_* value (defines above)
    int32_t seq_id;      // engine-native sequence id of the session's state
    int32_t last_token;  // vocabulary id of the most recently recorded token
    int32_t last_pos;    // highest recorded position (-1 = no tokens yet)
    int64_t n_tokens;    // total tokens recorded for the session
    int64_t updated_at;  // unix epoch seconds of the last update
} qml_session_meta_t;

//------------------------------------------------------------
//qml_vllm_session_meta_t (base)
//vLLM-specific session metadata. Embeds the universal base struct
//and adds fields for PagedAttention's block-based KV cache.
//------------------------------------------------------------
typedef struct
{
    qml_session_meta_t base;
    char    model_hash[65]; // 64 hex chars (SHA256) + NUL -- see core/qml_model_hash.c
    char    lmdb_prefix[256]; // Key prefix for physical blocks in LMDB
    int32_t block_size;       // Size of each physical block
    int32_t num_blocks;       // Total number of physical blocks used

    // The logical-to-physical block mapping (slot_mapping).
    // This is stored as a separate entry in LMDB, keyed by e.g.
    // lmdb_prefix + "_slot_mapping".
} qml_vllm_session_meta_t;

#ifdef __cplusplus
}
#endif

#endif // QML_FORMAT_H
