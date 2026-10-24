// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
//
// qml_model_hash.h
//
// Real content-fingerprinting for model checkpoint files, replacing the
// filename/repo-id-string placeholder that was previously used everywhere
// as "model_hash" (see docs -- task #22). Scope is deliberately narrow,
// per an explicit user correction earlier in this project: this can ONLY
// detect a same-path/same-format checkpoint silently swapped for a
// different one. It CANNOT and must NOT be used to claim cross-format or
// cross-engine model identity (e.g. comparing a GGUF fingerprint against a
// safetensors fingerprint of "the same" model) -- those are genuinely
// different byte content (lossy quantization, different serialization),
// so a hash mismatch there is correct behavior, not a bug.

#ifndef QML_MODEL_HASH_H
#define QML_MODEL_HASH_H

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------
//Func : qml_model_file_hash
//Computes a fast content fingerprint of the file at `path`, written as a
//64-char lowercase hex string into out_hex (must be at least 65 bytes,
//for the NUL terminator).
//
//Deliberately NOT a full-file SHA256 -- for multi-GB checkpoints that
//would add many seconds to every server startup for negligible extra
//confidence (measured: ~14s for a 2.4GB file on this project's own test
//hardware). Instead hashes: file size + mtime + the first 4MiB + the
//last 4MiB (whole file if smaller than 8MiB). This is a sampled
//fingerprint, not a cryptographic proof the ENTIRE file is byte-identical
//-- sufficient for this function's actual purpose (catching "someone
//replaced the checkpoint at this path"), not a security boundary.
//
//returns 0 on success, -1 if the file cannot be opened/stat'd.
//  in: path — checkpoint file to fingerprint
//  in: out_hex — (output) buffer of at least 65 bytes; receives 64 lowercase
//      hex chars + NUL
//------------------------------------------------------------
int qml_model_file_hash(const char * path, char out_hex[65]);

#ifdef __cplusplus
}
#endif

#endif // QML_MODEL_HASH_H
