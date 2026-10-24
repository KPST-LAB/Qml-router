// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
// core/qml_model_hash.c
//
// See qml_model_hash.h for the scope/rationale.

#include "qml_model_hash.h"

#include <openssl/evp.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define QML_SAMPLE_CHUNK_BYTES (4 * 1024 * 1024)

// Converts a byte buffer to a NUL-terminated lowercase hex string.
//  in: bytes — input bytes to encode
//  in: n — number of input bytes
//  in: out_hex — output buffer; caller must provide at least 2*n + 1 chars
// out: none (writes 2*n hex digits plus a trailing NUL into out_hex)
static void bytes_to_hex(const unsigned char * bytes, size_t n, char * out_hex)
{
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < n; ++i)
    {
        out_hex[i * 2]     = digits[(bytes[i] >> 4) & 0xF];
        out_hex[i * 2 + 1] = digits[bytes[i] & 0xF];
    }
    out_hex[n * 2] = '\0';
}

// Computes a fast SHA-256 fingerprint of a model file: size + mtime + sampled content
// (whole file when <= 8 MiB, otherwise the first and last 4 MiB chunks).
//  in: path — filesystem path of the model file
//  in: out_hex — output buffer for 64 lowercase hex chars + NUL (65 bytes)
// out: 0 on success; -1 on stat/open/digest-init/allocation failure (out_hex then unset)
int qml_model_file_hash(const char * path, char out_hex[65])
{
    struct stat st;
    if (stat(path, &st) != 0)
    {
        return -1;
    }

    FILE * f = fopen(path, "rb");
    if (f == NULL)
    {
        return -1;
    }

    EVP_MD_CTX * ctx = EVP_MD_CTX_new();
    if (ctx == NULL || EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) != 1)
    {
        if (ctx != NULL) EVP_MD_CTX_free(ctx);
        fclose(f);
        return -1;
    }

    const int64_t size64  = (int64_t) st.st_size;
    const int64_t mtime64 = (int64_t) st.st_mtime;
    EVP_DigestUpdate(ctx, &size64,  sizeof(size64));
    EVP_DigestUpdate(ctx, &mtime64, sizeof(mtime64));

    unsigned char * buf = (unsigned char *) malloc(QML_SAMPLE_CHUNK_BYTES);
    if (buf == NULL)
    {
        EVP_MD_CTX_free(ctx);
        fclose(f);
        return -1;
    }

    if (st.st_size <= (off_t) (2 * QML_SAMPLE_CHUNK_BYTES))
    {
        // small enough -- just hash the whole thing
        size_t nread;
        while ((nread = fread(buf, 1, QML_SAMPLE_CHUNK_BYTES, f)) > 0)
        {
            EVP_DigestUpdate(ctx, buf, nread);
        }
    }
    else
    {
        // first chunk
        size_t nread = fread(buf, 1, QML_SAMPLE_CHUNK_BYTES, f);
        EVP_DigestUpdate(ctx, buf, nread);

        // last chunk
        if (fseeko(f, -(off_t) QML_SAMPLE_CHUNK_BYTES, SEEK_END) == 0)
        {
            nread = fread(buf, 1, QML_SAMPLE_CHUNK_BYTES, f);
            EVP_DigestUpdate(ctx, buf, nread);
        }
    }

    free(buf);
    fclose(f);

    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int  digest_len = 0;
    EVP_DigestFinal_ex(ctx, digest, &digest_len);
    EVP_MD_CTX_free(ctx);

    bytes_to_hex(digest, digest_len, out_hex);

    return 0;
}
