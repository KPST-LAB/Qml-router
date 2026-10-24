# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(qml_router_llama_hook CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
find_package(SQLite3 REQUIRED)
find_library(LMDB_LIBRARY NAMES lmdb REQUIRED)
find_path(LMDB_INCLUDE_DIR NAMES lmdb.h)
find_package(OpenSSL REQUIRED) # SHA256 for core/qml_model_hash.c (task #22/#48)

# When this project is built as a subdirectory of a tree whose vendored
# subprojects already ran find_package(OpenSSL) (llama.cpp does, via
# cpp-httplib), the OpenSSL::* imported targets can be absent from THIS
# directory scope even though the libraries were found. Recreate them from the
# plain variables so the link step below works regardless of build order.
# Hit on an AGX Orin build (2026-08-03) where the libraries resolved but the
# targets did not exist.
if (OPENSSL_FOUND AND NOT TARGET OpenSSL::Crypto)
    add_library(OpenSSL::Crypto UNKNOWN IMPORTED)
    set_target_properties(OpenSSL::Crypto PROPERTIES
        IMPORTED_LOCATION             "${OPENSSL_CRYPTO_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENSSL_INCLUDE_DIR}")
endif()
if (OPENSSL_FOUND AND NOT TARGET OpenSSL::SSL)
    add_library(OpenSSL::SSL UNKNOWN IMPORTED)
    set_target_properties(OpenSSL::SSL PROPERTIES
        IMPORTED_LOCATION             "${OPENSSL_SSL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENSSL_INCLUDE_DIR}")
endif()

# ────────────────────────────────────────────────────────────
# libqml_llama_preload.so — LD_PRELOAD symbol interceptor
#
# No llama.cpp source file is modified and no add_subdirectory() build of
# third_party/llama.cpp is needed: every llama_* symbol this library touches
# (intercepted or plain helper call) is resolved at runtime via
# dlsym(RTLD_NEXT, ...) — see hooks/llama.cpp/qml_llama_preload.cpp. That keeps
# this build free of any link-time dependency on a specific libllama.so, so
# the same .so can be LD_PRELOADed into either the prebuilt llama-bin
# binaries or a freshly built third_party/llama.cpp checkout.
# ────────────────────────────────────────────────────────────
add_library(qml_llama_preload SHARED
    hooks/llama.cpp/qml_llama_preload.cpp
    core/qml_session.c
    core/qml_window.c
    core/qml_rig.c
    storage/sqlite/qml_sqlite_store.c
    storage/lmdb/qml_lmdb_store.c
)

target_include_directories(qml_llama_preload PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include      # llama.h (public API declarations only)
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include # ggml.h -- pulled in by llama.h, types only, not linked
    ${CMAKE_CURRENT_SOURCE_DIR}/include                            # qml_router.h
    ${CMAKE_CURRENT_SOURCE_DIR}/storage/sqlite                     # qml_sqlite_store.h, qml_sqlite_schema.h
    ${CMAKE_CURRENT_SOURCE_DIR}/storage/lmdb                       # qml_lmdb_store.h
    ${LMDB_INCLUDE_DIR}
)

target_link_libraries(qml_llama_preload PRIVATE
    ${CMAKE_DL_LIBS}      # dlsym/RTLD_NEXT
    Threads::Threads
    SQLite::SQLite3
    ${LMDB_LIBRARY}
)

# ────────────────────────────────────────────────────────────
# qml_session_store — engine-agnostic session storage, as a linkable
# STATIC library rather than a preload .so.
#
# For hosts that compile our session persistence directly into their own
# build instead of intercepting it via LD_PRELOAD (e.g. llama-server,
# see docs/udf-control-plane-proposal.md background and the llama-server
# integration plan) — no llama.cpp headers are needed here at all, since
# core/qml_session.c and the storage layer only ever deal in raw byte
# buffers and token id arrays, never llama.cpp types.
# ────────────────────────────────────────────────────────────
add_library(qml_session_store STATIC
    core/qml_session.c
    core/qml_window.c
    core/qml_rig.c
    core/qml_model_hash.c
    storage/sqlite/qml_sqlite_store.c
    storage/lmdb/qml_lmdb_store.c
)

target_include_directories(qml_session_store PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/storage/sqlite
    ${CMAKE_CURRENT_SOURCE_DIR}/storage/lmdb
    ${LMDB_INCLUDE_DIR}
)

target_link_libraries(qml_session_store PUBLIC
    Threads::Threads
    SQLite::SQLite3
    ${LMDB_LIBRARY}
    OpenSSL::Crypto
)

# ────────────────────────────────────────────────────────────
# qml_session_store_shared — same sources as qml_session_store, built as a
# shared object instead of a static archive, purely so non-C hosts (the
# vLLM/SGLang Python adapters, see hooks/vllm and hooks/sglang) can dlopen
# it via ctypes without needing to link/compile anything themselves.
# llama-server does NOT use this target -- it links the STATIC
# qml_session_store directly, compiled into its own binary.
# ────────────────────────────────────────────────────────────
add_library(qml_session_store_shared SHARED
    core/qml_session.c
    core/qml_window.c
    core/qml_rig.c
    core/qml_model_hash.c
    storage/sqlite/qml_sqlite_store.c
    storage/lmdb/qml_lmdb_store.c
)

set_target_properties(qml_session_store_shared PROPERTIES OUTPUT_NAME qml_session_store)

target_include_directories(qml_session_store_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/storage/sqlite
    ${CMAKE_CURRENT_SOURCE_DIR}/storage/lmdb
    ${LMDB_INCLUDE_DIR}
)

target_link_libraries(qml_session_store_shared PUBLIC
    Threads::Threads
    SQLite::SQLite3
    ${LMDB_LIBRARY}
    OpenSSL::Crypto
)

# ────────────────────────────────────────────────────────────
# qml_llama_hook — llama.cpp-aware wrapper around qml_session_store
# (see hooks/llama.cpp/qml_llama_hook.cpp for the 3-layer rationale).
#
# Only added when this CMakeLists.txt is pulled in via add_subdirectory()
# from inside an actual llama.cpp build (the `llama` target already
# exists in that case) -- a standalone qml-router build (e.g. for the
# LD_PRELOAD target above) has no llama.cpp tree to link against and
# doesn't need this target at all.
# ────────────────────────────────────────────────────────────
if (TARGET llama)
    add_library(qml_llama_hook STATIC
        hooks/llama.cpp/qml_llama_hook.cpp
        # First component of the routing layer's C implementation — in-process embedding engine
        # (separate embedding server removed, 2026-08-29 author's direction)
        router/src/qml_router.c
    )

    target_include_directories(qml_llama_hook PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        # Embedding server HTTP client (httplib) and response JSON parser (nlohmann) —
        # reuses what llama.cpp already vendors (no new dependency)
        ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib
        ${CMAKE_SOURCE_DIR}/vendor
    )

    target_link_libraries(qml_llama_hook PUBLIC
        qml_session_store
        llama
    )
endif()
