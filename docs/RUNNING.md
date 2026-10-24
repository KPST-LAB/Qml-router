<!--
SPDX-License-Identifier: AGPL-3.0-only
Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
-->

# qml-router — Building and Running

Everything below was exercised on: x86_64 (Ubuntu 22.04/18.04, RTX 4090,
CUDA 11.8–12.x) and Jetson AGX Orin (JetPack, CUDA 13.2, `sm_87`).
Engine versions used in the paper: llama.cpp build b10141 (upstream commit
`0d47ea742`) + the qml patch, vLLM 0.26.0, SGLang 0.5.16.

## 1. llama.cpp + qml (the primary path)

```bash
# dependencies (Debian/Ubuntu)
sudo apt install cmake build-essential libsqlite3-dev liblmdb-dev libssl-dev

# apply the patch onto the pinned upstream commit
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout 0d47ea742
git -C third_party/llama.cpp apply ../../patches/llama.cpp-qml-session.patch

# build (set your GPU arch: 89 = RTX 4090, 87 = AGX Orin)
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DLLAMA_QML_SESSION=ON -DQML_ROUTER_DIR=$(pwd) -DLLAMA_CURL=OFF
cmake --build third_party/llama.cpp/build --target llama-server -j
```

Run (the recommended full configuration measured in the paper — shared pool,
engine RAM cache off, 4-bit KV, asynchronous save/restore):

```bash
third_party/llama.cpp/build/bin/llama-server \
  -m models/Llama-3.2-3B-Instruct-F16.gguf -ngl 99 -fa on \
  -c 131072 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 \
  --port 8397 \
  --qml-session-store store/qml.db --qml-lmdb-store store/qml_lmdb
```

A session request sends only the new turn:

```bash
curl http://127.0.0.1:8397/completion -d '{
  "prompt": "and what was my name again?",
  "session_id": "alice::chat-1", "cache_prompt": true, "n_predict": 64}'
```

Tunables (env): `QML_CAPTURE_MIN_DELTA` (default 512 — below this token
delta a turn saves tokens only, keeping the previous blob as a restore
prefix).

## 2. vLLM + qml (official KVConnector, zero source changes)

```bash
pip install vllm==0.26.0
# store library the connector loads (build/libqml_session_store.so)
cmake -S . -B build && cmake --build build --target qml_session_store_shared -j
mkdir -p store
VLLM_PLUGINS="" PYTHONPATH=hooks/vllm python3 -m vllm.entrypoints.openai.api_server \
  --model unsloth/Llama-3.2-3B-Instruct --port 8000 \
  --kv-transfer-config '{"kv_connector":"QmlVllmConnector","kv_connector_module_path":"qml_vllm_connector","kv_role":"kv_both","kv_buffer_device":"cpu","kv_connector_extra_config":{"qml_sqlite_path":"store/qml.db","qml_lmdb_path":"store/qml_lmdb","qml_self_url":"http://127.0.0.1:8000","qml_read_threads":"8"}}'
```

Identity is carried per request by the `cache_salt` field (set it to the session id). At-rest
INT8 quantization of blocks is optional and auto-detected on load. Behind the session layer of
section 4, add `"qml_token_log":"off"` to `kv_connector_extra_config`.

## 3. SGLang + qml (official HiCacheStorage backend, zero source changes)

```bash
pip install sglang==0.5.16
mkdir -p store
PYTHONPATH=hooks/sglang python3 -m sglang.launch_server \
  --model-path unsloth/Llama-3.2-3B-Instruct --port 30000 --page-size 64 \
  --enable-hierarchical-cache --hicache-write-policy write_through \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{"module_path":"qml_sglang_storage","class_name":"QmlSglangStorage","backend_name":"qml","qml_sqlite_path":"store/qml.db","qml_lmdb_path":"store/qml_lmdb"}'
```

Identity is carried per request by `extra_key` (set it to the session id). Behind the session
layer of section 4, add `"qml_token_log":"off"` to the extra config.

## 4. Session layer for vLLM and SGLang

```
client  ->  router/qml_openai_gateway.py (:8090)  ->  vLLM (:8000) or SGLang (:30000)
```

```bash
python3 router/qml_openai_gateway.py --engine vllm --engine-url http://127.0.0.1:8000 \
  --model MODEL_NAME --tokenizer MODEL_DIR --store-db store/qml.db \
  --store-lmdb store/qml_lmdb --window 3000 --port 8090
```

The engine runs with the qml-router connector (vLLM) or backend (SGLang) on the same store,
started as in sections 2 and 3 with `"qml_token_log":"off"`, so the token log is kept by the
session layer only. `docs/API.md` section 3 lists the endpoints.

## 5. Benchmarks

Every measurement in the paper has a script under `tests/` (each script's
docstring states usage, required engine state, and output). The
methodology contract they share: save time excluded from timers, cache
state proved (not assumed) via `mincore()`, temperature 0, and **every
timed response gated by needle recall** — an unverified time is not a
result. On edge devices, record the power mode with every run
(`/var/lib/nvpmodel/status`); it moves prefill by more than 3x.

## 6. Operational notes

- The store directory is portable by file copy while the engine is down;
  cross-engine continuation rebuilds KV from the token log (one prefill).
- Never point two engines at one store's LMDB concurrently from different
  machines; the command bus and locks assume one host per store.
- The gateway's stores and the engine flags must reference the same paths.
- On NVMe edge boards, verify `dmesg` shows no `completion polled` NVMe
  timeouts before trusting any storage-bound number (ASPM power management
  can silently degrade reads by ~66x; fix: `pcie_aspm=performance`,
  `pm_qos_latency_tolerance_us=0`).
