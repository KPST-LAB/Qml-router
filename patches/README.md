# llama.cpp patch

qml-router's llama-server session persistence requires modifying the llama.cpp source
(LD_PRELOAD cannot reach `server_slot::prompt.tokens` —
see `docs/4-engines/engine-llama-cpp.md`).

## License of the patches

The original llama.cpp code that the patches touch stays under the MIT license of llama.cpp.
The modifications the patches add (this file's `llama.cpp-qml-session.patch` and
`hooks/llama.cpp/server-context-resident-dedup.patch`) are copyright Korea Platform Service
Technology Co., Inc. and are licensed like the rest of this repository: AGPL-3.0-only, or the
commercial license in `LICENSE-COMMERCIAL.md`. See `NOTICE`.

## llama.cpp-qml-session.patch

| Item | Value |
|---|---|
| Base commit | `0d47ea742` ("mtmd: fix android build (#26150)") — the patch was also regenerated against this commit (2026-08-02, including asynchronous restore) |
| Target | `third_party/llama.cpp`, branch `qml-session` (commit `b60c6afbd`). `$HOME/llama.cpp` has been excluded from this project as of 2026-07-31 |
| Build options | `-DLLAMA_QML_SESSION=ON -DQML_ROUTER_DIR=<qml-router path>` |

### Applying

```
cd /path/to/llama.cpp
git apply -3 /path/to/qml-router/patches/llama.cpp-qml-session.patch   # 3-way: applies even if the base differs
cmake -S . -B build -DLLAMA_QML_SESSION=ON -DQML_ROUTER_DIR=/path/to/qml-router -DGGML_CUDA=ON
cmake --build build --target llama-server -j
```

### Modified files and reasons

| File | Content |
|---|---|
| `CMakeLists.txt`, `tools/server/CMakeLists.txt` | `LLAMA_QML_SESSION` option + `qml_session_store` link (OFF by default; when OFF there is no effect at all) |
| `common/arg.cpp`, `common/common.h` | `--qml-session-store`, `--qml-lmdb-store` CLI flags |
| `tools/server/server-task.h`, `server-schema.cpp` | `session_id` / `user_id` fields of the request |
| `tools/server/server-context.cpp` | Main part — restore check in `update_slots()`, asynchronous save in `release()`, auto-continue (resident/saved sessions), control command dispatch, SWA policy, model fingerprint |
| ~~`src/llama-*.h` visibility relaxation~~ | **Removed (2026-07-31)** — it was left over from the investigation phase, and the current code does not access those internals. All changes are now inside `#ifdef LLAMA_QML_SESSION` |

**All new code is inside `#ifdef LLAMA_QML_SESSION`, and the stock logic was not
modified.** The only change of an exceptional nature aligns the `pos_max` argument at the
**call site** of `create_checkpoint()` with the stock convention (`llama_memory_seq_pos_max()`, inclusive);
the stock function body is unchanged (`docs/analysis-vllm-comparison-and-kv-proposals.md` section 8).

### Regenerating

The changes are now committed on the `qml-session` branch, not in the working tree, so the base commit must be
given explicitly (`git diff` without arguments produces an empty result — overwriting with that would erase the whole patch).

```
cd $REPO/third_party/llama.cpp
git diff 0d47ea742 -- \
    CMakeLists.txt common/arg.cpp common/common.h tools/server/ \
    > $REPO/patches/llama.cpp-qml-session.patch
```

After rebasing onto upstream, put the new base (`git merge-base master qml-session`) in place of the base commit,
and update the base commit value in the table above as well.

After regenerating, always check with `git diff --stat` that the size of 7 files, 874 insertions / 4 deletions (1,100 lines) is preserved.

---

## Why only llama.cpp has a patch (there are no vLLM / SGLang patches)

**There is no vLLM patch file, and none is needed.** The three engines differ structurally in how they are extended.

| Engine | Extension method | Patch needed | Location of our code |
|---|---|---|---|
| llama.cpp | **No** official plugin extension point. Handling session state requires access to `server_slot::prompt.tokens` (a C++ object statically compiled into the server binary), which LD_PRELOAD cannot reach in principle | **Needed** (this directory) | patch + `hooks/llama.cpp/qml_llama_hook.cpp` |
| vLLM | `KVConnectorBase_V1` + `kv_connector_module_path` (`distributed/kv_transfer/kv_connector/factory.py`) — imports an external module at runtime | Not needed | `hooks/vllm/qml_vllm_connector.py`, `qml_store_bindings.py` |
| SGLang | `HiCacheStorage` + `--hicache-storage-backend` dynamic backend | Not needed | `hooks/sglang/qml_sglang_storage.py` |

Measured check (2026-07-30): searching the whole installed `vllm` / `sglang` package trees for the string `qml`
returned **0 hits for each** — the installed copies of both engines were not modified by our work at all.
Therefore the vLLM/SGLang integration is reproduced without a patch, using only the files under `hooks/` and runtime configuration
(`--kv-transfer-config`, `--hicache-storage-backend`) (see the
"vLLM connector setup" section of the README).

This asymmetry is also the basis of the project's positioning: the reason existing KV cache
layers such as LMCache/Mooncake support only vLLM/SGLang is exactly that those two engines have extension points and llama.cpp
does not, and filling the gap on the llama.cpp side requires this patch
(`docs/analysis-vllm-comparison-and-kv-proposals.md` sections 6-7).
