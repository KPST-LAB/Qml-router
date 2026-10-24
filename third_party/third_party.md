# third_party: engine checkouts

Engine checkouts in this folder are ignored by git; only this file is tracked.

| Path | Source | Role |
|---|---|---|
| `llama.cpp/` | `https://github.com/ggml-org/llama.cpp` at upstream commit `0d47ea742` (build b10141) | Build target. `scripts/setup_llama_cpp.sh` clones it, applies `patches/llama.cpp-qml-session.patch` and builds `llama-server`. The patch equals the qml-router fork commit `ffd4ec446` named in the paper (Table 15). |
| vLLM, SGLang | installed with pip (`requirements-engines.txt`) | No source changes; the integrations are `hooks/vllm/` and `hooks/sglang/`. |

## Building llama.cpp by hand

```
cd third_party/llama.cpp
git checkout 0d47ea742
git apply ../../patches/llama.cpp-qml-session.patch
cmake -S . -B build -DLLAMA_QML_SESSION=ON -DQML_ROUTER_DIR=$REPO -DGGML_CUDA=ON
cmake --build build --target llama-server -j
```

The run scripts in `tests/` expect the binary at `third_party/llama.cpp/build/bin/llama-server`.
