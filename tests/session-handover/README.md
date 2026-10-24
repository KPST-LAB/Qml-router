# session-handover

One session continued across engines: the token log is carried, not KV, and each engine
re-forwards the history before it continues (vLLM through `hooks/vllm`, SGLang through
`hooks/sglang`, llama.cpp through `hooks/llama.cpp`).

| Program | Path |
|---|---|
| `bench_migration.py` (modes `build_llama`, `continue_vllm`, `continue_llama`) | llama.cpp -> vLLM -> llama.cpp |
| inline Python in `paper/fig05_session_portability/fig05_sglang_to_llama.sh` | SGLang -> llama.cpp |

Paper: Fig. 5 and Section V-B3 (`paper/fig05_session_portability/`).
