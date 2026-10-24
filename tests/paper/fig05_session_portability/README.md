# Fig. 5: engine-agnostic session portability, token-log replay (Section V-B3)

- Model: Llama-3.2-3B-Instruct F16 (`models/Llama-3.2-3B-Instruct-F16.gguf` for llama.cpp, KV cache Q4_0 where the script sets `-ctk q4_0 -ctv q4_0`; `unsloth/Llama-3.2-3B-Instruct` on Hugging Face for vLLM and SGLang)
- Method: none (correctness check of one continued session)
- Programs: `tests/session-handover/bench_migration.py`; inline Python for SGLang -> llama.cpp

| Script | Paper item | Engine | Hardware | Frozen log |
|---|---|---|---|---|
| `fig05_continuation_vllm.sh` | Fig. 5, bars 1-3 (llama.cpp -> vLLM -> llama.cpp) | llama.cpp, vLLM | RTX 4090 | `artifact/paper/logs/fig05_continuation_vllm.log` |
| `fig05_sglang_to_llama.sh` | Fig. 5, bar 4 (SGLang -> llama.cpp) | SGLang, llama.cpp | RTX 4090 | `artifact/paper/logs/fig05_sglang_to_llama.log` |

Numbers are those of the paper; `artifact/README.md` maps every reported number to its log, script and data.

Both scripts were reconstructed from the paper's logs because the original run scripts were not kept. On 2026-10-24 their engine commands were corrected to the ones recorded in those logs: vLLM with `QmlVllmConnector` (module `qml_vllm_connector`, store paths in `kv_connector_extra_config`, port 8331, `--max-model-len 10240`), SGLang with the `dynamic` HiCache backend `qml_sglang_storage` (port 8333, `--context-length 8192`, write-through) and the session identity sent as `extra_key`, and llama.cpp for the SGLang handover on port 8398 with `-c 8192 -np 1`. The earlier reconstruction named a connector and a backend that do not exist.
