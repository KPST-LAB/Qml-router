# Table 4: adoption cost per engine, baseline engine -> engine + qml-router (Section V-A1)

- Model: Llama-3.2-3B-Instruct F16 (`models/Llama-3.2-3B-Instruct-F16.gguf` for llama.cpp, KV cache Q4_0 where the script sets `-ctk q4_0 -ctv q4_0`; `unsloth/Llama-3.2-3B-Instruct` on Hugging Face for vLLM and SGLang); Qwen3-4B F16 for the size-matched check
- Method: none (fixed client workload)
- Programs: `tests/adoption-overhead/bench_overhead_client.py`

| Script | Paper item | Engine | Hardware | Frozen log |
|---|---|---|---|---|
| `tab04_adoption_llama.sh` | Table 4, llama.cpp row | llama.cpp | RTX 4090 | `artifact/paper/logs/tab04_adoption_llama.log` |
| `tab04_adoption_vllm_sglang.sh` | Table 4, vLLM and SGLang rows | vLLM 0.26.0, SGLang 0.5.16 | RTX 4090 | `artifact/paper/logs/tab04_adoption_vllm_sglang.log` |
| `tab04_adoption_orin.sh` | Table 4 protocol on the Jetson AGX Orin (not in the paper's Table 4, which has no Orin row) | llama.cpp | Jetson AGX Orin | `artifact/paper/logs/tab04_adoption_orin.log` |
| `panel_adoption_qwen4b_4090.sh` | Table 4 protocol on Qwen3-4B (Section V-A2) | llama.cpp | RTX 4090 | `artifact/paper/logs/tab04_adoption_qwen4b.log` |

The Qwen3-4B script is not in `artifact/paper/scripts/`; it differs from the one that produced the log only in paths, comment wording and one console message.

Numbers are those of the paper; `artifact/README.md` maps every reported number to its log, script and data.
