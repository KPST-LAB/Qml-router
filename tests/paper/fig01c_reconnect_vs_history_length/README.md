# Fig. 1(c): single-user reconnect latency versus history length (Section V-A)

- Model: Llama-3.2-3B-Instruct F16 (`models/Llama-3.2-3B-Instruct-F16.gguf` for llama.cpp, KV cache Q4_0 where the script sets `-ctk q4_0 -ctv q4_0`; `unsloth/Llama-3.2-3B-Instruct` on Hugging Face for vLLM and SGLang)
- Method: none
- Programs: `tests/reconnect-restore/bench_crossover.py`

| Script | Paper item | Engine | Hardware | Frozen log |
|---|---|---|---|---|
| `fig01_crossover.sh` | Fig. 1(c) | llama.cpp | RTX 4090 | `artifact/paper/logs/fig01_crossover.json` |

Numbers are those of the paper; `artifact/README.md` maps every reported number to its log, script and data.
