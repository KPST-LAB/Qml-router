# Table 5 and Fig. 4: context-window overflow over 500 turns, three engines, baseline versus qml-router (Section V-A2)

- Model: Llama-3.2-3B-Instruct F16 (`models/Llama-3.2-3B-Instruct-F16.gguf` for llama.cpp, KV cache Q4_0 where the script sets `-ctk q4_0 -ctv q4_0`; `unsloth/Llama-3.2-3B-Instruct` on Hugging Face for vLLM and SGLang); Qwen3-4B F16 (`models/Qwen3-4B-F16.gguf`), RTX 4090; the size-matched non-Llama check of Section V-A2 (same protocol, only the model changed; chat template `QML_BENCH_TEMPLATE=qwen3`)
- Method: A. continuous session on llama.cpp with qml-router (questions asked in the session, question turns removed afterwards); separate client-built requests on the other engines and configurations (see `tests/multi-turn/README.md`)
- Programs: `tests/multi-turn/continuous-session/bench_tab5_crossover_32k.py`; Fig. 4: `plot_overflow_latency.py` in this folder

| Script | Paper item | Engine | Hardware | Frozen log |
|---|---|---|---|---|
| `tab05_500turn_4090.sh` | Table 5, rows 1-6; Fig. 4 | llama.cpp, vLLM, SGLang | RTX 4090, NVMe | `artifact/paper/logs/tab05_500turn_4090.log` |
| `tab05_500turn_orin.sh` | Table 5, rows 7-8 (MAXN) | llama.cpp | Jetson AGX Orin | `artifact/paper/logs/tab05_500turn_orin.log` |
| `panel_tab5_qwen4b_4090.sh` | Table 5, last two rows (Qwen3-4B, baseline and qml-router) | llama.cpp | RTX 4090 | `artifact/paper/logs/tab05_500turn_qwen4b.log` |
| `chain_qwen4b_4090.sh` | runs `panel_tab5_qwen4b_4090.sh` for baseline and then qml-router | llama.cpp | RTX 4090 | (script only) |

Run the Qwen3-4B adoption panel (Table 4 folder) and this chain one after the other, never at the same time on one GPU.

Numbers are those of the paper; `artifact/README.md` maps every reported number to its log, script and data.
