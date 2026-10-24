# Section V-B2: persistence across restarts and reconnects (restart recovery)

- Model: Llama-3.2-3B-Instruct F16 (`models/Llama-3.2-3B-Instruct-F16.gguf` for llama.cpp, KV cache Q4_0 where the script sets `-ctk q4_0 -ctv q4_0`; `unsloth/Llama-3.2-3B-Instruct` on Hugging Face for vLLM and SGLang)
- Method: none
- Programs: inline Python in the script

| Script | Paper item | Engine | Hardware | Frozen log |
|---|---|---|---|---|
| `secVB2_restart_recovery.sh` | Section V-B2, restart recovery (242 ms) | llama.cpp | RTX 4090 | `artifact/paper/logs/secVB2_restart_recovery.log` |

Numbers are those of the paper; `artifact/README.md` maps every reported number to its log, script and data.
