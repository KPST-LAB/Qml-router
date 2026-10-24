# Table 6: repeated runs, n per scenario under identical conditions (Section V-A3)

- Model: Llama-3.2-3B-Instruct F16 (`models/Llama-3.2-3B-Instruct-F16.gguf` for llama.cpp, KV cache Q4_0 where the script sets `-ctk q4_0 -ctv q4_0`; `unsloth/Llama-3.2-3B-Instruct` on Hugging Face for vLLM and SGLang)
- Method: none
- Programs: `tests/reconnect-restore/bench_crossover.py`, `tests/mixed-load/bench_mixed_load.py`, `tests/reconnect-restore/bench_long_context.py`

| Script | Paper item | Engine | Hardware | Frozen log |
|---|---|---|---|---|
| `tab06_repeat_crossover.sh` | Table 6, rows 1-2 | llama.cpp | RTX 4090 | `artifact/paper/logs/tab06_repeat_crossover.log` |
| `tab06_repeat_mixed.sh` | Table 6, rows 3-4 | llama.cpp | RTX 4090 | `artifact/paper/logs/tab06_repeat_mixed.log` |
| `tab06_tab07_orin_repeat_maxn.sh` | Table 6 row 5, Table 7 row 3 (MAXN) | llama.cpp | Jetson AGX Orin | `artifact/paper/logs/tab06_tab07_orin_repeat_maxn.log` |
| `tab06_tab07_orin_repeat_30w.sh` | Table 6 row 6, Table 7 row 4 (30 W) | llama.cpp | Jetson AGX Orin | `artifact/paper/logs/tab06_tab07_orin_repeat_30w.log` |

Further runs (scripts in `artifact/paper/scripts/`; they expect the run-folder layout of the measurement host):

| Script | Paper item |
|---|---|
| `artifact/paper/scripts/reviewer2_minor3_step1_engine_config.sh`, `reviewer2_minor3_step2_crossover.{sh,py}` | Table 6 caption: same-day re-run of both 32K procedures |

Numbers are those of the paper; `artifact/README.md` maps every reported number to its log, script and data.
