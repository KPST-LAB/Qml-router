# Fig. 1(a)-(b): reconnect latency and short-request throughput as N users reconnect (Section V-A3)

- Model: Llama-3.2-3B-Instruct F16 (`models/Llama-3.2-3B-Instruct-F16.gguf` for llama.cpp, KV cache Q4_0 where the script sets `-ctk q4_0 -ctv q4_0`; `unsloth/Llama-3.2-3B-Instruct` on Hugging Face for vLLM and SGLang)
- Method: none (reconnect timing; each answer is checked)
- Programs: `tests/reconnect-restore/bench_density_curve.py`

| Script | Paper item | Engine | Hardware | Frozen log |
|---|---|---|---|---|
| `fig01_density.sh` | Fig. 1(a)-(b): the conditions of `reviewer1_edit_14_step1_density_raw` below (N = 2, 4, 8, 16, 4 s apart, about 17,650 tokens per user) with this repository's program | llama.cpp | RTX 4090 | `artifact/paper/logs/reviewer1_edit_14_step1_density_raw.log` |

An earlier density run (N = 4 to 64, about 5,200 tokens per user) is not cited and is not in `artifact/`. Before 2026-10-24 `fig01_density.sh` ran those earlier conditions.

Further runs (scripts in `artifact/paper/scripts/`; they expect the run-folder layout of the measurement host):

| Script | Paper item |
|---|---|
| `artifact/paper/scripts/reviewer1_edit_14_step1_density_raw.{sh,py}` | Fig. 1(a)-(b) data with per-reconnect samples, Section V-A concurrency, Table 8 row |
| `artifact/paper/scripts/reviewer2_minor1_density_verified_only.py`, `reviewer2_minor1_plot_fig1.py` | Fig. 1 of v66: latency over passing reconnects only (29/30 qml-router, 30/30 baseline); no new run |

Numbers are those of the paper; `artifact/README.md` maps every reported number to its log, script and data.
