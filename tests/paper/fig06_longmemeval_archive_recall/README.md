# Fig. 6 and Section V-C: selective history retrieval, archive recall on LongMemEval

- Model: Llama-3.3-70B-Instruct Q8_0 (`$DATA/models/Llama-3.3-70B-Instruct-Q8_0/.../Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf`), one NVIDIA H200; embedding retrieval runs use `bge-m3-f16.gguf` on a separate llama.cpp embedding server
- Method: C. per-question history (each question has its own history in the benchmark)
- Programs: `tests/longmemeval/bench_longmemeval.py`, `lme_fullhist_resume.py`, `lme_recent_window.py`

| Script | Paper item | Engine | Hardware | Frozen log |
|---|---|---|---|---|
| `fig06_longmemeval_archive_recall.sh` | Fig. 6, Section V-C: LongMemEval_S with archive recall, k=32 | llama.cpp | H200 | `artifact/paper/logs/fig06_longmemeval_archive_recall.log` |
| `fig06_longmemeval_full_history_chain.sh` | Fig. 6(b), "Full history, same reader" | llama.cpp | H200 | `artifact/paper/logs/fig06_longmemeval_full_history.log` |
| `fig06_longmemeval_recent_window_chain.sh` | Fig. 6(b), recent-window rows | llama.cpp | H200 | `artifact/paper/logs/fig06_longmemeval_recent_window.log` |

Further runs (scripts in `artifact/paper/scripts/`; they expect the run-folder layout of the measurement host):

| Script | Paper item |
|---|---|
| `artifact/paper/scripts/reviewer1_edit_17_step1_retrieval_curve.{sh,py}` | Fig. 6(a) embedding curve, Section V-C recall figures |
| `artifact/paper/scripts/reviewer1_edit_16_step1_oracle_context.py` | Section V-C oracle-context sentence |

The LongMemEval data (`longmemeval_s_cleaned.json`) is not redistributed; the 13 files in `artifact/paper/data/longmemeval` are the inputs and outputs of the runs.

Numbers are those of the paper; `artifact/README.md` maps every reported number to its log, script and data.
