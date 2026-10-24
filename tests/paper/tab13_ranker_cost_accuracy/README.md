# Table 13 and Appendix A-G: ranker cost and end-to-end accuracy

- Model: Llama-3.3-70B-Instruct Q8_0 (`$DATA/models/Llama-3.3-70B-Instruct-Q8_0/.../Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf`), one NVIDIA H200; cost rows also on RTX 4090
- Method: C. per-question history (LongMemEval)
- Programs: scripts in `artifact/paper/scripts/` only (below); `tests/longmemeval/` holds the base programs

Further runs (scripts in `artifact/paper/scripts/`; they expect the run-folder layout of the measurement host):

| Script | Paper item |
|---|---|
| `artifact/paper/scripts/reviewer1_edit_17_step2_*` | Table 13 cost rows (H200 and RTX 4090) |
| `artifact/paper/scripts/reviewer1_edit_17_step3_*`, `reviewer1_edit_17_step3b_lex_rerun_h200.sh` | Table 13 accuracy rows, Section V-C |

Numbers are those of the paper; `artifact/README.md` maps every reported number to its log, script and data.
