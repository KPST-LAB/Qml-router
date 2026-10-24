# Table 14 and Appendix A-H: persistence window and loss under back-to-back turns

- Model: Llama-3.2-3B-Instruct F16 (`models/Llama-3.2-3B-Instruct-F16.gguf` for llama.cpp, KV cache Q4_0 where the script sets `-ctk q4_0 -ctv q4_0`; `unsloth/Llama-3.2-3B-Instruct` on Hugging Face for vLLM and SGLang)
- Method: none
- Programs: scripts in `artifact/paper/scripts/` only (below)

Further runs (scripts in `artifact/paper/scripts/`; they expect the run-folder layout of the measurement host):

| Script | Paper item |
|---|---|
| `artifact/paper/scripts/reviewer1_edit_18_step1_durability_window.{sh,py}` | Section III-B persistence window, Table 14 |
| `artifact/paper/scripts/reviewer1_edit_18_step2_burst_kill.{sh,py}` | Table 14, burst and kill |

Numbers are those of the paper; `artifact/README.md` maps every reported number to its log, script and data.
