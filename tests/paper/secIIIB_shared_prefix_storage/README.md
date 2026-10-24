# Section III-B: storage added by a shared prefix (identity-keyed lookup)

- Model: Llama-3.2-3B-Instruct F16 (`models/Llama-3.2-3B-Instruct-F16.gguf` for llama.cpp, KV cache Q4_0 where the script sets `-ctk q4_0 -ctv q4_0`; `unsloth/Llama-3.2-3B-Instruct` on Hugging Face for vLLM and SGLang)
- Method: none
- Programs: script in `artifact/paper/scripts/` only (below)

Further runs (scripts in `artifact/paper/scripts/`; they expect the run-folder layout of the measurement host):

| Script | Paper item |
|---|---|
| `artifact/paper/scripts/reviewer1_edit_10_step1_shared_prefix_storage.{sh,py}` | Section III-B shared-prefix check (3,916 prompt tokens, 1.27 GB) |

Numbers are those of the paper; `artifact/README.md` maps every reported number to its log, script and data.
