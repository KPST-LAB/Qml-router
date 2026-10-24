# Table 7: the restore/recompute boundary across hardware, 32K reconnect (Section V-A4)

- Model: Llama-3.2-3B-Instruct F16 (`models/Llama-3.2-3B-Instruct-F16.gguf` for llama.cpp, KV cache Q4_0 where the script sets `-ctk q4_0 -ctv q4_0`; `unsloth/Llama-3.2-3B-Instruct` on Hugging Face for vLLM and SGLang)
- Method: none
- Programs: `tests/reconnect-restore/bench_long_context.py`

| Script | Paper item | Engine | Hardware | Frozen log |
|---|---|---|---|---|
| `tab07_hw_devserver.sh` | Table 7 row 1 (same GPU, 0.55 GB/s storage) | llama.cpp | RTX 4090, container overlay | `artifact/paper/logs/tab07_hw_devserver.log` |
| `tab07_hw_nvme.sh` | Table 7 row 2 (same GPU, NVMe) | llama.cpp | RTX 4090, NVMe | `artifact/paper/logs/tab07_hw_nvme.log` |

Table 7 rows 3-4 (Jetson AGX Orin) come from the scripts in `../tab06_repeated_runs/`.

Numbers are those of the paper; `artifact/README.md` maps every reported number to its log, script and data.
