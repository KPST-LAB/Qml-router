# Appendix A-E: attributing the gain, layer versus engine-side change (four-build check)

- Model: Llama-3.2-3B-Instruct F16 (`models/Llama-3.2-3B-Instruct-F16.gguf` for llama.cpp, KV cache Q4_0 where the script sets `-ctk q4_0 -ctv q4_0`; `unsloth/Llama-3.2-3B-Instruct` on Hugging Face for vLLM and SGLang)
- Method: none
- Programs: `tests/reconnect-restore/bench_density_ablation.py`

| Script | Paper item | Engine | Hardware | Frozen log |
|---|---|---|---|---|
| `appE_four_build_check.sh` | Appendix A-E, four-build check | llama.cpp (no-patch and with-patch builds) | RTX 4090 | `artifact/paper/logs/appE_four_build_check.log` |

The script expects four engine builds under `$REPO/ablation/<build>/build/bin` and the program `bench_density_ablation.py` in `$REPO/ablation/`.

Numbers are those of the paper; `artifact/README.md` maps every reported number to its log, script and data.
