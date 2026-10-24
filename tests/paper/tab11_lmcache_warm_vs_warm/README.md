# Table 11 and Section V-D: identity-based state isolation, warm-versus-warm comparison with LMCache on vLLM

- Model: Llama-3.1-8B-Instruct AWQ-INT4 (`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`) on vLLM 0.26.0, FP16 KV cache, RTX 4090
- Method: none (mixed load; answers checked for content of other sessions)
- Programs: `tests/mixed-load/bench_mixed_load.py`

| Script | Paper item | Engine | Hardware | Frozen log |
|---|---|---|---|---|
| `tab11_lmcache_mp.sh` | Table 11 row 1, Section V-D: LMCache 0.5.2 in multi-process (MP) mode | vLLM 0.26.0 + LMCache | RTX 4090 | `artifact/paper/logs/tab11_lmcache_mp.log` |
| `tab11_qml_connector.sh` | Table 11 row 2: vLLM with the qml-router connector (`hooks/vllm`) | vLLM 0.26.0 | RTX 4090 | `artifact/paper/logs/tab11_qml_connector.log` |

The baseline-vLLM row of Table 11 has no client log; only engine server logs were kept (counts in `docs/4-engines/engine-comparison.md`). The LMCache in-process mode was excluded because warm reconnects returned other sessions' content in that mode (LMCache issue #4385). The scripts write their working files to `$REPO/tests/_lm/`.

Numbers are those of the paper; `artifact/README.md` maps every reported number to its log, script and data.
