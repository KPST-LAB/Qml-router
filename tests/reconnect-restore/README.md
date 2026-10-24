# reconnect-restore

Reconnect latency, restore versus recompute, and reconnect density on llama.cpp (no questions
about the history; correctness is checked per answer where the program says so).

| Program | Paper |
|---|---|
| `bench_density_curve.py` | Fig. 1(a)-(b) (`paper/fig01ab_reconnect_density/`) |
| `bench_crossover.py` | Fig. 1(c), Table 6 rows 1-2 (`paper/fig01c_reconnect_vs_history_length/`, `paper/tab06_repeated_runs/`) |
| `bench_long_context.py` | Table 7, Table 6 rows 5-6 (`paper/tab07_restore_recompute_boundary/`, `paper/tab06_repeated_runs/`) |
| `bench_density_ablation.py` | Appendix A-E (`paper/appE_layer_vs_engine_attribution/`) |
