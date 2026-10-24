# paper

One folder per experiment of the paper, named after the
table, figure or section it produced. Each README gives the model, engine, hardware, the
method used for questions about the history (A, B, C in `../README.md`), the run scripts and
the frozen logs in `artifact/`.

| Folder | Paper item | Method |
|---|---|---|
| `fig01ab_reconnect_density/` | Fig. 1(a)-(b): reconnect latency and short-request throughput as N users reconnect (Section V-A3) | none |
| `fig01c_reconnect_vs_history_length/` | Fig. 1(c): single-user reconnect latency versus history length (Section V-A) | none |
| `tab04_adoption_cost/` | Table 4: adoption cost per engine, baseline engine -> engine + qml-router (Section V-A1) | none |
| `tab05_context_overflow_500turn/` | Table 5 and Fig. 4: context-window overflow over 500 turns, three engines, baseline versus qml-router (Section V-A2) | A. continuous session on llama.cpp with qml-router |
| `tab06_repeated_runs/` | Table 6: repeated runs, n per scenario under identical conditions (Section V-A3) | none |
| `tab07_restore_recompute_boundary/` | Table 7: the restore/recompute boundary across hardware, 32K reconnect (Section V-A4) | none |
| `secVB1_session_beyond_limits/` | Section V-B1: the session beyond both limits (20,000-turn session; recall gate at 20,000 and 2,000 turns) | B. per-question copy |
| `secVB2_persistence_across_restarts/` | Section V-B2: persistence across restarts and reconnects (restart recovery) | none |
| `fig05_session_portability/` | Fig. 5: engine-agnostic session portability, token-log replay (Section V-B3) | none |
| `fig06_longmemeval_archive_recall/` | Fig. 6 and Section V-C: selective history retrieval, archive recall on LongMemEval | C. per-question history |
| `tab11_lmcache_warm_vs_warm/` | Table 11 and Section V-D: identity-based state isolation, warm-versus-warm comparison with LMCache on vLLM | none |
| `appE_layer_vs_engine_attribution/` | Appendix A-E: attributing the gain, layer versus engine-side change (four-build check) | none |
| `tab13_ranker_cost_accuracy/` | Table 13 and Appendix A-G: ranker cost and end-to-end accuracy | C. per-question history |
| `tab14_persistence_window/` | Table 14 and Appendix A-H: persistence window and loss under back-to-back turns | none |
| `secIIIB_shared_prefix_storage/` | Section III-B: storage added by a shared prefix (identity-keyed lookup) | none |
| `figures/` | `plot_paper_figs.py`: Fig. 1 in the earlier layout; the paper's Fig. 1 is drawn by `artifact/paper/scripts/reviewer2_minor1_plot_fig1.py` | |

