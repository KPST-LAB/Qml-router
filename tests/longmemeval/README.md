# longmemeval

LongMemEval_S runs on llama.cpp with the qml-router patch (method C: the benchmark gives each
question its own history; every question is ingested into its own session and asked once).

| Program | Role |
|---|---|
| `bench_longmemeval.py` | ingest, archive recall, ask and judge (modes `ingest`, `ask`, `recallk`, `judge`, `paperjudge`) |
| `lme_fullhist_resume.py` | full-history baseline, same reader |
| `lme_recent_window.py` | recent-window baseline, same reader |

Paper: Fig. 6 and Section V-C (`paper/fig06_longmemeval_archive_recall/`), Table 13
(`paper/tab13_ranker_cost_accuracy/`). The dataset (`longmemeval_s_cleaned.json`) is not
redistributed.
