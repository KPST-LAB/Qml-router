# Section V-B1: the session beyond both limits (20,000-turn session; recall gate at 20,000 and 2,000 turns)

- Model: Llama-3.3-70B-Instruct Q8_0 (`$DATA/models/Llama-3.3-70B-Instruct-Q8_0/.../Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf`), one NVIDIA H200
- Method: B. per-question copy (the finished session is copied once per question; see below)
- Programs: `tests/multi-turn/common/bench_weather_longevity.py` (entry script for this method: `tests/multi-turn/per-question-copy/weather_per_question_copy.sh`)

| Script | Paper item | Engine | Hardware | Frozen log |
|---|---|---|---|---|
| `secVB1_20000turn_session.sh` | Section V-B1, the 20,000-turn session | llama.cpp | H200 | `artifact/paper/logs/secVB1_20000turn_session.log` |
| `secVB1_20000turn_recall.sh` | Section V-B1, recall gate on that session | llama.cpp | H200 | `artifact/paper/logs/secVB1_20000turn_recall.log` |
| `secVB1_20000turn_latency.sh` | Section V-B1, per-question latency split | llama.cpp | H200 | `artifact/paper/logs/secVB1_20000turn_latency.log` |

Further runs (scripts in `artifact/paper/scripts/`; they expect the run-folder layout of the measurement host):

| Script | Paper item |
|---|---|
| `artifact/paper/scripts/reviewer1_edit_4_step1_longevity_slope.py` | Section V-B1 p50/p95 per fifth and regression slope (no new run) |

How the test works (read this before interpreting the numbers):

- `feed` sends the turns to the engine one by one. When the session passes the window, the client's `trim()` function moves the overflow turns to the archive: it rewrites the token log and the archive tables of the store with SQL, keeping the system head and the last turns, after sending `PURGE_CACHE` and `DROP_BLOB` to the engine. This is a client-side implementation of the gateway's turn-aligned trim; the engine's own trim code was not used in these runs.
- `clone` copies the finished session once per question (`clone_session()`), and `ask` asks each question on its own copy (the scripts set `WX_QUESTION_MODE=clone`; the conversation method, where the questions continue the fed conversation, is described in `tests/multi-turn/README.md`). Retrieval and injection of archived turns during `ask` are done by the engine (RIG, Recall Injection Gate).
- The 2,000-turn figures of the same paragraph (21 questions) come from the same program and procedure on a 2,000-turn session with the same model; that run's script is not in `artifact/paper/scripts/`.

The recorded results (540 trims, 27/27 with recall on and 0/27 with recall off, the token counts and latencies) are as logged.

Numbers are those of the paper; `artifact/README.md` maps every reported number to its log, script and data.
