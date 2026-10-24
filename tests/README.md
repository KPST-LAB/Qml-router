# tests

Benchmark programs and run scripts behind the tables and figures of the paper named in the
top-level README. The paper's Appendix A refers to this folder.

## Layout

| Path | Contents |
|---|---|
| `paper/<item>/` | Every experiment of the paper, one folder per table, figure or section, named after it (`tab05_context_overflow_500turn`, `secVB1_session_beyond_limits`, ...). Each folder has the run scripts, a README with model, engine, hardware and method, and the location of the frozen log in `artifact/`. Index: `paper/README.md`. |
| `multi-turn/` | Multi-turn session programs on llama.cpp, split by method (below). |
| `longmemeval/` | LongMemEval programs (method C). |
| `reconnect-restore/` | Reconnect, restore and density programs (no questions). |
| `mixed-load/` | Mixed long-session and short-request load (llama.cpp, vLLM, SGLang). |
| `adoption-overhead/` | Per-request cost of adding qml-router to an engine (llama.cpp, vLLM, SGLang). |
| `session-handover/` | One session handed over between engines (llama.cpp, vLLM, SGLang). |
| `unit/` | Unit tests that need no GPU, engine or database. |

The engine integrations themselves are in `hooks/llama.cpp` (with `patches/`), `hooks/vllm` and
`hooks/sglang`. vLLM and SGLang are installed with pip (`requirements-engines.txt`); only
llama.cpp is checked out under `third_party/`.

## Methods for questions asked about a session's history

| Method | What it does | Where |
|---|---|---|
| A. continuous session | Questions are asked in the session itself. The question turns are removed from the session log after the probe, or only one question is asked per planted fact, so an earlier answer cannot answer a later question. | `multi-turn/continuous-session/` (Table 5 on llama.cpp with qml-router; weather test, mode `ask1`) |
| B. per-question copy | The session is copied once per question and each question is asked on its own copy (each question sees the same fixed history, like the "golden context" of multi-turn benchmarks). | `multi-turn/per-question-copy/` (Section V-B1) |
| C. per-question history | The benchmark gives every question its own history (LongMemEval). | `longmemeval/` |

Every paper folder states which method its runs used.

## Relation to `artifact/`

`artifact/paper/` holds the logs, data and scripts behind the paper's numbers; its SHA-256
manifest covers every file, so nothing in it is edited. The files here are working copies:

- The programs are the copies from `artifact/paper/scripts/bin`. Compared with the originals
  that produced the logs, only comments, docstrings and console message text differ; the
  program logic is the same (Python AST compared with string literals and docstrings ignored).
  Exceptions (2026-10-24): `reconnect-restore/bench_crossover.py`, `bench_density_curve.py` and
  `bench_density_ablation.py` start llama-server as a child process and stop only that process
  group (they used to stop every llama-server by name), and take the server, model, store,
  environment, port, log and result paths from the `QML_BENCH_*` settings below, with defaults
  equal to the earlier fixed values; `mixed-load/bench_mixed_load.py` has the option
  `QML_BENCH_SALT`, off by default. The server flags and the measurement steps are unchanged.
  `reconnect-restore/bench_crossover.py` takes its store and engine paths from `$REPO`, and
  `paper/tab05_context_overflow_500turn/plot_overflow_latency.py` reads the logs from
  `$REPO/artifact/paper/logs/`.
- The run scripts are copies of `artifact/paper/scripts/*.sh` with the program paths changed
  to this layout. In the two LongMemEval chain scripts the H200 data path is replaced by
  `$DATA`. The Qwen3-4B scripts are not in `artifact/paper/scripts/`; they come from the repository with
  internal paths replaced by `$REPO` and `$BENCH`.
- `multi-turn/*/weather_*.sh` are entry scripts added for this layout; they only call the
  modes of `multi-turn/common/bench_weather_longevity.py`.
- The scripts of the later analyses in `artifact/paper/scripts/` are not copied here; they
  expect the run-folder layout of the measurement host. The paper folders list them.

Several run scripts were reconstructed on 2026-09-07 from their logs because the original
one-line invocations were not kept. Their header says so (`#Purpose : reconstructed ...`).

Earlier layouts of this folder (2026-10-22) kept the programs at `tests/<name>.py` and the run
scripts in one folder per model; the file names are unchanged.

## Environment variables used by the run scripts

| Variable | Meaning |
|---|---|
| `REPO` | Repository root (the scripts run `cd $REPO` and use `models/`, `third_party/llama.cpp/build/bin`; `bench_crossover.py` uses the seeded store `$REPO/store-tplseed`) |
| `BENCH` | Scratch folder for stores and results on the storage medium under test |
| `DATA` | Model and dataset root on the H200 host |
| `NVME_MOUNT` | Mount point of the NVMe drive on the RTX 4090 host |
| `QML_ENGINE_PID` | PID of the llama-server the multi-turn programs send control commands to |

The run scripts start every server with `setsid`, record its PID (the process group) and stop
only that group (`kill -- -PID`, SIGKILL after 30 s, or SIGKILL at once where the script used
SIGKILL before). They never stop a process by name, so another project's servers under the same
user account are not touched. `tab06_repeat_crossover.sh` and `tab06_repeat_mixed.sh` no longer
restart the services on `store-tplseed` at the end.

Settings of `reconnect-restore/bench_crossover.py`, `bench_density_curve.py` and
`bench_density_ablation.py` (each program starts and stops its own llama-server):

| Variable | Meaning | Default |
|---|---|---|
| `QML_BENCH_SERVER` | llama-server binary | `$REPO/third_party/llama.cpp/build/bin/llama-server` (`bench_density_ablation.py`: `DENS_BIN` first) |
| `QML_BENCH_MODEL` | Model file | `$REPO/models/Llama-3.2-3B-Instruct-F16.gguf` |
| `QML_BENCH_STORE` | Store folder; the server uses `qml.db` and `qml_lmdb` in it, and the program reads the same `qml.db` | `$REPO/store-tplseed` |
| `QML_BENCH_ENV` | Extra environment assignments for the server, separated by spaces (`$VAR` is expanded) | `LD_LIBRARY_PATH=$REPO/env/lib:/usr/local/cuda/lib64` |
| `QML_BENCH_PORT` | Server port; the program stops with an error if another process already answers on it | `8397` (`bench_density_ablation.py`: `8399`) |
| `QML_BENCH_LOG` | Server log (overwritten at every start) | `$REPO/srv_density.log` |
| `QML_BENCH_RESULTS` | Result JSON | `/tmp/crossover_results.json`; density programs: `DENS_RESULTS`, then `/tmp/density_results.json` |

`mixed-load/bench_mixed_load.py`: `QML_BENCH_SALT=1` makes mode `stateless-openai` send
`cache_salt` set to the long session's id (`mix-N`, the id that mode `build` stores the session
under). The build requests in `paper/tab11_lmcache_warm_vs_warm/*.sh` send the same value when the
variable is set. Default `0`: the requests are unchanged.

Working folders that scripts create under `tests/` (names starting with `_`) are ignored by git.

## Running a paper experiment

1. Build llama.cpp with the qml-router patch: `scripts/setup_llama_cpp.sh`.
2. Put the model file under `$REPO/models/` (file names are in each paper folder README).
3. Set the variables above and run the script, for example
   `REPO=$PWD BENCH=/mnt/nvme/bench bash tests/paper/tab05_context_overflow_500turn/tab05_500turn_4090.sh llama qml`.
