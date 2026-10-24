# mixed-load

Long sessions and short stateless requests on one server (`bench_mixed_load.py`, modes
`session`, `stateless`, `build`). Engines: llama.cpp with qml-router, vLLM (with the qml-router
connector or LMCache), SGLang.

`QML_BENCH_SALT=1` (default `0`) makes mode `stateless-openai` send `cache_salt` set to the long
session's id `mix-N` (the id mode `build` stores the session under); see `tests/README.md`.

Paper: Table 6 rows 3-4 (`paper/tab06_repeated_runs/`), Table 11 and Section V-D
(`paper/tab11_lmcache_warm_vs_warm/`).

Chat-format modes (`build-chat`, `session-chat`, `stateless-chat`, added 2026-10-24). In the paper's
raw-text protocol every reconnect adds about 22 tokens to the session, so a session that starts at
32,008 tokens fills a 32,768-token slot after 34 reconnects (six sessions: 204 reconnects) and the
server then refuses it rather than answer without the history. The RTX 4090 runs of Table 6 completed
200 to 205 reconnects in 150 s, at that limit; a faster GPU passes it within the run. The chat modes
build the same sessions as Llama-3 chat turns (user turns of about 900 tokens, the vault code in the
system message), so the server's window trim (`QML_TRIM_WINDOW`, `QML_TRIM_EOT`) can cut a session at a
user-turn boundary; `stateless-chat` keeps the same window on the client. `mixed_chat_tawt.sh` runs
the Table 6 rows 3-4 conditions this way.

