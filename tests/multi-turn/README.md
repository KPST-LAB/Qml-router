# multi-turn

Multi-turn session programs on llama.cpp with the qml-router patch. A session is driven turn
by turn through the engine; when it passes the live window, the turns that overflow are moved
to the archive at a turn boundary (TAWT), and later questions are answered with archive
recall (RIG, Recall Injection Gate; options in `docs/API.md` 5.3).

| Path | Method | Program | Paper |
|---|---|---|---|
| `continuous-session/bench_tab5_crossover_32k.py` | A. continuous session on llama.cpp with qml-router: at turns 100, 300 and 500 every fact that has left the window is asked inside the session (engine-side recall), and the question turns are removed from the session log afterwards. On the other engines and configurations (baseline llama.cpp, vLLM, SGLang) the client sends each question as a separate request built from the live window it holds; in their qml-router configuration the client prepends the archived excerpts it retrieved ("Archived conversation excerpts: ...") | 500-turn overflow, three engines | Table 5, Fig. 4 (`paper/tab05_context_overflow_500turn/`) |
| `continuous-session/weather_continuous_session.sh` | A. conversation: stage `ask` asks every question in order, continuing the fed conversation (`WX_QUESTION_MODE=conversation`; question and answer turns stay, the trim runs as during the feed); stage `ask1` asks one question per planted fact | weather test | not used in the paper |
| `per-question-copy/weather_per_question_copy.sh` | B. per-question copy: the fed session is copied once per question and each question is asked on its own copy (`WX_QUESTION_MODE=clone`) | weather test | Section V-B1 (`paper/secVB1_session_beyond_limits/`) |
| `common/bench_weather_longevity.py` | shared by both weather entry scripts | feed, trim, copy and ask modes; the planted facts and questions | |

The two weather methods use the same program and the same fed session; only the ask stage
differs. `WX_QUESTION_MODE` selects it (`conversation` is the default of the program). The
paper's runs used method B. The paraphrase question about the PM2.5 bulletin reads "How high
did the fine dust (PM2.5) reading go in Seoul at its peak?"; the paper's runs used the wording
without "(PM2.5)" (`artifact/paper/scripts/bin/bench_weather_longevity.py`).

In the weather program the overflow trim is done by the client (`trim()`, SQL on the store,
equivalent to the gateway's trim), not by the engine; see the Section V-B1 README.

## How to run

GitHub Markdown has no tab control, so each run below is a collapsible section. Open the one
you need.

| Run | Turns | Questions | Question method | Paper |
|---|---|---|---|---|
| 20,000-turn session | 20,000 | 27 (9 facts, 3 questions each) | B. per-question copy | Section V-B1 |
| 20,000-turn session, conversation method | 20,000 | 27 | A. conversation | not in the paper |
| 2,000-turn weather test | 2,000 | 21 (7 facts, 3 questions each) | A or B | Section V-B1 (2,000-turn figures, method B) |
| 500-turn context overflow | 500 | 8, 28 and 44-46 at turns 100, 300 and 500 | asked inside the session, then removed | Table 5, Fig. 4 |

Every weather run has two stages. The `feed` stage runs against a patched `llama-server` with
archive recall off. The `ask` stage runs after the server is restarted with recall on.
`QML_ENGINE_PID` must hold the PID of the running server, because the client sends it
control commands. `PORT`, `STORE` (store folder) and `SID` (session id) are your values.
The commands below run from `tests/multi-turn/`.

Server with recall off (feed stage):

```bash
llama-server -m MODEL.gguf -ngl 99 -fa on -c 16384 -np 2 --port PORT \
  --qml-session-store STORE/qml.db --qml-lmdb-store STORE/qml_lmdb
```

Server with recall on (ask stage), with the settings of the paper's runs (`docs/API.md` 5.3):

```bash
QML_ARCHIVE_RECALL=1 QML_ARCHIVE_RECALL_TOPK=10 QML_ARCHIVE_RECALL_BUDGET=1024 \
QML_ARCHIVE_RECALL_PAIR_TURNS=1 \
llama-server -m MODEL.gguf -ngl 99 -fa on -c 16384 -np 2 --port PORT \
  --qml-session-store STORE/qml.db --qml-lmdb-store STORE/qml_lmdb
```

<details>
<summary><b>20,000-turn session (Section V-B1, per-question copy)</b></summary>

The paper's run used Llama-3.3-70B-Instruct Q8_0 on one NVIDIA H200. The facts are planted
at turns 10, 500, 2,000, 5,000, 8,000, 11,000, 14,000, 17,000 and 19,500.

```bash
export WX_N_TURNS=20000
export WX_FACT_TURNS="10,500,2000,5000,8000,11000,14000,17000,19500"
# server with recall off
QML_ENGINE_PID=<pid> per-question-copy/weather_per_question_copy.sh feed PORT STORE SID 20000
# restart the server with recall on
QML_ENGINE_PID=<pid> per-question-copy/weather_per_question_copy.sh ask PORT STORE SID on
python3 common/bench_weather_longevity.py report 0 STORE SID
```

The `ask` stage copies the session once per question (27 copies) and asks each question on
its own copy. The complete script of the paper's run, including the recall-off control
condition, is `tests/paper/secVB1_session_beyond_limits/secVB1_20000turn_session.sh`.

</details>

<details>
<summary><b>20,000-turn session, conversation method</b></summary>

The feed stage is the same as above. The questions then continue the fed conversation in
order, on one copy of it (`SID-conv-on`). Question and answer turns stay in the history, and
the overflow trim runs as during the feed.

```bash
export WX_N_TURNS=20000
export WX_FACT_TURNS="10,500,2000,5000,8000,11000,14000,17000,19500"
# server with recall off
QML_ENGINE_PID=<pid> continuous-session/weather_continuous_session.sh feed PORT STORE SID 20000
# restart the server with recall on
QML_ENGINE_PID=<pid> continuous-session/weather_continuous_session.sh ask PORT STORE SID on
```

A session fed once can be asked both ways: the conversation method works on its own copy
and leaves the fed session unchanged.

</details>

<details>
<summary><b>2,000-turn weather test (21 questions)</b></summary>

Without `WX_FACT_TURNS` the program plants its 7 default facts (21 questions) in 2,000 turns.
The paper's 2,000-turn figures used method B with Llama-3.3-70B-Instruct Q8_0.

```bash
# server with recall off
QML_ENGINE_PID=<pid> continuous-session/weather_continuous_session.sh feed PORT STORE SID 2000
# restart the server with recall on, then one of:
QML_ENGINE_PID=<pid> continuous-session/weather_continuous_session.sh ask PORT STORE SID on   # A. conversation
QML_ENGINE_PID=<pid> per-question-copy/weather_per_question_copy.sh ask PORT STORE SID on     # B. per-question copy
```

</details>

<details>
<summary><b>500-turn context overflow (Table 5, three engines)</b></summary>

The program is `continuous-session/bench_tab5_crossover_32k.py`. The run scripts of the paper
start the servers and run every engine and configuration:
`tests/paper/tab05_context_overflow_500turn/tab05_500turn_4090.sh` (RTX 4090: llama.cpp,
vLLM, SGLang) and `tab05_500turn_orin.sh` (Jetson AGX Orin: llama.cpp). How the trim and
the questions are handled on each engine is in `docs/API.md` 5.7.

</details>
