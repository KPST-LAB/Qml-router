<!--
SPDX-License-Identifier: AGPL-3.0-only
Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/banner-dark.svg">
    <img src="docs/images/banner-light.svg" width="760" alt="qml-router: persistent session state for LLM serving on edge and on-premises hardware">
  </picture>
</p>

<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/license-AGPL--3.0%20%2B%20commercial-blue" alt="License: AGPL-3.0 and commercial"></a>
  <a href="#paper"><img src="https://img.shields.io/badge/IEEE%20Access-accepted-2ea44f" alt="Paper: IEEE Access, accepted"></a>
  <img src="https://img.shields.io/badge/engines-llama.cpp%20%7C%20vLLM%20%7C%20SGLang-555" alt="Engines: llama.cpp, vLLM, SGLang">
  <img src="https://img.shields.io/badge/code-after%20final%20publication-lightgrey" alt="Code: after the final publication">
</p>

<p align="center">
  <a href="#paper">Paper</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#results">Results</a> ·
  <a href="#license">License</a> ·
  <a href="mailto:jun@kpst.co.kr">Contact</a>
</p>

# qml-router

qml-router keeps a conversation's state outside the LLM serving process. A session survives
process restarts, a reconnect restores the KV cache instead of recomputing it, history past
the context window is archived and retrieved when a question needs it, and a session can
continue on another engine. It targets resource-constrained hardware: on-premises servers
and edge devices.

The design rests on one invariant: the canonical state of a session is a token log owned by
an explicit identity (`user::session`) and independent of the engine at the storage layer;
each engine's KV cache is a rebuildable acceleration derived from that log.

> [!NOTE]
> The source code (session core, storage, engine integrations for llama.cpp, vLLM and SGLang,
> gateway), the benchmark programs and the paper's experiment artifact are published here after
> the final publication of the paper. Until then this repository holds the description, the
> license and the contribution terms.

## Updates

- **2026-10** — The paper on qml-router is accepted by *IEEE Access* ([Paper](#paper)).
- **2026-10** — This repository opens with the description and the license terms; the code
  follows the final publication.

## At a glance

<table>
  <tr>
    <td align="center" width="33%"><h3>4.6×</h3>faster 32K-token reconnect<br>restore 0.66 s vs recompute 3.03 s<br><sub>RTX 4090, NVMe</sub></td>
    <td align="center" width="33%"><h3>20.7×</h3>faster on an edge device<br>4.2 s vs 87.9 s<br><sub>Jetson AGX Orin, 30 W</sub></td>
    <td align="center" width="33%"><h3>20,000 turns</h3>one session, no material increase<br>in per-turn latency<br><sub>Llama-3.3-70B, H200</sub></td>
  </tr>
</table>

The name "qml" stands for query-based machine learning. It is unrelated to the Qt QML
language and to quantum machine learning.

## Architecture

<p align="center">
  <img src="docs/images/fig3-architecture.png" width="760" alt="llama.cpp, vLLM and SGLang with qml-router attached, saving to and restoring from the qml-router persistence tier (SQLite sessions and token log, LMDB KV payload)">
</p>

The system after qml-router is attached: the reuse scope extends beyond GPU memory and engine
boundaries.

## Comparison with persistent-KV systems

✓ = supported, — = not provided, N/R = the cited work does not report the property (not taken
as evidence of its absence). Cells reflect the cited publications and, for LMCache, its 2026
public documentation.

| System | Canonical state | Identity-keyed session | Restart persistence | Heterogeneous engines | Lossless overflow archive | Selective history retrieval | Process isolation boundary | Encryption at rest | Correctness verification |
|---|---|---|---|---|---|---|---|---|---|
| LMCache | KV bytes (content hash) | prefix-keyed chunks; tenant salt, no session identity | ✓ (standalone daemon) | vLLM, SGLang | — | — | out-of-process daemon (IPC per operation) | ✓ (persistent tier) | N/R |
| KVGov | KV bytes (content hash, principal salt) | prefix-keyed within a principal; no session identity | — | vLLM, llama.cpp | — | — | in-process (engine cache) | N/R | N/R |
| CachedAttention | KV bytes | session-oriented | ✓ (hierarchical store) | single engine | — | — | N/R | N/R | N/R |
| agent-memory | KV bytes (Q4) | prefix-keyed per agent (Q4 KV) | N/R | single runtime (MLX) | — | N/R | in-process (MLX runtime) | N/R | N/R |
| **qml-router** (llama.cpp, vLLM, SGLang) | token record; KV derived | ✓ (`user::session`) | ✓ | ✓ | ✓ (TAWT) | ✓ (RIG) | in-process store (trust boundary = process) | — (roadmap) | ✓ |

## How it works

```text
                            QML ROUTER
┌────────────────────────────────────────────────────────────────┐
│                        Client / Gateway                        │
│ OpenAI-compatible API │ new turn + session_id (user::session)  │
├────────────────────────────────────────────────────────────────┤
│                         Session Layer                          │
│ Identity │ Token Log │ Restore │ TAWT │ RIG (FTS5, BM25)       │
├────────────────────────────────────────────────────────────────┤
│                       Engine Integration                       │
│ llama.cpp server patch + hook                                  │
│ vLLM KVConnector │ SGLang HiCache backend                      │
├────────────────────────────────────────────────────────────────┤
│                             C Core                             │
│ TAWT policy (qml_cache) │ RIG policy (qml_ml)                  │
├────────────────────────────────────────────────────────────────┤
│             Embedded Store (in the engine process)             │
│ SQLite: session │ token log │ archive │ FTS5 index             │
│ LMDB: KV payload                                               │
├────────────────────────────────────────────────────────────────┤
│                            Hardware                            │
│ RTX 4090 │ H200 │ Jetson AGX Orin │ NVMe                       │
└────────────────────────────────────────────────────────────────┘
```

One turn of a session:

```text
Client ── new turn + session_id ──▶ qml-router
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│ 1. Look up the session by its identity                         │
│ 2. Stored KV matches the engine configuration → load it;       │
│    otherwise replay the token log (prefill)                    │
│ 3. RIG: search this session's archive (FTS5), gate the         │
│    candidates, inject the chosen turns before the new turn     │
│ 4. The engine generates the answer                             │
│ 5. Append the turn to the token log, save the KV               │
│ 6. TAWT: past the window, whole turns move to the archive;     │
│    the system prefix stays                                     │
└────────────────────────────────────────────────────────────────┘
```

The canonical state is the token log, owned by the session identity. A KV payload is an
acceleration that can always be rebuilt from it, and a session can continue on another engine
that shares the tokenizer by replaying it.

## What it does

| Function | Description |
|---|---|
| Identity-keyed persistence | Session state is stored and looked up by session identity, not by content hash, so the storage lookup has no cross-session content matching step. |
| Restore instead of recompute | On reconnect, the stored KV payload is loaded into the engine; the token log is the fallback when the payload does not match the engine configuration. |
| Turn-aligned window trim (TAWT) | When a session passes its window, whole turns move to a per-session archive without loss, keeping the system prefix. |
| Retrieval of archived turns (RIG, Recall Injection Gate) | Archived turns are indexed with SQLite FTS5 (BM25); a gate with a relative-confidence threshold and a token budget decides what to inject on a later turn. |
| Cross-engine continuation | A session continues on another engine by replaying the token log followed by prefill, given a shared tokenizer and a compatible model configuration. |

Storage is embedded in the serving process: SQLite for sessions, token log, archive and
control commands; LMDB for KV payloads. There is no separate service process.

## Results

Measured with the engine's restore path modified as described in the paper.

| Measurement | Result |
|---|---|
| 32K-token reconnect, RTX 4090 | 0.66 s restore vs 3.03 s recomputation |
| 32K-token reconnect, Jetson AGX Orin at 30 W | 4.2 s vs 87.9 s (20.7x) |
| 20,000-turn session (Llama-3.3-70B, H200) | no material increase in per-turn latency |
| Cross-engine continuation | llama.cpp, vLLM and SGLang, by token-log replay followed by prefill |

Regimes where recomputation is as fast or faster (short histories on a fast GPU) are
reported in the paper.

### KV restoration

All rows use Llama-3.2-3B-Instruct unless the row names another model; every reconnect answer is checked against a value planted
in its own history. Baseline = the same engine without qml-router.

<p align="center">
  <img src="docs/images/fig1ab-reconnect-density.png" width="34%" alt="Reconnect latency and short-request throughput as N users reconnect">
  <img src="docs/images/fig1c-reconnect-vs-history.png" width="64%" alt="Single-user reconnect latency versus history length">
</p>

Left: reconnect latency and short-request throughput as N users reconnect at once (latency over
passing answers only: 29/30 qml-router, 30/30 baseline; throughput not answer-checked). Right:
one user's reconnect latency versus history length; restoration stays near-flat.

**Reconnect latency (llama.cpp)**

| Condition | Baseline | qml-router | Ratio |
|---|---|---|---|
| 16 users reconnect at once, RTX 4090: slowest / median reconnect | 59.9 s / 10.0 s | 0.51 s / 0.49 s | — |
| Same run: short-request throughput | 1.0 req/s | 29.0 req/s | 29x |
| One user, 49K-token history, RTX 4090 | 14.5 s | 1.3 s | 11x |
| 32K-token history, RTX 4090, 10 runs (median) | 5.433 s | 0.760 s | 7.1x |
| 49K-token history, RTX 4090, 10 runs (median) | 14.400 s | 1.329 s | 10.8x |

**Restore versus recomputation by storage and device (32K-token history)**

| Environment | Recompute | Restore | Ratio |
|---|---|---|---|
| RTX 4090, storage 0.55 GB/s (container) | 3.16 s | 2.24 s | 1.4x |
| Same GPU, NVMe 1.6-1.9 GB/s (host) | 3.03 s | 0.66 s | 4.6x |
| Jetson AGX Orin, MAXN | 23.93 s | 2.26 s | 10.6x |
| Jetson AGX Orin, 30 W | 87.87 s | 4.24 s | 20.7x |

**Cost of adding qml-router (RTX 4090, 12 sessions per run, 3 runs)**

| Engine | GPU memory | Latency per request |
|---|---|---|
| llama.cpp | +0 MiB | 225 -> 252 ms (+27) |
| vLLM | +0 MiB | 180 -> 259 ms (+79) |
| SGLang | +47 MiB | 166 -> 185 ms (+19) |

**500-turn session past a 32K-token window: facts answered at turn 500**

<p align="center">
  <img src="docs/images/overflow-latency.png" width="560" alt="Per-turn p50 latency at turn 500, log scale, baseline and qml-router on five engine and device pairs, with the evicted facts answered">
</p>

Per-turn p50 latency at turn 500 (log scale); bar labels give the evicted facts answered.

| Engine and device | Baseline | qml-router | Throughput (turns/s), baseline -> qml-router |
|---|---|---|---|
| llama.cpp, RTX 4090 | 0/46 | 42/44 | 0.42 -> 0.49 |
| vLLM, RTX 4090 | 0/46 | 46/46 | 3.95 -> 1.49 |
| SGLang, RTX 4090 | 0/46 | 40/46 | 4.23 -> 2.53 |
| llama.cpp, Jetson AGX Orin | 0/46 | 42/44 | 0.06 -> 0.13 |
| llama.cpp, RTX 4090, Qwen3-4B | 0/46 | 46/48 | 0.32 -> 0.44 |

On vLLM and SGLang the baseline drops the history that passed the window and stays faster;
qml-router keeps the history and answers the questions about it.

## Models and hardware used in the experiments

<details>
<summary>Every model run in the measurements, by device, including experiments that are not in the paper (GGUF models on llama.cpp; HF models on vLLM or SGLang)</summary>

| Device | Model (precision, format) | Engine | Experiments | In the paper |
|---|---|---|---|---|
| NVIDIA H200 (143 GB) | Llama-3.3-70B-Instruct Q8_0, GGUF | llama.cpp + qml-router | 20,000-turn session (27 questions); 2,000-turn weather recall (21 questions); LongMemEval_S archive recall (reader and judge); ranker accuracy; needle recall in a continuous session (90 questions); load tests | Section V-B1, Fig. 6, Table 13 (needle recall and load tests: no) |
| NVIDIA H200 | Llama-3.3-70B-Instruct FP8-dynamic, HF | vLLM 0.24.0, unmodified (retrieval in the client) | 2,000-turn weather recall (21/21); 20,000-turn session (27/27, stock 0/27); needle recall (88/90) | No |
| NVIDIA H200 | Llama-3.1-8B-Instruct BF16, HF and GGUF | llama.cpp, vLLM | Engine timing on identical weights; 20,000-turn session (llama.cpp 25/27, vLLM 27/27) | No |
| NVIDIA H200 | Llama-3.1-8B-Instruct Q8_0 and Llama-3.2-3B-Instruct F16, GGUF | llama.cpp + qml-router | 2,000-turn weather recall by reader size (3B, 8B, 70B) | No |
| NVIDIA H200 | Qwen3-32B Q8_0, GGUF | llama.cpp + qml-router | Adoption cost; 500-turn context overflow | No (removed from the paper) |
| NVIDIA H200 | bge-m3 F16, GGUF | llama.cpp (embedding) | Embedding retrieval and ranker latency | Fig. 6(a), Table 13 |
| RTX 4090 (24 GB) | Llama-3.2-3B-Instruct F16, GGUF, Q4_0 KV | llama.cpp + qml-router | Reconnect density and crossover; adoption cost; 500-turn context overflow; repeated runs; restore/recompute boundary; persistence window; layer-versus-engine attribution; write volume; 2,000-turn weather recall checks | Fig. 1, Tables 4-7, 14, Appendix A-E (weather recall checks: no) |
| RTX 4090 | Llama-3.2-3B-Instruct FP16, HF | vLLM 0.26.0, SGLang 0.5.16 | Adoption cost; 500-turn context overflow; mixed workload; cross-engine continuation; multi-turn session load (96 turns) | Tables 4, 5, 10, Fig. 5 (load: no) |
| RTX 4090 | Llama-3.1-8B-Instruct AWQ-INT4, HF | vLLM 0.26.0 (qml-router connector, LMCache 0.5.2), SGLang 0.5.16 | Identity-keyed versus content-keyed reuse (LMCache); mixed workload | Tables 10, 11 |
| RTX 4090 | Llama-3.1-8B-Instruct Q4_K_M, GGUF | llama.cpp | Mixed workload; restart resume | Table 10 (resume: no) |
| RTX 4090 | Llama-3.1-8B-Instruct Q8_0, GGUF | llama.cpp + qml-router | 20,000-turn session (25/27; 8/9 in one continuous session); retrieval selection comparisons | No |
| RTX 4090 | Qwen3-4B F16, GGUF, Q4_0 KV | llama.cpp + qml-router | Size-matched non-Llama check (adoption cost, 500-turn context overflow) | Section V-A2, Table 5 |
| RTX 4090 | gemma-3-4b-it, gemma-3-12b | vLLM 0.26.0, SGLang 0.5.16, llama.cpp | Early restart and resume tests; sliding-window attention checks | No |
| RTX 4090 | bge-m3 F16, GGUF | llama.cpp (embedding) | Ranker latency on GPU and CPU; lexical versus embedding retrieval | Table 13 |
| Jetson AGX Orin | Llama-3.2-3B-Instruct F16, GGUF, Q4_0 KV | llama.cpp + qml-router | Reconnect crossover at MAXN and 30 W; 500-turn context overflow; adoption cost | Abstract, Tables 5-7 (adoption cost: no) |
| NVIDIA B200 (183 GB) | Qwen3-30B-A3B (GGUF Q3_K_M, Q4_K, Q8; BF16 on SGLang 0.5.20), gpt-oss-20b, gpt-oss-120b (also on vLLM 0.30.0), Qwen3.5-35B-A3B, Llama-3.1-8B-Instruct Q8_0, Llama-3.2-3B-Instruct F16, Qwen3-32B Q8_0, a 70B Q4_K_M model; architecture checks on Qwen3.5-0.8B/4B, Gemma 3/4, DeepSeek-V2-Lite | qml-engine (a separate engine, not in this repository) | 20,000-turn memory stress test (500 questions); 32K restore and time to first token; 27-question weather recall | No |

</details>

## Engine integration

| Engine | Version used in the paper | Integration |
|---|---|---|
| llama.cpp | build b10141 (upstream commit `0d47ea742`) | Server patch plus a hook (llama.cpp has no storage extension point) |
| vLLM | 0.26.0 | `KVConnector` plug-in, no engine-source modification |
| SGLang | 0.5.16 | `HiCacheStorage` backend, no engine-source modification |
| LMCache (comparison only) | 0.5.2 | Used as the content-keyed baseline in Section V-D |

## Security assumptions

The isolation results hold under two assumptions stated in the paper (Section VI-A):
identities are assigned by a trusted gateway, and the serving process runs inside a
trusted host boundary, because the embedded store is readable by any code in the engine
process. The identity key is not an authentication or authorization mechanism. See
[SECURITY.md](SECURITY.md).

## Scope

Cross-engine results assume engines that share a tokenizer vocabulary and a single host.
Network filesystems and rotational media were not tested. The paper does not propose a
new eviction algorithm, live KV-byte migration between engines, or a long-term-memory
reasoning system.

## License

qml-router is dual-licensed:

- **Open source:** GNU Affero General Public License, version 3 only (AGPL-3.0-only); see
  [LICENSE](LICENSE). If you modify qml-router and offer it to users over a network, the
  AGPL requires you to make your modified source available to those users.
- **Commercial:** a separate commercial license from Korea Platform Service Technology
  Co., Inc. for use that does not follow the AGPL terms (for example, a closed-source
  product or service); see [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).

Third-party notices are in [NOTICE](NOTICE). Contributions are accepted only with a
contributor license agreement (CLA); see [CONTRIBUTING.md](CONTRIBUTING.md). Security
reports: see [SECURITY.md](SECURITY.md).

## Paper

> "Persistent Session-State Management for LLM Serving: An Identity-Keyed Architecture for
> Durable, Engine-Agnostic Sessions," *IEEE Access*, 2026 (accepted; manuscript ID
> Access-2026-45930).

Please cite the paper above. The DOI and a BibTeX entry will be added when the article is
published.

## Contact

jun@kpst.co.kr (Korea Platform Service Technology Co., Inc.) — questions, commercial license,
security reports ([SECURITY.md](SECURITY.md)).
