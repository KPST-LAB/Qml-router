#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
"""bench_weather_gateway.py

Purpose : the weather multi-turn test through the qml-router session layer for
          OpenAI-compatible engines (router/qml_openai_gateway.py in front of vLLM
          with the qml connector, or SGLang with the qml backend).
          The scenario (planted facts, daily reports, questions) is read from
          bench_weather_longevity.py in this folder, so both programs ask the
          same questions about the same conversation.
Modes   : feed GW STORE SID N_TURNS   feed N_TURNS turns through the gateway (run it
                                      with recall off); the gateway trims the window
          clone GW STORE SID COND     copy the fed session once per question
                                      (needed only for WX_QUESTION_MODE=clone)
          ask GW STORE SID COND       ask every question; WX_QUESTION_MODE selects how:
                                      conversation (default) = the questions continue
                                      the fed conversation in order on one copy of it
                                      (SID-conv-COND); clone = each question on its own copy
          report STORE SID            accuracy and per-turn latency
          GW is the gateway port; STORE is the folder holding the store's qml.db.
"""
import json
import os
import random
import sqlite3
import statistics
import sys
import time
import urllib.error
import urllib.request

MODE = sys.argv[1]
GW = sys.argv[2] if len(sys.argv) > 2 else "0"
STORE = sys.argv[3] if len(sys.argv) > 3 else ""
SID = sys.argv[4] if len(sys.argv) > 4 else "wx"
ARG5 = sys.argv[5] if len(sys.argv) > 5 else ""
BASE = f"http://127.0.0.1:{GW}"
SYSTEM = ("You are a weather log assistant. Acknowledge reports with one"
          " short sentence; answer questions briefly.")

# scenario shared with bench_weather_longevity.py: CITIES ... FACT_AT and report_text
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_weather_longevity.py")).read()
_a = _src.index("CITIES = [")
_b = _src.index("#Func : post")
exec(compile(_src[_a:_b], "bench_weather_longevity.py (scenario)", "exec"))


def chat(sid, text, max_tokens):
    """One turn through the gateway; returns (answer, qml info, elapsed ms)."""
    body = {"session_id": sid, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"[gateway {e.code}] session {sid}: {e.read().decode(errors='replace')[:500]}", flush=True)
        raise
    return d["choices"][0]["message"]["content"], d.get("qml") or {}, (time.time() - t0) * 1000


def db():
    return sqlite3.connect(f"{STORE}/qml.db", timeout=60)


def fnv_tag(sid_):
    h = 1469598103934665603
    for b in sid_.encode():
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return format(h, "016x")


def clone_session(src, dst):
    """SQL copy of a session: session row, token log, archive and FTS rows."""
    con = db()
    for t in ("qml_token_log", "qml_token_log_archive", "qml_session"):
        con.execute(f"DELETE FROM {t} WHERE session_id=?", (dst,))
    con.execute("INSERT INTO qml_token_log (session_id,pos,seq_id,token_id)"
                " SELECT ?, pos, seq_id, token_id FROM qml_token_log WHERE session_id=?", (dst, src))
    con.execute("INSERT INTO qml_token_log_archive SELECT ?, orig_pos, seq_id, token_id, archived_at"
                " FROM qml_token_log_archive WHERE session_id=?", (dst, src))
    cols = [r[1] for r in con.execute("PRAGMA table_info(qml_session)")]
    sel = ", ".join("?" if c == "session_id" else c for c in cols)
    con.execute(f"INSERT INTO qml_session ({', '.join(cols)}) SELECT {sel} FROM qml_session"
                " WHERE session_id=?", (dst, src))
    try:
        con.execute("DELETE FROM qml_archive_fts WHERE session_id=?", (dst,))
        con.execute("INSERT INTO qml_archive_fts (content, tag, session_id, turn_pos, turn_end)"
                    " SELECT content, ?, ?, turn_pos, turn_end FROM qml_archive_fts"
                    " WHERE session_id=?", (fnv_tag(dst), dst, src))
    except sqlite3.OperationalError:
        pass
    con.commit()
    con.close()


def qsid(i, cls, cond):
    return f"{SID}-q{i}-{cls}-{cond}"


if MODE == "feed":
    n_turns = int(ARG5) if ARG5.isdigit() else 2000
    lat, n_trims, t_start = [], 0, time.time()
    for i in range(n_turns):
        text = FACT_AT[i][1] if i in FACT_AT else report_text(i)
        ans, q, ms = chat(SID, text, 8)
        rec = {"turn": i, "ms": round(ms, 1), "engine_ms": q.get("engine_ms"),
               "layer_ms": q.get("layer_ms"), "prompt_tokens": q.get("prompt_tokens"),
               "cached_tokens": q.get("cached_tokens")}
        if q.get("trimmed_tokens"):
            n_trims += 1
            rec["trimmed"] = q["trimmed_tokens"]
        lat.append(rec)
        if (i + 1) % 500 == 0:
            print(f"[feed] {i+1}/{n_turns} turns, {n_trims} trims, log {q.get('log_tokens')} tokens,"
                  f" {time.time()-t_start:.0f}s", flush=True)
    json.dump(lat, open(f"{STORE}/latency_{SID}.json", "w"))
    con = db()
    arc = con.execute("SELECT COUNT(*) FROM qml_token_log_archive WHERE session_id=?", (SID,)).fetchone()[0]
    live = con.execute("SELECT COUNT(*) FROM qml_token_log WHERE session_id=?", (SID,)).fetchone()[0]
    print(f"[feed] done: {n_turns} turns, {n_trims} trims, live {live} / archive {arc} tokens,"
          f" {time.time()-t_start:.0f}s", flush=True)

elif MODE == "clone":
    for fi, (tix, stmt, qs) in enumerate(FACTS):
        for cls, q, gold in qs:
            clone_session(SID, qsid(fi, cls, ARG5))
    print(f"[clone] {len(FACTS)*3} question copies for condition {ARG5}", flush=True)

elif MODE == "ask":
    qmode = os.environ.get("WX_QUESTION_MODE", "conversation")
    if qmode not in ("conversation", "clone"):
        sys.exit(f"WX_QUESTION_MODE={qmode}: use conversation or clone")
    conv = f"{SID}-conv-{ARG5}"
    if qmode == "conversation":
        clone_session(SID, conv)
    out = []
    for fi, (tix, stmt, qs) in enumerate(FACTS):
        for cls, q, gold in qs:
            sid = conv if qmode == "conversation" else qsid(fi, cls, ARG5)
            ans, info, ms = chat(sid, q, 48)
            ok = gold.lower() in ans.lower()
            out.append({"fact_turn": tix, "class": cls, "condition": ARG5, "gold": gold,
                        "question_mode": qmode, "ok": ok, "ms": round(ms, 1),
                        "engine_ms": info.get("engine_ms"), "layer_ms": info.get("layer_ms"),
                        "prompt_tokens": info.get("prompt_tokens"),
                        "cached_tokens": info.get("cached_tokens"),
                        "injected_tokens": info.get("injected_tokens"),
                        "recalled_pos": info.get("recalled_pos"),
                        "trimmed_tokens": info.get("trimmed_tokens"),
                        "answer": ans.strip()[:160]})
            print(f"[ask/{ARG5}] fact@{tix} {cls}: {'O' if ok else 'X'} inj={info.get('injected_tokens')}"
                  f" ({ans.strip()[:60]!r})", flush=True)
    json.dump(out, open(f"{STORE}/answers_{SID}_{ARG5}_{qmode}.json", "w"), indent=1)
    n_ok = sum(r["ok"] for r in out)
    inj = [r["injected_tokens"] or 0 for r in out]
    print(f"[ask/{ARG5}] total {n_ok}/{len(out)} ({qmode}), injected tokens mean"
          f" {statistics.mean(inj):.0f}", flush=True)

elif MODE == "report":
    for fn in sorted(os.listdir(STORE)):
        if fn.startswith(f"answers_{SID}_"):
            rows = json.load(open(f"{STORE}/{fn}"))
            print(f"{fn}: {sum(r['ok'] for r in rows)}/{len(rows)}")
    lf = f"{STORE}/latency_{SID}.json"
    if os.path.exists(lf):
        lat = json.load(open(lf))
        n = len(lat)
        for k in range(5):
            part = [r["ms"] for r in lat[k * n // 5:(k + 1) * n // 5]]
            part.sort()
            print(f"turns {k*n//5}-{(k+1)*n//5 - 1}: p50 {part[len(part)//2]:.0f} ms,"
                  f" p95 {part[int(0.95*len(part))]:.0f} ms")
