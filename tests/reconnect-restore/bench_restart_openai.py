#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
"""bench_restart_openai.py

Purpose : restart recovery on vLLM (qml connector) and SGLang (qml HiCache backend).
          Long conversations are sent with a session identity (vLLM cache_salt,
          SGLang extra_key), the engine module saves their KV to the qml store, the
          engine is restarted, and the same requests are sent again: the KV comes
          back from the store instead of being recomputed. Every answer is checked
          against a code planted in its own conversation.
Modes   : build PORT ENGINE TOKDIR OUT N_SESS N_TOKENS   first pass (save)
          again PORT ENGINE TOKDIR OUT N_SESS N_TOKENS   after the restart: the same
                                                          sessions (restore), then the
                                                          same prompts under new session
                                                          ids (no stored KV: recompute)
          ENGINE is vllm or sglang; TOKDIR is the model directory (tokenizer);
          OUT is a JSON file the results are appended to.
"""
import json
import random
import sys
import time
import urllib.request

from transformers import AutoTokenizer

MODE, PORT, ENGINE, TOKDIR, OUT = sys.argv[1:6]
N_SESS, N_TOKENS = int(sys.argv[6]), int(sys.argv[7])
BASE = f"http://127.0.0.1:{PORT}"
KEY = "cache_salt" if ENGINE == "vllm" else "extra_key"
CITIES = ["Seoul", "Busan", "Daegu", "Incheon", "Gwangju", "Daejeon", "Ulsan", "Suwon"]


def conversation(s, tok):
    """Token ids of session s: a long report log with one planted code, then the question."""
    r = random.Random(7000 + s)
    code = f"QX-{r.randint(1000, 9999)}"
    lines, n = [], 0
    plant = r.randint(10, 60)
    i = 0
    while n < N_TOKENS:
        if i == plant:
            line = f"Station note {i}: the maintenance code for this log is {code}."
        else:
            line = (f"Station note {i}, {CITIES[(i + s) % 8]}: high {r.uniform(-5, 35):.1f} C,"
                    f" humidity {r.randint(20, 98)} percent, wind {r.randint(2, 90)} km/h.")
        lines.append(line)
        n += 24
        i += 1
    msgs = [{"role": "system", "content": "You read station logs and answer briefly."},
            {"role": "user", "content": "\n".join(lines) + "\n\nWhat is the maintenance code for this log?"
             " Reply with the code only."}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids), code


def ask(ids, sid):
    body = {"model": "qml-bench", "prompt": ids, "max_tokens": 12, "temperature": 0, KEY: sid}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    ms = (time.time() - t0) * 1000
    u = d.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    return d["choices"][0]["text"], ms, u.get("prompt_tokens"), cached


def run(label, sid_fmt, tok):
    rows = []
    for s in range(N_SESS):
        ids, code = conversation(s, tok)
        text, ms, pt, cached = ask(ids, sid_fmt.format(s=s))
        ok = code in text
        rows.append({"phase": label, "session": s, "ms": round(ms, 1), "prompt_tokens": pt,
                     "cached_tokens": cached, "ok": ok, "answer": text.strip()[:40]})
        print(f"[{label}] s{s}: {ms:.0f} ms, prompt {pt}, cached {cached}, {'OK' if ok else 'WRONG'}"
              f" ({text.strip()[:30]!r})", flush=True)
    return rows


tok = AutoTokenizer.from_pretrained(TOKDIR)
if MODE == "build":
    rows = run("build", "user::restart-{s}", tok)
else:
    rows = run("restore", "user::restart-{s}", tok)
    rows += run("recompute", "user::fresh-{s}", tok)
try:
    old = json.load(open(OUT))
except (OSError, ValueError):
    old = []
json.dump(old + [dict(r, engine=ENGINE) for r in rows], open(OUT, "w"), indent=1)
