#!/usr/bin/env python3
"""bench_migration.py

Purpose : Demonstrates cross-engine session migration llama.cpp -> vLLM -> llama.cpp: one session, one needle, three engine hops, every answer verified; the token log in the qml store is the engine-neutral carrier
Usage   : bench_migration.py MODE PORT STORE_DIR
Usage   : python3 bench_migration.py MODE PORT STORE_DIR
"""
import json
import sqlite3
import sys
import time
import urllib.request

MODE = sys.argv[1]
PORT = int(sys.argv[2])
STORE = sys.argv[3]

BASE = f"http://127.0.0.1:{PORT}"
MODEL = "unsloth/Llama-3.2-3B-Instruct"
SID = "alice::mig-1"
NEEDLE = "ZK-7777"
FILLER = ("The development of transcontinental railroads in the nineteenth century "
          "reshaped commerce, migration, and the perception of distance. ")
ASK = " Recalling the important record, the vault code is"
REPS = 8000 // 23

#------------------------------------------------------------
#Func : history
#------------------------------------------------------------
def history():
    tenth = max(1, REPS // 10)
    return (FILLER * tenth + f"IMPORTANT RECORD: the vault code is {NEEDLE}. "
            + FILLER * (REPS - tenth))

#------------------------------------------------------------
#Func : post
#------------------------------------------------------------
def post(path, payload, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

#------------------------------------------------------------
#Func : verdict
#------------------------------------------------------------
def verdict(label, t0, text, extra=""):
    el = (time.time() - t0) * 1000.0
    ok = NEEDLE in text
    print(f"[{label}] latency={el:,.0f} ms  correct={'OK' if ok else 'WRONG(' + text.strip()[:60] + ')'}"
          f"{'  ' + extra if extra else ''}", flush=True)
    return ok

#------------------------------------------------------------
#Func : token_log
#------------------------------------------------------------
def token_log():
    con = sqlite3.connect(f"{STORE}/qml.db")
    rows = con.execute(
        "SELECT token_id FROM qml_token_log WHERE session_id=? ORDER BY pos",
        (SID,)).fetchall()
    con.close()
    return [r[0] for r in rows]

if MODE == "build_llama":
    t0 = time.time()
    d = post("/completion", {"prompt": history() + ASK, "n_predict": 12,
                             "temperature": 0, "session_id": SID,
                             "cache_prompt": True})
    ok = verdict("1.build@llama.cpp", t0, d.get("content", ""),
                 f"prefill={int((d.get('timings') or {}).get('prompt_n') or 0):,} tokens")
    sys.exit(0 if ok else 1)

elif MODE == "continue_vllm":
    ids = token_log()
    print(f"[2.migrate] read {len(ids):,} tokens from the token log (engine-neutral carrier)", flush=True)
    ask_ids = post("/tokenize", {"prompt": ASK, "add_special_tokens": False})["tokens"]
    t0 = time.time()
    d = post("/v1/completions", {"model": MODEL, "prompt": ids + ask_ids,
                                 "max_tokens": 12, "temperature": 0,
                                 "cache_salt": SID})
    text = d["choices"][0]["text"]
    ok = verdict("2.continue@vLLM", t0, text,
                 f"(migration cost = one prefill of {len(ids) + len(ask_ids):,} tokens on the target engine)")
    sys.exit(0 if ok else 1)

elif MODE == "continue_llama":
    t0 = time.time()
    d = post("/completion", {"prompt": ASK + " Answer once more: the vault code is",
                             "n_predict": 12, "temperature": 0,
                             "session_id": SID, "cache_prompt": True})
    ok = verdict("3.return@llama.cpp", t0, d.get("content", ""),
                 f"prefill={int((d.get('timings') or {}).get('prompt_n') or 0):,} tokens")
    sys.exit(0 if ok else 1)

else:
    sys.exit(f"unknown mode {MODE}")
