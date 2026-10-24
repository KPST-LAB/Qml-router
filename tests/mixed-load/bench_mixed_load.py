#!/usr/bin/env python3
"""bench_mixed_load.py

Purpose : Measures how much short-request traffic survives while long-context sessions keep returning: session restore (qml) versus stateless full resend (plain llama.cpp) versus stateless against an OpenAI-compatible engine (stock vLLM / SGLang)
Usage   : bench_mixed_load.py PORT MODE STORE_DIR [NLONG=6] [LTOK=32000] [DUR=150] [LOG]
Usage   : python3 bench_mixed_load.py PORT MODE STORE_DIR [NLONG=6] [LTOK=32000]
Modes   : build, session, stateless, stateless-openai: the paper's protocol (one raw-text history per
          session, no turn boundaries; every reconnect adds about 22 tokens, so a session that starts
          at 32,008 tokens fills a 32,768-token slot after 34 reconnects and the server then refuses it)
          build-chat, session-chat, stateless-chat: the same sessions in the Llama-3 chat format. The
          history is a series of user turns of about 900 tokens, each answered "OK."; the vault code is
          in the system message, which a window trim keeps. With QML_TRIM_WINDOW and QML_TRIM_EOT set
          on the server, the session is trimmed at a user-turn boundary instead of being refused
Settings: QML_BENCH_SALT=1 makes mode stateless-openai send "cache_salt" = the long session's id
          (mix-N, the id the build mode stores the session under) on every long request; the
          default 0 sends the requests unchanged
"""
import concurrent.futures
import json
import os
import sqlite3
import sys
import threading
import time
import urllib.request

PORT = int(sys.argv[1])
MODE = sys.argv[2]
STORE = sys.argv[3]
NLONG = int(sys.argv[4]) if len(sys.argv) > 4 else 6
LTOK = int(sys.argv[5]) if len(sys.argv) > 5 else 32000
DUR = int(sys.argv[6]) if len(sys.argv) > 6 else 150
LOGFILE = sys.argv[7] if len(sys.argv) > 7 else f"{STORE}/server.log"
SALT = os.environ.get("QML_BENCH_SALT", "0") == "1"

BASE = f"http://127.0.0.1:{PORT}"
FILLER = ("The development of transcontinental railroads in the nineteenth century "
          "reshaped commerce, migration, and the perception of distance. ")
ASK = " Recalling the important record, the vault code is"
REPS = max(4, LTOK // 23)

#------------------------------------------------------------
#Func : needle
#------------------------------------------------------------
def needle(i):
    return f"ZK-{6000 + i}"

#------------------------------------------------------------
#Func : history
#------------------------------------------------------------
def history(i):
    tenth = max(1, REPS // 10)
    return (FILLER * tenth + f"IMPORTANT RECORD: the vault code is {needle(i)}. "
            + FILLER * (REPS - tenth))

EOT = "<|eot_id|>"
CHAT_ASK = ("<|start_header_id|>user<|end_header_id|>\n\nWhat is the vault code in the important"
            " record? Answer with the code only." + EOT + "<|start_header_id|>assistant<|end_header_id|>\n\n")
TURN_SENTENCES = 40   # filler sentences per user turn (about 900 tokens)
# stateless-chat keeps the same window as the server's trim (QML_TRIM_WINDOW of the qml server; 0 = none)
CHAT_WINDOW = int(os.environ.get("QML_TRIM_WINDOW", "0"))

#------------------------------------------------------------
#Func : history_chat
#chat-format history of session i: system message with the vault code, then user turns of filler
#------------------------------------------------------------
def history_chat(i):
    text = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant."
            f" IMPORTANT RECORD: the vault code is {needle(i)}." + EOT)
    left = REPS
    while left > 0:
        k = min(TURN_SENTENCES, left)
        left -= k
        text += ("<|start_header_id|>user<|end_header_id|>\n\n" + FILLER * k + EOT
                 + "<|start_header_id|>assistant<|end_header_id|>\n\nOK." + EOT)
    return text

#------------------------------------------------------------
#Func : tokenize
#------------------------------------------------------------
def tokenize(text):
    req = urllib.request.Request(BASE + "/tokenize",
                                 data=json.dumps({"content": text, "add_special": False}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["tokens"]

#------------------------------------------------------------
#Func : last_token
#last token of the stored session (None = no session row)
#------------------------------------------------------------
def last_token(sid):
    con = sqlite3.connect(f"file:{STORE}/qml.db?mode=ro", uri=True, timeout=30)
    try:
        row = con.execute("SELECT last_token FROM qml_session WHERE session_id=?", (sid,)).fetchone()
    finally:
        con.close()
    return row[0] if row else None

#------------------------------------------------------------
#Func : completion
#------------------------------------------------------------
def completion(payload, timeout=1200):
    req = urllib.request.Request(BASE + "/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

OPENAI_MODEL = None

#------------------------------------------------------------
#Func : completion_openai
#stock vLLM/SGLang path. Returns (text, gpu_prefill_tokens); salt = cache_salt to send (None = none)
#------------------------------------------------------------
def completion_openai(prompt, n_predict, timeout=1200, salt=None):
    global OPENAI_MODEL
    if OPENAI_MODEL is None:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
            OPENAI_MODEL = json.loads(r.read())["data"][0]["id"]
    payload = {"model": OPENAI_MODEL, "prompt": prompt, "max_tokens": n_predict,
               "temperature": 0}
    if salt is not None:
        payload["cache_salt"] = salt
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    text = d["choices"][0]["text"]
    usage = d.get("usage") or {}
    ptok = int(usage.get("prompt_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    return text, max(0, ptok - cached)

#------------------------------------------------------------
#Func : build
#------------------------------------------------------------
def build():
    t0 = time.time()
    for i in range(NLONG):
        prompt = tokenize(history_chat(i)) if MODE == "build-chat" else history(i)
        d = completion({"prompt": prompt, "n_predict": 1, "temperature": 0,
                        "session_id": f"mix-{i}", "cache_prompt": True})
        pn = int((d.get("timings") or {}).get("prompt_n") or 0)
        print(f"[build] session mix-{i}: {pn:,} tokens", flush=True)

    db = f"{STORE}/qml.db"
    prev = -1
    while True:
        time.sleep(2)
        con = sqlite3.connect(db)
        n_rows, total = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(state_size),0) FROM qml_session s "
            "JOIN qml_llama_session_ext e ON s.session_id = e.session_id "
            "WHERE s.session_id LIKE 'mix-%'").fetchone()
        con.close()
        if n_rows >= NLONG and total == prev:
            break
        prev = total
    print(f"[build] settled: {n_rows} sessions, {total / 2**30:.1f} GiB state "
          f"({time.time() - t0:.0f}s)", flush=True)

#------------------------------------------------------------
#Func : measure
#------------------------------------------------------------
def measure():
    if MODE == "stateless-chat":
        hist = {i: history_chat(i) for i in range(NLONG)}
    elif MODE.startswith("stateless"):
        hist = {i: history(i) for i in range(NLONG)}
    else:
        hist = None
    eot_id = tokenize(EOT)[0] if MODE == "session-chat" else None
    stop = threading.Event()
    lock = threading.Lock()
    shorts, longs = [], []

    def long_worker():
        i = 0
        while not stop.is_set():
            sid = i % NLONG
            i += 1
            t = time.time()
            try:
                if MODE == "stateless-openai":
                    text, pn = completion_openai(hist[sid] + ASK, 12,
                                                 salt=f"mix-{sid}" if SALT else None)
                    el = (time.time() - t) * 1000.0
                    ok = needle(sid) in text
                    hist[sid] = hist[sid] + ASK + text
                else:
                    if MODE == "session":
                        payload = {"prompt": ASK, "n_predict": 12, "temperature": 0,
                                   "session_id": f"mix-{sid}", "cache_prompt": True}
                    elif MODE == "session-chat":
                        # close the previous answer when the stored log does not end with eot
                        ask = CHAT_ASK if last_token(f"mix-{sid}") == eot_id else EOT + CHAT_ASK
                        payload = {"prompt": tokenize(ask), "n_predict": 12, "temperature": 0,
                                   "session_id": f"mix-{sid}", "cache_prompt": True}
                    elif MODE == "stateless-chat":
                        toks = tokenize(hist[sid] + CHAT_ASK)
                        # the client keeps the window the server keeps for a session: drop the
                        # oldest user turn and its answer (the system message stays)
                        while CHAT_WINDOW and len(toks) > CHAT_WINDOW:
                            head_end = hist[sid].index(EOT) + len(EOT)
                            pair_end = hist[sid].index(EOT, hist[sid].index(EOT, head_end) + len(EOT)) + len(EOT)
                            hist[sid] = hist[sid][:head_end] + hist[sid][pair_end:]
                            toks = tokenize(hist[sid] + CHAT_ASK)
                        payload = {"prompt": toks, "n_predict": 12,
                                   "temperature": 0, "cache_prompt": True}
                    else:
                        payload = {"prompt": hist[sid] + ASK, "n_predict": 12,
                                   "temperature": 0, "cache_prompt": True}
                    d = completion(payload)
                    el = (time.time() - t) * 1000.0
                    text = d.get("content", "")
                    ok = needle(sid) in text
                    pn = int((d.get("timings") or {}).get("prompt_n") or 0)
                    if MODE == "stateless":
                        hist[sid] = hist[sid] + ASK + text
                    elif MODE == "stateless-chat":
                        hist[sid] = hist[sid] + CHAT_ASK + text + EOT
            except Exception:
                el, ok, pn = float("inf"), False, 0
            with lock:
                longs.append((el, ok, pn))

    def short_worker(w):
        k = 0
        while not stop.is_set():
            k += 1
            tag = w * 100000 + k
            prompt = (f"Customer note {tag}. Category: billing. Priority: normal. "
                      f"The ticket number is TK-{tag}. Please keep this on file. "
                      f"Question: what is the ticket number? The ticket number is")
            t = time.time()
            try:
                if MODE == "stateless-openai":
                    text, pn = completion_openai(prompt, 16)
                    el = (time.time() - t) * 1000.0
                    ok = f"TK-{tag}" in text
                else:
                    d = completion({"prompt": prompt, "n_predict": 16, "temperature": 0,
                                    "cache_prompt": True})
                    el = (time.time() - t) * 1000.0
                    ok = f"TK-{tag}" in d.get("content", "")
                    pn = int((d.get("timings") or {}).get("prompt_n") or 0)
            except Exception:
                el, ok, pn = float("inf"), False, 0
            with lock:
                shorts.append((el, ok, pn))

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        ex.submit(long_worker)
        for w in range(3):
            ex.submit(short_worker, w)
        time.sleep(DUR)
        stop.set()
    wall = time.time() - t0

    def pct(v, p):
        s = sorted(v)
        return s[min(len(s) - 1, int(len(s) * p))] if s else float("nan")

    with lock:
        sl = [r[0] for r in shorts]
        ll = [r[0] for r in longs]
        print(f"\n=== {MODE}: {NLONG} long sessions x {LTOK:,} tokens + short requests, 3 workers, "
              f"{wall:.0f}s ===")
        print(f"  short requests : completed {len(shorts)}  correct {sum(1 for r in shorts if r[1])}  "
              f"p50={pct(sl, .5):,.0f}  p90={pct(sl, .9):,.0f} ms  "
              f"throughput {len(shorts) / wall:.2f} req/s")
        print(f"  long reconnects: completed {len(longs)}  correct {sum(1 for r in longs if r[1])}  "
              f"p50={pct(ll, .5):,.0f}  p90={pct(ll, .9):,.0f} ms")
        print(f"  GPU prefill    : short {sum(r[2] for r in shorts):,} tokens, "
              f"long {sum(r[2] for r in longs):,} tokens", flush=True)

if MODE in ("build", "build-chat"):
    build()
elif MODE in ("session", "stateless", "stateless-openai", "session-chat", "stateless-chat"):
    measure()
else:
    sys.exit(f"unknown mode {MODE}")
