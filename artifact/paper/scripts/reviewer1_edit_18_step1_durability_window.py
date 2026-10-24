#!/usr/bin/env python3
"""reviewer1_edit_18_step1_durability_window.py

Purpose : reviewer1_edit_18 step1 - durability window of the asynchronous save
Record  : TEST-LEDGER 102 (docs/3-status/TEST-LEDGER.md)
Usage   : reviewer1_edit_18_step1_durability_window.py PORT STORE_DIR OUT_JSON [N_TURNS] [DEV]
"""
import json, os, sqlite3, sys, time, urllib.request

PORT, STORE, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
N_TURNS = int(sys.argv[4]) if len(sys.argv) > 4 else 100
DEV = sys.argv[5] if len(sys.argv) > 5 else "nvme0n1"
BASE = "http://127.0.0.1:%d" % PORT
SID = "durability-1"
DB = os.path.join(STORE, "qml.db"); MDB = os.path.join(STORE, "qml_lmdb", "data.mdb")
FILLER = ("The development of transcontinental railroads in the nineteenth century "
          "reshaped commerce, migration, and the perception of distance. ")
POLL_S, BLOB_WAIT_S, TOK_WAIT_S = 0.002, 10.0, 30.0
now = time.perf_counter

#------------------------------------------------------------
#Func : post
#------------------------------------------------------------
def post(path, payload, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

#------------------------------------------------------------
#Func : turn_text
#------------------------------------------------------------
def turn_text(i):
    return FILLER * 11 + "Turn %d marker is TM-%04d. Reply with OK." % (i, i)

#------------------------------------------------------------
#Func : sectors_written
#------------------------------------------------------------
def sectors_written():
    try:
        with open("/sys/block/%s/stat" % DEV) as f:
            return int(f.read().split()[6])
    except Exception:
        return -1

#------------------------------------------------------------
#Func : mdb_stat
#------------------------------------------------------------
def mdb_stat():
    try:
        st = os.stat(MDB); return (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        return (0, 0)

#------------------------------------------------------------
#Func : token_count
#------------------------------------------------------------
def token_count():
    try:
        con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=5)
        con.execute("PRAGMA busy_timeout=5000")
        n = con.execute("SELECT COUNT(*) FROM qml_token_log WHERE session_id=?", (SID,)).fetchone()[0]
        con.close(); return n
    except Exception:
        return -1

#------------------------------------------------------------
#Func : state_size
#------------------------------------------------------------
def state_size():
    try:
        con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=5)
        con.execute("PRAGMA busy_timeout=5000")
        r = con.execute("SELECT state_size FROM qml_llama_session_ext WHERE session_id=?", (SID,)).fetchone()
        con.close(); return r[0] if r else -1
    except Exception:
        return -2

sw0 = sectors_written()
recs = []
print("start: sectors_written=%d" % sw0, flush=True)
for i in range(1, N_TURNS + 1):
    n_tok_before = token_count(); size_before = state_size(); mdb_before = mdb_stat()
    t_send = now()
    d = post("/completion", {"prompt": turn_text(i), "n_predict": 8, "temperature": 0,
                             "session_id": SID, "cache_prompt": True})
    t_resp = now()
    prompt_n = d.get("tokens_evaluated", d.get("timings", {}).get("prompt_n"))
    t_tokens = None
    deadline = t_resp + TOK_WAIT_S
    while now() < deadline:
        if token_count() > n_tok_before:
            t_tokens = now(); break
        time.sleep(POLL_S)
    t_blob = None; size_after = size_before
    deadline = t_resp + BLOB_WAIT_S
    while now() < deadline:
        cur = state_size()
        if cur != size_before:
            t_blob = now(); size_after = cur; break
        time.sleep(POLL_S)
    rec = {"turn": i, "prompt_n": prompt_n, "resp_s": round(t_resp - t_send, 4),
           "window_tokens_s": None if t_tokens is None else round(t_tokens - t_resp, 4),
           "window_blob_s": None if t_blob is None else round(t_blob - t_resp, 4),
           "state_size_after": size_after, "mdb_mtime_changed": mdb_stat()[0] != mdb_before[0]}
    recs.append(rec)
    if i % 10 == 0 or i <= 3:
        print("[turn %3d] prompt_n=%s resp %.2fs window_tokens %s window_blob %s state %.1f MB" % (
            i, prompt_n, rec["resp_s"], rec["window_tokens_s"], rec["window_blob_s"], max(0, size_after) / 1e6), flush=True)
    json.dump(recs, open(OUT, "w"), indent=1)

sw1 = sectors_written()
tok_w = [r["window_tokens_s"] for r in recs if r["window_tokens_s"] is not None]
blob_w = [r["window_blob_s"] for r in recs if r["window_blob_s"] is not None]

#------------------------------------------------------------
#Func : summ
#------------------------------------------------------------
def summ(xs):
    xs = sorted(xs)
    return {} if not xs else {"n": len(xs), "median_s": round(xs[len(xs) // 2], 4), "p95_s": round(xs[min(len(xs) - 1, int(0.95 * len(xs)))], 4),
                              "max_s": round(xs[-1], 4), "min_s": round(xs[0], 4)}
summary = {"turns": N_TURNS, "tokens_only_window": summ(tok_w), "full_capture_window": summ(blob_w),
           "turns_with_full_capture": len(blob_w), "turns_without_token_save_observed": sum(1 for r in recs if r["window_tokens_s"] is None),
           "host_bytes_written": None if sw0 < 0 or sw1 < 0 else (sw1 - sw0) * 512,
           "final_state_size_bytes": state_size()}
json.dump({"summary": summary, "turns": recs}, open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
