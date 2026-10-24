#!/usr/bin/env python3
"""reviewer1_edit_10_step1_shared_prefix_storage.py

Purpose : reviewer1_edit_10 step1 - storage cost of identity keying for a shared prefix
Record  : TEST-LEDGER 103 (docs/3-status/TEST-LEDGER.md)
Usage   : reviewer1_edit_10_step1_shared_prefix_storage.py PORT STORE_DIR OUT_JSON [N_SESSIONS]
"""
import json, os, sqlite3, subprocess, sys, time, urllib.request

PORT, STORE, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 10
BASE = "http://127.0.0.1:%d" % PORT
DB = os.path.join(STORE, "qml.db"); MDB_DIR = os.path.join(STORE, "qml_lmdb")
FILLER = ("The development of transcontinental railroads in the nineteenth century "
          "reshaped commerce, migration, and the perception of distance. ")
now = time.perf_counter

#------------------------------------------------------------
#Func : post
#------------------------------------------------------------
def post(path, payload, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

#------------------------------------------------------------
#Func : du
#------------------------------------------------------------
def du(path):
    try:
        return int(subprocess.check_output(["du", "-s", "-B1", path]).split()[0])
    except Exception:
        return -1

#------------------------------------------------------------
#Func : db_bytes
#------------------------------------------------------------
def db_bytes():
    return sum(os.path.getsize(DB + x) for x in ("", "-wal") if os.path.exists(DB + x))

#------------------------------------------------------------
#Func : q
#------------------------------------------------------------
def q(sql, args=()):
    con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=5)
    con.execute("PRAGMA busy_timeout=5000")
    r = con.execute(sql, args).fetchone(); con.close()
    return r

#------------------------------------------------------------
#Func : state_size
#------------------------------------------------------------
def state_size(sid):
    try:
        r = q("SELECT state_size FROM qml_llama_session_ext WHERE session_id=?", (sid,))
        return r[0] if r else -1
    except Exception:
        return -2

n_tok = post("/tokenize", {"content": FILLER, "add_special": False})["tokens"]
reps = max(1, 4096 // len(n_tok))
PREFIX = FILLER * reps
prefix_tokens = len(post("/tokenize", {"content": PREFIX, "add_special": False})["tokens"])
print("prefix tokens=%d (filler %d tokens x %d)" % (prefix_tokens, len(n_tok), reps), flush=True)

recs = []; lmdb0, db0 = du(MDB_DIR), db_bytes()
print("store before: lmdb=%d db=%d" % (lmdb0, db0), flush=True)
for i in range(1, N + 1):
    sid = "user%02d::shared" % i
    t0 = now()
    d = post("/completion", {"prompt": PREFIX + " Reply with OK.", "n_predict": 4, "temperature": 0,
                             "session_id": sid, "cache_prompt": True})
    t_resp = now() - t0
    tm = d.get("timings", {})
    deadline = now() + 60.0; ss = -1
    while now() < deadline:
        ss = state_size(sid)
        if ss > 0:
            break
        time.sleep(0.01)
    time.sleep(1.0)
    lmdb, dbsz = du(MDB_DIR), db_bytes()
    rec = {"session": sid, "n": i, "resp_s": round(t_resp, 3), "prompt_n": tm.get("prompt_n"),
           "tokens_evaluated": d.get("tokens_evaluated"), "tokens_cached": d.get("tokens_cached"),
           "session_tokens": (tm.get("prompt_n") or 0) + (tm.get("predicted_n") or 0),
           "state_size": ss, "lmdb_bytes_cum": lmdb - lmdb0, "db_bytes_cum": dbsz - db0}
    recs.append(rec)
    print("[%2d] prompt_n=%s cached=%s state=%.1f MB lmdb_cum=%.1f MB db_cum=%.1f MB resp=%.2fs" % (
        i, rec["prompt_n"], rec["tokens_cached"], max(0, ss) / 1e6, rec["lmdb_bytes_cum"] / 1e6,
        rec["db_bytes_cum"] / 1e6, t_resp), flush=True)
    json.dump(recs, open(OUT, "w"), indent=1)

sizes = [r["state_size"] for r in recs if r["state_size"] > 0]
toks = [r["session_tokens"] for r in recs]
b_per_tok = [s / t for s, t in zip(sizes, toks) if t]
summary = {"sessions": N, "prefix_tokens": prefix_tokens,
           "state_size_first": sizes[0] if sizes else None, "state_size_sum": sum(sizes),
           "bytes_per_token_median": sorted(b_per_tok)[len(b_per_tok) // 2] if b_per_tok else None,
           "lmdb_growth_bytes": recs[-1]["lmdb_bytes_cum"], "db_growth_bytes": recs[-1]["db_bytes_cum"],
           "predicted_paper_31_4KB": N * (toks[0] if toks else 0) * 31.4e3,
           "content_keying_one_copy": sizes[0] if sizes else None,
           "prompt_n_first": recs[0]["prompt_n"], "prompt_n_rest_max": max([r["prompt_n"] or 0 for r in recs[1:]] or [0])}
json.dump({"summary": summary, "sessions": recs}, open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
