#!/usr/bin/env python3
"""reviewer1_edit_18_step2_burst_kill.py

Purpose : reviewer1_edit_18 step2 - turns lost when the process is killed inside the save window
Record  : TEST-LEDGER 104 (docs/3-status/TEST-LEDGER.md)
Usage   : reviewer1_edit_18_step2_burst_kill.py PORT STORE_DIR OUT_JSON SERVER_CMD_FILE [BURST] [REPEATS]
"""
import json, os, re, shutil, signal, sqlite3, subprocess, sys, time, urllib.request

PORT, STORE, OUT, CMDF = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
BURST = int(sys.argv[5]) if len(sys.argv) > 5 else 5
REPEATS = int(sys.argv[6]) if len(sys.argv) > 6 else 2
TARGETS = [5000, 15000, 25000]
BASE = "http://127.0.0.1:%d" % PORT
SID = "burst-kill-1"
DB = os.path.join(STORE, "qml.db")
FILLER = ("The development of transcontinental railroads in the nineteenth century "
          "reshaped commerce, migration, and the perception of distance. ")
SRVLOG = os.environ.get("SRV_LOG", OUT + ".srv.log")
now = time.perf_counter

lines = [l.rstrip("\n") for l in open(CMDF) if l.strip()]
SERVER_CMD = lines[0].split()
SERVER_ENV = dict(os.environ)
for l in lines[1:]:
    k, v = l.split("=", 1); SERVER_ENV[k] = v

#------------------------------------------------------------
#Func : post
#------------------------------------------------------------
def post(path, payload, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

#------------------------------------------------------------
#Func : health
#------------------------------------------------------------
def health():
    try:
        return b"ok" in urllib.request.urlopen(BASE + "/health", timeout=2).read()
    except Exception:
        return False

#------------------------------------------------------------
#Func : start_server
#------------------------------------------------------------
def start_server():
    cmd = SERVER_CMD + ["--port", str(PORT), "--qml-session-store", DB,
                        "--qml-lmdb-store", os.path.join(STORE, "qml_lmdb")]
    lf = open(SRVLOG, "ab")
    p = subprocess.Popen(cmd, env=SERVER_ENV, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT)
    t0 = now()
    while now() - t0 < 600:
        if p.poll() is not None:
            raise SystemExit("server exited rc=%s" % p.returncode)
        if health():
            return p
        time.sleep(0.5)
    raise SystemExit("server did not come up")

#------------------------------------------------------------
#Func : stop_server
#------------------------------------------------------------
def stop_server(p):
    p.send_signal(signal.SIGTERM); p.wait(timeout=120)

#------------------------------------------------------------
#Func : turn_text
#------------------------------------------------------------
def turn_text(i):
    return FILLER * 11 + "Turn %d marker is TM-%04d. Reply with OK." % (i, i)

#------------------------------------------------------------
#Func : send
#------------------------------------------------------------
def send(i):
    d = post("/completion", {"prompt": turn_text(i), "n_predict": 8, "temperature": 0,
                             "session_id": SID, "cache_prompt": True})
    tm = d.get("timings", {})
    return tm.get("prompt_n"), (d.get("tokens_cached") or 0)

#------------------------------------------------------------
#Func : token_ids
#------------------------------------------------------------
def token_ids():
    con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=5)
    con.execute("PRAGMA busy_timeout=5000")
    r = [x[0] for x in con.execute("SELECT token_id FROM qml_token_log WHERE session_id=? ORDER BY seq_id", (SID,))]
    con.close(); return r

#------------------------------------------------------------
#Func : markers_in_log
#------------------------------------------------------------
def markers_in_log():
    ids = token_ids()
    if not ids:
        return set(), 0
    text = post("/detokenize", {"tokens": ids})["content"]
    return set(int(m) for m in re.findall(r"TM-(\d{4})", text)), len(ids)

#------------------------------------------------------------
#Func : integrity
#------------------------------------------------------------
def integrity():
    try:
        con = sqlite3.connect(DB, timeout=5)
        r = con.execute("PRAGMA integrity_check").fetchone()[0]; con.close(); return r
    except Exception as e:
        return "error: %s" % e

#------------------------------------------------------------
#Func : wait_durable
#------------------------------------------------------------
def wait_durable(i, timeout=30.0):
    t0 = now()
    while now() - t0 < timeout:
        m, _ = markers_in_log()
        if i in m:
            return True
        time.sleep(0.05)
    return False

events = []
for rep in range(1, REPEATS + 1):
    if os.path.isdir(STORE):
        shutil.rmtree(STORE)
    os.makedirs(STORE)
    srv = start_server()
    turn = 0; history = 0
    for target in TARGETS:
        while history < target:
            turn += 1
            pn, cached = send(turn); history = (pn or 0) + cached + 8
            wait_durable(turn)
        pre_markers, pre_tokens = markers_in_log()
        burst_ids = []; t_b0 = now()
        for _ in range(BURST):
            turn += 1; burst_ids.append(turn)
            pn, cached = send(turn); history = (pn or 0) + cached + 8
        t_kill = now()
        os.kill(srv.pid, signal.SIGKILL); srv.wait()
        burst_s = t_kill - t_b0
        t_r0 = now(); srv = start_server(); restart_s = now() - t_r0
        post_markers, post_tokens = markers_in_log()
        lost = [i for i in burst_ids if i not in post_markers]
        kept = [i for i in burst_ids if i in post_markers]
        pre_lost = [i for i in range(1, burst_ids[0]) if i not in post_markers]
        turn += 1
        t0 = now(); pn, cached = send(turn); cont_s = now() - t0
        wait_durable(turn)
        ev = {"repeat": rep, "history_tokens_before_burst": history - BURST * 276, "history_tokens_after_burst": history,
              "burst_turns": burst_ids, "burst_wall_s": round(burst_s, 3), "durable_turns_of_burst": kept,
              "lost_turns_of_burst": lost, "lost_before_burst": pre_lost,
              "token_log_before": pre_tokens, "token_log_after": post_tokens,
              "integrity_check": integrity(), "restart_s": round(restart_s, 2),
              "continue_prompt_n": pn, "continue_tokens_cached": cached, "continue_s": round(cont_s, 3)}
        events.append(ev)
        print(json.dumps(ev), flush=True)
        json.dump(events, open(OUT, "w"), indent=1)
        history = (pn or 0) + cached + 8
    stop_server(srv)
summary = {"repeats": REPEATS, "burst": BURST, "events": len(events),
           "lost_per_event": [len(e["lost_turns_of_burst"]) for e in events],
           "max_lost": max(len(e["lost_turns_of_burst"]) for e in events),
           "any_lost_before_burst": any(e["lost_before_burst"] for e in events),
           "integrity_all_ok": all(e["integrity_check"] == "ok" for e in events)}
json.dump({"summary": summary, "events": events}, open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
