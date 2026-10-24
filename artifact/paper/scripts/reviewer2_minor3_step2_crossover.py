#!/usr/bin/env python3
"""bench_crossover.py

Purpose : Measures the reconnect-latency versus conversation-length crossover: stock (full recompute of the resent history) versus qml (session restore) at approximately 2K/8K/16K/32K/48K tokens, needle-verified, with a cold engine restart before each condition
Usage   : [DENS_CTX=131072] python3 bench_crossover.py
"""
import json, os, sqlite3, subprocess, sys, time, urllib.request

ENGINE = "http://127.0.0.1:8597"
DB = "$REPO/r2-3-check/store_xover/qml.db"
CTX = int(os.environ.get("DENS_CTX", "131072"))
SIZES = [300, 1200, 2400, 4800, 7200]
RESULTS = "$REPO/r2-3-check/xover_r23.json"
ENGINE_CMD = ("cd $REPO && env LD_LIBRARY_PATH=$REPO/env/lib:/usr/local/cuda/lib64 "
              "nohup third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf "
              f"-ngl 99 -fa on -c {CTX} --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port 8597 "
              "--qml-session-store $REPO/r2-3-check/store_xover/qml.db "
              "--qml-lmdb-store $REPO/r2-3-check/store_xover/qml_lmdb "
              "</dev/null > $REPO/r2-3-check/srv_xover_r23.log 2>&1 &")

#------------------------------------------------------------
#Func : http
#------------------------------------------------------------
def http(path, body, timeout=900):
    req = urllib.request.Request(ENGINE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

#------------------------------------------------------------
#Func : tok
#------------------------------------------------------------
def tok(text):
    return http("/tokenize", {"content": text, "add_special": False}, 300)["tokens"]

#------------------------------------------------------------
#Func : restart_engine
#------------------------------------------------------------
def restart_engine():
    subprocess.run(["pkill", "-x", "llama-server"])
    for _ in range(30):
        time.sleep(0.5)
        if subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode != 0:
            break
    subprocess.run(["bash", "-c", ENGINE_CMD])
    for _ in range(180):
        time.sleep(1)
        try:
            urllib.request.urlopen(ENGINE + "/health", timeout=2)
            return
        except Exception:
            pass
    raise RuntimeError("engine down")

#------------------------------------------------------------
#Func : q
#------------------------------------------------------------
def q(sql, p=()):
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return c.execute(sql, p).fetchall()
    finally:
        c.close()

#------------------------------------------------------------
#Func : sid
#------------------------------------------------------------
def sid(i):
    return f"xover::s{i}"

#------------------------------------------------------------
#Func : code
#------------------------------------------------------------
def code(i):
    return f"X{i}QQ7"

PROBE = ("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
         "What is the vault code? Answer with the code only."
         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

restart_engine()
sizes_tok = []
for i, reps in enumerate(SIZES):
    filler = " ".join(f"s{i} item{j} ok." for j in range(reps))
    p = tok("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            + filler + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nOK.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\nThe vault code is {code(i)}. "
            "Remember it.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    http("/completion", {"prompt": p, "n_predict": 8, "temperature": 0,
                         "session_id": sid(i), "user_id": "xover", "cache_prompt": True})
    sizes_tok.append(len(p) + 10)
    print(f"built {sid(i)}: ~{len(p)} tokens", flush=True)
for _ in range(300):
    time.sleep(2)
    if q("SELECT COUNT(*) FROM qml_llama_session_ext WHERE session_id LIKE 'xover::%'")[0][0] >= len(SIZES):
        break
print("blobs persisted", flush=True)

probe_tokens = tok(PROBE)
res = {"sizes_tokens": sizes_tok}
for mode in ("qml", "stock"):
    restart_engine()
    rows = []
    for i in range(len(SIZES)):
        if mode == "qml":
            body = {"prompt": probe_tokens, "n_predict": 8, "temperature": 0,
                    "session_id": sid(i), "user_id": "xover", "cache_prompt": True}
        else:
            hist = [r[0] for r in q("SELECT token_id FROM qml_token_log WHERE session_id=? ORDER BY pos", (sid(i),))]
            body = {"prompt": hist + probe_tokens, "n_predict": 8, "temperature": 0,
                    "cache_prompt": True}
        t0 = time.time()
        d = http("/completion", body)
        dt = round(time.time() - t0, 3)
        content = d.get("content", "")
        ok = code(i) in content
        pn = (d.get("timings") or {}).get("prompt_n")
        rows.append({"tokens": sizes_tok[i], "latency": dt, "correct": ok, "prompt_n": pn})
        print(f"[{mode}] ~{sizes_tok[i]} tok: {dt}s correct={ok} prompt_n={pn}", flush=True)
    res[mode] = rows
json.dump(res, open(RESULTS, "w"), indent=1)

pid_out = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True)
pid = int(pid_out.stdout.split()[0]) if pid_out.stdout.strip() else 0
con = sqlite3.connect(DB, timeout=30)
if pid:
    for i in range(len(SIZES)):
        con.execute("INSERT INTO qml_control_cmd (target_engine,target_pid,cmd_type,session_id,created_at) "
                    "VALUES ('llama.cpp',?,'DROP_BLOB',?,strftime('%s','now'))", (pid, sid(i)))
    con.commit()
    time.sleep(15)
for t in ("qml_token_log", "qml_llama_session_ext", "qml_session"):
    con.execute(f"DELETE FROM {t} WHERE session_id LIKE 'xover::%'")
con.commit(); con.close()
print("cleanup done", flush=True)
