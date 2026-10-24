#!/usr/bin/env python3
"""bench_density_curve.py

Purpose : Density-limit curve: the distribution of reconnect completion times when N users reconnect at once, stock (full history resend and recompute) versus qml (session restore), on the same engine with the same content
Usage   : python3 bench_density_curve.py build|qml|stock|cleanup|all
Settings: QML_BENCH_SERVER (llama-server path), QML_BENCH_MODEL (model file),
          QML_BENCH_STORE (store folder, default $REPO/store-tplseed), QML_BENCH_ENV (extra environment
          assignments for the server), QML_BENCH_PORT (default 8397), QML_BENCH_LOG (server log),
          QML_BENCH_RESULTS (result JSON, falls back to DENS_RESULTS); the program stops only the server
          process it started itself (its process group), never by name
"""
import atexit, json, math, os, shlex, signal, socket, sqlite3, subprocess, sys, threading, time, urllib.request

REPO = os.environ.get("REPO", "")   # repository root; the server runs with this as its working folder
PORT = int(os.environ.get("QML_BENCH_PORT", "8397"))
ENGINE = f"http://127.0.0.1:{PORT}"
SERVER = os.environ.get("QML_BENCH_SERVER", f"{REPO}/third_party/llama.cpp/build/bin/llama-server")
MODEL = os.environ.get("QML_BENCH_MODEL", f"{REPO}/models/Llama-3.2-3B-Instruct-F16.gguf")
STORE = os.environ.get("QML_BENCH_STORE", f"{REPO}/store-tplseed")
DB = f"{STORE}/qml.db"
ENVSTR = os.environ.get("QML_BENCH_ENV", f"LD_LIBRARY_PATH={REPO}/env/lib:/usr/local/cuda/lib64")
SRVLOG = os.environ.get("QML_BENCH_LOG", f"{REPO}/srv_density.log")
LEVELS = [int(x) for x in os.environ.get("DENS_LEVELS", "4,8,16,32,64").split(",")]
TOTAL = sum(LEVELS)
H_REPS = int(os.environ.get("DENS_H_REPS", "850"))
STAGGER = float(os.environ.get("DENS_STAGGER", "0"))
CTX = int(os.environ.get("DENS_CTX", "65536"))
RESULTS = os.environ.get("QML_BENCH_RESULTS", os.environ.get("DENS_RESULTS", "/tmp/density_results.json"))
ENGINE_ARGS = [SERVER, "-m", MODEL,
               "-ngl", "99", "-fa", "on", "-c", str(CTX), "--kv-unified", "-np", "8",
               "-ctk", "q4_0", "-ctv", "q4_0", "--cache-ram", "0", "--port", str(PORT),
               "--qml-session-store", DB,
               "--qml-lmdb-store", f"{STORE}/qml_lmdb"]
ENGINE_ENV = dict(os.environ)
for _kv in shlex.split(ENVSTR):
    _k, _, _v = _kv.partition("=")
    ENGINE_ENV[_k] = os.path.expandvars(_v)
_proc = None   # the llama-server this program started (None = none running)

#------------------------------------------------------------
#Func : http
#------------------------------------------------------------
def http(path, body, timeout=600):
    req = urllib.request.Request(ENGINE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

#------------------------------------------------------------
#Func : tok
#------------------------------------------------------------
def tok(text):
    return http("/tokenize", {"content": text, "add_special": False}, 120)["tokens"]

#------------------------------------------------------------
#Func : sid
#------------------------------------------------------------
def sid(i):
    return f"dens::u{i}"

#------------------------------------------------------------
#Func : code
#------------------------------------------------------------
def code(i):
    return f"K{i}QQ7"

#------------------------------------------------------------
#Func : stop_engine
#SIGTERM to the process group of the server this program started, SIGKILL after 15 s
#------------------------------------------------------------
def stop_engine():
    global _proc
    if _proc is None:
        return
    try:
        os.killpg(_proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        _proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(_proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _proc.wait()
    _proc = None

#------------------------------------------------------------
#Func : port_busy
#------------------------------------------------------------
def port_busy():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        return s.connect_ex(("127.0.0.1", PORT)) == 0

#------------------------------------------------------------
#Func : restart_engine
#------------------------------------------------------------
def restart_engine():
    global _proc
    stop_engine()
    if port_busy():
        raise RuntimeError(f"port {PORT} is in use by a process this program did not start")
    with open(SRVLOG, "w") as log:
        _proc = subprocess.Popen(ENGINE_ARGS, cwd=REPO or None, env=ENGINE_ENV, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(180):
        time.sleep(1)
        if _proc.poll() is not None:
            raise RuntimeError(f"engine exited with code {_proc.returncode} (log {SRVLOG})")
        try:
            urllib.request.urlopen(ENGINE + "/health", timeout=2)
            print("engine up", flush=True)
            return
        except Exception:
            pass
    raise RuntimeError("engine did not come up")

atexit.register(stop_engine)

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
#Func : build
#------------------------------------------------------------
def build():
    restart_engine()
    t0 = time.time()
    for i in range(TOTAL):
        filler = " ".join(f"u{i} item{j} ok." for j in range(H_REPS))
        p = tok("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                + filler + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nOK.<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\nThe vault code is {code(i)}. "
                "Remember it.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        d = http("/completion", {"prompt": p, "n_predict": 8, "temperature": 0,
                                 "session_id": sid(i), "user_id": "dens", "cache_prompt": True})
        if i % 20 == 0:
            print(f"built {i}/{TOTAL} ({round(time.time()-t0)}s) prompt_n="
                  f"{(d.get('timings') or {}).get('prompt_n')}", flush=True)
    for _ in range(300):
        time.sleep(2)
        n_ext = q("SELECT COUNT(*) FROM qml_llama_session_ext WHERE session_id LIKE 'dens::%'")[0][0]
        if n_ext >= TOTAL:
            break
    print(f"blobs persisted: {n_ext}/{TOTAL}", flush=True)
    n = q("SELECT COUNT(*) FROM qml_session WHERE session_id LIKE 'dens::%'")[0][0]
    tokens = q("SELECT SUM(n_tokens) FROM qml_session WHERE session_id LIKE 'dens::%'")[0][0]
    print(f"build done: {n} sessions, {tokens} tokens total, {round(time.time()-t0)}s", flush=True)

PROBE = ("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
         "What is the vault code? Answer with the code only."
         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

#------------------------------------------------------------
#Func : one_request
#------------------------------------------------------------
def one_request(mode, i, probe_tokens, out):
    t0 = time.time()
    try:
        if mode == "qml":
            body = {"prompt": probe_tokens, "n_predict": 8, "temperature": 0,
                    "session_id": sid(i), "user_id": "dens", "cache_prompt": True}
        else:
            hist = [r[0] for r in q("SELECT token_id FROM qml_token_log WHERE session_id=? ORDER BY pos", (sid(i),))]
            body = {"prompt": hist + probe_tokens, "n_predict": 8, "temperature": 0,
                    "cache_prompt": True}
        d = http("/completion", body, timeout=900)
        dt = time.time() - t0
        content = d.get("content", "")
        out[i] = {"latency": round(dt, 3), "correct": code(i) in content,
                  "prompt_n": (d.get("timings") or {}).get("prompt_n"),
                  "answer": content[:30]}
    except Exception as e:
        out[i] = {"latency": round(time.time() - t0, 3), "correct": False,
                  "prompt_n": None, "answer": f"ERR {type(e).__name__}: {str(e)[:60]}"}

SHORT_Q = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
           "In one word, what color is the clear daytime sky?"
           "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
SHORT_WORKERS = 4

#------------------------------------------------------------
#Func : short_worker
#------------------------------------------------------------
def short_worker(stop, short_tokens, lats):
    while not stop[0]:
        t0 = time.time()
        try:
            http("/completion", {"prompt": short_tokens, "n_predict": 16,
                                 "temperature": 0, "cache_prompt": False}, 300)
            lats.append(time.time() - t0)
        except Exception:
            lats.append(999.0)

#------------------------------------------------------------
#Func : sweep
#------------------------------------------------------------
def sweep(mode):
    restart_engine()
    probe_tokens = tok(PROBE)
    short_tokens = tok(SHORT_Q)
    results = {}
    base = 0
    for n in LEVELS:
        ids = list(range(base, base + n))
        base += n
        out = {}
        stop = [False]
        short_lats = []
        workers = [threading.Thread(target=short_worker, args=(stop, short_tokens, short_lats))
                   for _ in range(SHORT_WORKERS)]
        for w in workers:
            w.start()
        time.sleep(2)
        threads = [threading.Thread(target=one_request, args=(mode, i, probe_tokens, out))
                   for i in ids]
        t0 = time.time()
        for t in threads:
            t.start()
            if STAGGER:
                time.sleep(STAGGER)
        for t in threads:
            t.join()
        wall = round(time.time() - t0, 1)
        stop[0] = True
        for w in workers:
            w.join()
        lats = sorted(out[i]["latency"] for i in ids)
        correct = sum(1 for i in ids if out[i]["correct"])
        s = sorted(short_lats) or [0]
        median = (lats[len(lats)//2] if len(lats) % 2 else (lats[len(lats)//2 - 1] + lats[len(lats)//2]) / 2)
        row = {"N": n, "wall_s": wall,
               "p50": lats[len(lats)//2], "p95": lats[max(0, int(len(lats)*0.95)-1)],
               "median": round(median, 4), "p95_nr": lats[math.ceil(0.95 * len(lats)) - 1],
               "min": lats[0], "max": lats[-1], "lats": lats,
               "per_user": {str(i): out[i] for i in ids}, "correct": correct,
               "mean_prompt_n": round(sum(out[i]["prompt_n"] or 0 for i in ids) / n),
               "short_p50": round(s[len(s)//2], 3),
               "short_p95": round(s[max(0, int(len(s)*0.95)-1)], 3),
               "short_rate": round(len(short_lats) / max(wall, 0.1), 2),
               "short_n": len(short_lats),
               "errors": [out[i]["answer"] for i in ids if not out[i]["correct"]][:3]}
        results[str(n)] = row
        print(f"[{mode}] N={n}: recon median={row['median']}s max={row['max']}s lats={lats} correct={correct}/{n} "
              f"prompt_n={row['mean_prompt_n']} | short p95={row['short_p95']}s "
              f"rate={row['short_rate']}/s (n={row['short_n']}) | wall={wall}s", flush=True)
    all_r = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    all_r[mode] = results
    json.dump(all_r, open(RESULTS, "w"), indent=1)

#------------------------------------------------------------
#Func : cleanup
#------------------------------------------------------------
def cleanup():
    # DROP_BLOB needs a running server on this store: start one if this run has none
    if _proc is None:
        restart_engine()
    pid = _proc.pid   # the server this program started
    con = sqlite3.connect(DB, timeout=30)
    if pid:
        for i in range(TOTAL):
            con.execute("INSERT INTO qml_control_cmd (target_engine,target_pid,cmd_type,session_id,created_at) "
                        "VALUES ('llama.cpp',?, 'DROP_BLOB', ?, strftime('%s','now'))", (pid, sid(i)))
        con.commit()
        time.sleep(30)
    for t in ("qml_token_log", "qml_llama_session_ext", "qml_session"):
        con.execute(f"DELETE FROM {t} WHERE session_id LIKE 'dens::%'")
    con.commit(); con.close()
    print("cleanup done", flush=True)

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"== bench_density_curve {what} | levels={LEVELS} total={TOTAL} h_reps={H_REPS} "
          f"ctx={CTX} stagger={STAGGER} ==", flush=True)
    if what in ("build", "all"):
        build()
    if what in ("qml", "all"):
        sweep("qml")
    if what in ("stock", "all"):
        sweep("stock")
    if what in ("cleanup", "all"):
        cleanup()
    if os.path.exists(RESULTS):
        print(json.dumps(json.load(open(RESULTS)), indent=1))
