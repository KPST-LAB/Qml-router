#!/usr/bin/env python3
"""bench_crossover.py

Purpose : Measures the reconnect-latency versus conversation-length crossover: stock (full recompute of the resent history) versus qml (session restore) at approximately 2K/8K/16K/32K/48K tokens, needle-verified, with a cold engine restart before each condition
Usage   : [DENS_CTX=131072] python3 bench_crossover.py
Settings: QML_BENCH_SERVER (llama-server path), QML_BENCH_MODEL (model file), QML_BENCH_STORE (store folder,
          default $REPO/store-tplseed), QML_BENCH_ENV (extra environment assignments for the server),
          QML_BENCH_PORT (default 8397), QML_BENCH_LOG (server log), QML_BENCH_RESULTS (result JSON);
          the program stops only the server process it started itself (its process group), never by name
"""
import atexit, json, os, shlex, signal, socket, sqlite3, subprocess, sys, time, urllib.request

REPO = os.environ["REPO"]   # repository root on the storage medium under test
PORT = int(os.environ.get("QML_BENCH_PORT", "8397"))
ENGINE = f"http://127.0.0.1:{PORT}"
SERVER = os.environ.get("QML_BENCH_SERVER", f"{REPO}/third_party/llama.cpp/build/bin/llama-server")
MODEL = os.environ.get("QML_BENCH_MODEL", f"{REPO}/models/Llama-3.2-3B-Instruct-F16.gguf")
STORE = os.environ.get("QML_BENCH_STORE", f"{REPO}/store-tplseed")
DB = f"{STORE}/qml.db"
ENVSTR = os.environ.get("QML_BENCH_ENV", f"LD_LIBRARY_PATH={REPO}/env/lib:/usr/local/cuda/lib64")
SRVLOG = os.environ.get("QML_BENCH_LOG", f"{REPO}/srv_density.log")
CTX = int(os.environ.get("DENS_CTX", "131072"))
SIZES = [300, 1200, 2400, 4800, 7200]
RESULTS = os.environ.get("QML_BENCH_RESULTS", "/tmp/crossover_results.json")
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
        _proc = subprocess.Popen(ENGINE_ARGS, cwd=REPO, env=ENGINE_ENV, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(180):
        time.sleep(1)
        if _proc.poll() is not None:
            raise RuntimeError(f"engine exited with code {_proc.returncode} (log {SRVLOG})")
        try:
            urllib.request.urlopen(ENGINE + "/health", timeout=2)
            return
        except Exception:
            pass
    raise RuntimeError("engine down")

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

pid = _proc.pid if _proc is not None else 0   # the server this program started
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
