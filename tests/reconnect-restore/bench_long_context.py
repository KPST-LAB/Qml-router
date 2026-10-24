#!/usr/bin/env python3
"""bench_long_context.py

Purpose : Long-context resume cost at ~TOKENS tokens: qml KV restore versus full recompute (prefill), plus supporting probes (warm restore, raw disk read bandwidth, hot-slot request cost); needle-verified
Usage   : bench_long_context.py PORT MODE STORE_DIR TOKENS [LOG]
Usage   : python3 bench_long_context.py PORT MODE STORE_DIR TOKENS [LOG]
"""
import glob
import json
import os
import sqlite3
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
MODE = sys.argv[2]
STORE = sys.argv[3]
TOKENS = int(sys.argv[4])
LOGFILE = sys.argv[5] if len(sys.argv) > 5 else f"{STORE}/server.log"

BASE = f"http://127.0.0.1:{PORT}"
FILLER = ("The development of transcontinental railroads in the nineteenth century "
          "reshaped commerce, migration, and the perception of distance. ")
NEEDLE = "ZK-7777"
ASK = " Recalling the important record, the vault code is"
REPS = max(4, TOKENS // 23)
SID = f"long-{TOKENS}"

#------------------------------------------------------------
#Func : history
#------------------------------------------------------------
def history():
    tenth = max(1, REPS // 10)
    return (FILLER * tenth + f"IMPORTANT RECORD: the vault code is {NEEDLE}. "
            + FILLER * (REPS - tenth))

#------------------------------------------------------------
#Func : completion
#------------------------------------------------------------
def completion(payload, timeout=3600):
    req = urllib.request.Request(BASE + "/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

#------------------------------------------------------------
#Func : resident_bytes
#How much of this file is currently in the OS page cache, via mincore()
#------------------------------------------------------------
def resident_bytes(path):
    import ctypes
    size = os.path.getsize(path)
    if size == 0:
        return 0
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    PROT_READ, MAP_SHARED = 0x1, 0x01
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
        if addr in (None, 0) or addr == (1 << 64) - 1:
            return -1
        try:
            pagesz = os.sysconf("SC_PAGE_SIZE")
            npages = (size + pagesz - 1) // pagesz
            vec = (ctypes.c_ubyte * npages)()
            if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
                return -1
            return sum(1 for b in vec if b & 1) * pagesz
        finally:
            libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
    finally:
        os.close(fd)

#------------------------------------------------------------
#Func : drop_page_cache
#Evict the LMDB data pages from the OS page cache -> disk-cold restore
#------------------------------------------------------------
def drop_page_cache():
    paths = sorted(p for p in glob.glob(f"{STORE}/qml_lmdb/*.mdb")
                   if os.path.getsize(p) > 1 << 20)
    before = {p: resident_bytes(p) for p in paths}
    os.sync()
    for path in paths:
        for _ in range(3):
            fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            if resident_bytes(path) < (8 << 20):
                break
            os.sync()
            time.sleep(1)
    for path in paths:
        b, a = before[path], resident_bytes(path)
        print(f"[page cache] {os.path.basename(path)}: resident {b / 2**20:,.0f} MiB "
              f"-> {a / 2**20:,.0f} MiB "
              f"({'cold confirmed' if a < (8 << 20) else 'NOT COLD -- this measurement is void'})",
              flush=True)

#------------------------------------------------------------
#Func : build
#------------------------------------------------------------
def build():
    t0 = time.time()
    d = completion({"prompt": history(), "n_predict": 1, "temperature": 0,
                    "session_id": SID, "cache_prompt": True})
    pn = int((d.get("timings") or {}).get("prompt_n") or 0)
    print(f"[build] prefilled {pn:,} tokens in {time.time() - t0:.0f}s", flush=True)

    db = f"{STORE}/qml.db"
    prev, size = -1, 0
    while True:
        time.sleep(2)
        try:
            con = sqlite3.connect(db)
            row = con.execute(
                "SELECT e.state_size FROM qml_session s "
                "JOIN qml_llama_session_ext e ON s.session_id = e.session_id "
                "WHERE s.session_id = ?", (SID,)).fetchone()
            con.close()
        except sqlite3.OperationalError:
            continue
        if row is not None:
            size = row[0]
            if size == prev and size > 0:
                break
            prev = size
    print(f"[build] stored state: {size / 2**30:.2f} GiB for {pn:,} tokens "
          f"({size // max(1, pn) // 1024} KiB/token)", flush=True)

#------------------------------------------------------------
#Func : ask
#------------------------------------------------------------
def ask(payload_extra, label):
    t0 = time.time()
    d = completion({"n_predict": 12, "temperature": 0, "cache_prompt": True,
                    **payload_extra})
    el = (time.time() - t0) * 1000.0
    text = d.get("content", "")
    ok = NEEDLE in text
    tm = d.get("timings") or {}
    pn = int(tm.get("prompt_n") or 0)
    print(f"[{label}] latency={el:,.0f} ms  correct={'OK' if ok else 'WRONG(' + text.strip()[:40] + ')'}  "
          f"GPU prefill={pn:,} tokens", flush=True)
    p_ms = float(tm.get("prompt_ms") or 0.0)
    g_ms = float(tm.get("predicted_ms") or 0.0)
    print(f"[{label}] server compute: prompt={p_ms:,.0f} ms + predict={g_ms:,.0f} ms "
          f"= {p_ms + g_ms:,.0f} ms; non-compute remainder={el - p_ms - g_ms:,.0f} ms",
          flush=True)

#------------------------------------------------------------
#Func : log_count
#------------------------------------------------------------
def log_count(marker):
    try:
        with open(LOGFILE, errors="ignore") as f:
            return f.read().count(marker)
    except OSError:
        return 0

if MODE == "build":
    build()
elif MODE == "restore":
    drop_page_cache()
    before = log_count("restored session")
    ask({"prompt": ASK, "session_id": SID}, f"restore-{TOKENS}")
    after = log_count("restored session")
    print(f"[restore-{TOKENS}] server log kv_restore events: +{after - before} (provenance gate)",
          flush=True)
elif MODE == "restore_warm":
    before = log_count("restored session")
    ask({"prompt": ASK, "session_id": SID}, f"restore_warm-{TOKENS}")
    after = log_count("restored session")
    print(f"[restore_warm-{TOKENS}] server log kv_restore events: +{after - before} (provenance gate)",
          flush=True)
elif MODE == "diskbw":
    import ctypes as _ct
    probe = f"{STORE}/bwprobe.bin"
    SZ = 2 << 30
    buf = os.urandom(8 << 20)
    with open(probe, "wb") as f:
        for _ in range(SZ // len(buf)):
            f.write(buf)
        f.flush()
        os.fsync(f.fileno())
    os.sync()
    rates = []
    for rep in range(3):
        fd = os.open(probe, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        res = resident_bytes(probe)
        n, t0 = 0, time.time()
        with open(probe, "rb", buffering=0) as f:
            while True:
                b = f.read(8 << 20)
                if not b:
                    break
                n += len(b)
        el = time.time() - t0
        rates.append(n / el / 1e9)
        print(f"[diskbw] run {rep + 1}: {n / 2**30:.2f} GiB in {el:.2f}s = "
              f"{n / el / 1e9:.2f} GB/s (resident before read {res / 2**20:,.0f} MiB)", flush=True)
    os.remove(probe)
    rates.sort()
    print(f"[diskbw] median {rates[len(rates) // 2]:.2f} GB/s  "
          f"min {rates[0]:.2f}  max {rates[-1]:.2f}", flush=True)
elif MODE == "hot":
    ask({"prompt": ASK + " Answer again: the vault code is", "session_id": SID},
        f"hot-{TOKENS}")
elif MODE == "recompute":
    ask({"prompt": history() + ASK}, f"recompute-{TOKENS}")
else:
    sys.exit(f"unknown mode {MODE}")
