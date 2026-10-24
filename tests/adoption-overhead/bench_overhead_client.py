#!/usr/bin/env python3
"""bench_overhead_client.py

Purpose : Fixed, engine-neutral client workload for measuring qml's per-request overhead: N sequential needle sessions of ~TOKENS tokens each, every answer verified
Usage   : bench_overhead_client.py PORT API TAG [N_SESSIONS] [TOKENS]
Usage   : python3 bench_overhead_client.py PORT API TAG [N_SESSIONS=12]
"""
import json
import statistics
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
API = sys.argv[2]
TAG = sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 12
TOKENS = int(sys.argv[5]) if len(sys.argv) > 5 else 1500

BASE = f"http://127.0.0.1:{PORT}"
FILLER = ("The development of transcontinental railroads in the nineteenth century "
          "reshaped commerce, migration, and the perception of distance. ")
ASK = " Recalling the important record, the vault code is"
REPS = TOKENS // 23

#------------------------------------------------------------
#Func : history
#------------------------------------------------------------
def history(i):
    tenth = max(1, REPS // 10)
    return (FILLER * tenth + f"IMPORTANT RECORD: the vault code is ZK-{8200 + i}. "
            + FILLER * (REPS - tenth))

#------------------------------------------------------------
#Func : post
#------------------------------------------------------------
def post(path, payload, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

model = None
if API == "openai":
    model = post("/v1/models", None) if False else json.loads(
        urllib.request.urlopen(BASE + "/v1/models", timeout=60).read())["data"][0]["id"]

lat, ok_n, prefill_total = [], 0, 0
t_wall = time.time()
for i in range(N):
    t0 = time.time()
    if API == "llama":
        d = post("/completion", {"prompt": history(i) + ASK, "n_predict": 12,
                                 "temperature": 0, "cache_prompt": True,
                                 "session_id": f"user::overhead-{i}"})
        text = d.get("content", "")
        prefill_total += int((d.get("timings") or {}).get("prompt_n") or 0)
    else:
        d = post("/v1/completions", {"model": model, "prompt": history(i) + ASK,
                                     "max_tokens": 12, "temperature": 0,
                                     "cache_salt": f"user::overhead-{i}"})
        text = d["choices"][0]["text"]
        prefill_total += int((d.get("usage") or {}).get("prompt_tokens") or 0)
    lat.append((time.time() - t0) * 1000.0)
    ok_n += 1 if f"ZK-{8200 + i}" in text else 0
wall = time.time() - t_wall

lat.sort()
print(f"[{TAG}] {N} sessions x ~{TOKENS} tokens: wall {wall:,.1f}s  "
      f"mean {statistics.mean(lat):,.0f} ms  p50 {lat[len(lat) // 2]:,.0f} ms  "
      f"GPU prefill total {prefill_total:,}  correct {ok_n}/{N}", flush=True)
