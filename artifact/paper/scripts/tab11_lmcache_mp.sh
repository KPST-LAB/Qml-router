#!/bin/bash
#------------------------------------------------------------
#tab11_lmcache_mp.sh
#Purpose : Test 7 under LMCache MULTI-PROCESS (MP) mode -- upstream's requested re-test (#4385 reply, 2026-08-04)
#------------------------------------------------------------
R=$REPO/tests/_lm/results7_mp.log
DN=$REPO/tests/_lm
exec > $R 2>&1
echo "=== TEST 7 MP MODE: $(date) ==="
free -g | head -2

rm -rf $DN/lmcache_disk_mp
mkdir -p $DN/lmcache_disk_mp

nohup lmcache server --host 127.0.0.1 --port 5560 \
  --l1-size-gb 8 --eviction-policy LRU \
  --l2-adapter '{"type":"fs","base_path":"$REPO/tests/_lm/lmcache_disk_mp"}' \
  > $DN/lmcache_mp_server.log 2>&1 &
LMC_PID=$!
sleep 10
kill -0 $LMC_PID 2>/dev/null || { echo "LMCACHE MP SERVER FAILED"; tail -20 $DN/lmcache_mp_server.log; exit 2; }
echo "lmcache mp server up (pid $LMC_PID)"

VLLM_PLUGINS="" nohup python3 -m vllm.entrypoints.openai.api_server \
  --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --port 8330 \
  --max-model-len 34000 --gpu-memory-utilization 0.90 \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1","lmcache.mp.port":5560}}' \
  > $DN/vllm_lmc7mp.log 2>&1 &

if ! timeout 600 bash -c 'until curl -s http://127.0.0.1:8330/v1/models 2>/dev/null | grep -q "\"id\""; do sleep 5; done'; then
  echo "VLLM SERVER FAILED TO START"; tail -30 $DN/vllm_lmc7mp.log; kill -9 $LMC_PID; exit 2
fi
echo "vllm up"

python3 - <<'PYEOF'
import json, time, urllib.request
FILLER = ("The development of transcontinental railroads in the nineteenth century "
          "reshaped commerce, migration, and the perception of distance. ")
REPS = 32000 // 23
def history(i):
    tenth = max(1, REPS // 10)
    return (FILLER * tenth + f"IMPORTANT RECORD: the vault code is ZK-{6000+i}. "
            + FILLER * (REPS - tenth))
with urllib.request.urlopen("http://127.0.0.1:8330/v1/models", timeout=30) as r:
    model = json.loads(r.read())["data"][0]["id"]
for i in range(6):
    t0 = time.time()
    req = urllib.request.Request("http://127.0.0.1:8330/v1/completions",
        data=json.dumps({"model": model, "prompt": history(i), "max_tokens": 1,
                         "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        r.read()
    print(f"[build] session {i}: {time.time()-t0:.0f}s", flush=True)
PYEOF

sleep 30
du -sh $DN/lmcache_disk_mp

echo "--- warm mixed load 150s ---"
timeout 600 python3 $REPO/tests/bench_mixed_load.py 8330 stateless-openai $DN 6 32000 150

echo "--- diagnosis: 6 isolated warm reconnects, needle verified ---"
python3 - <<'PYEOF'
import json, urllib.request
FILLER = ("The development of transcontinental railroads in the nineteenth century "
          "reshaped commerce, migration, and the perception of distance. ")
REPS = 32000 // 23
def history(i):
    tenth = max(1, REPS // 10)
    return (FILLER * tenth + f"IMPORTANT RECORD: the vault code is ZK-{6000+i}. "
            + FILLER * (REPS - tenth))
ASK = " Recalling the important record, the vault code is"
with urllib.request.urlopen("http://127.0.0.1:8330/v1/models", timeout=30) as r:
    model = json.loads(r.read())["data"][0]["id"]
wrong = 0
for i in range(6):
    req = urllib.request.Request("http://127.0.0.1:8330/v1/completions",
        data=json.dumps({"model": model, "prompt": history(i) + ASK, "max_tokens": 12,
                         "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    text = d["choices"][0]["text"].strip().replace("\n", " ")
    expected = f"ZK-{6000+i}"
    ok = expected in text
    wrong += 0 if ok else 1
    print(f"[diag {i}] expected={expected}  got=\"{text[:60]}\"  -> {'OK' if ok else 'WRONG'}", flush=True)
print(f"[diag] wrong={wrong}/6", flush=True)
PYEOF

pkill -9 -f "[v]llm.entrypoints"; sleep 2; pkill -9 -f "VLLM::EngineCore"; sleep 3
kill -9 $LMC_PID 2>/dev/null; pkill -9 -f "[l]mcache server" 2>/dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "=== TEST 7 MP DONE ==="
