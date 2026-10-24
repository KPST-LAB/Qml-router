#!/bin/bash
#------------------------------------------------------------
#tab11_qml_connector.sh
#Purpose : Test 8: post-fix qml connector warm mixed load (isolated)
#------------------------------------------------------------
R=$REPO/tests/_lm/results8.log
DN=$REPO/tests/_lm
mkdir -p $DN   # the result file below lives in this folder
exec > $R 2>&1
echo "=== TEST 8: qml connector (async save + pipelined load) warm, $(date) ==="

rm -rf $DN/lmcache_disk
rm -rf $DN/qml_store && mkdir -p $DN/qml_store
df -h / | tail -1 | awk '{print "disk "$4" free at start"}'

cd $DN
SRV_PID=""   # process group of the server this script started (setsid makes the server the group leader)
# stop_group PID [KILL] : stop only the process group this script started with setsid, never by name;
# SIGTERM, then SIGKILL after 30 s; KILL sends SIGKILL at once
stop_group()
{
  local pid=${1:-} k
  [ -n "$pid" ] || return 0
  if [ "${2:-}" = KILL ]; then kill -9 -- -$pid 2>/dev/null; else kill -- -$pid 2>/dev/null; fi
  for k in $(seq 1 30); do kill -0 -- -$pid 2>/dev/null || return 0; sleep 1; done
  kill -9 -- -$pid 2>/dev/null
}
VLLM_PLUGINS="" PYTHONPATH=$REPO/hooks/vllm setsid nohup python3 -m vllm.entrypoints.openai.api_server \
  --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --port 8330 \
  --max-model-len 34000 --gpu-memory-utilization 0.90 \
  --kv-transfer-config '{"kv_connector":"QmlVllmConnector","kv_connector_module_path":"qml_vllm_connector","kv_role":"kv_both","kv_buffer_device":"cpu","kv_connector_extra_config":{"qml_sqlite_path":"'"$REPO"'/tests/_lm/qml_store/qml.db","qml_lmdb_path":"'"$REPO"'/tests/_lm/qml_store/qml_lmdb","qml_self_url":"http://127.0.0.1:8330","qml_read_threads":"8"}}' \
  > $DN/vllm_qml8.log 2>&1 &
SRV_PID=$!

if ! timeout 600 bash -c 'until curl -s http://127.0.0.1:8330/v1/models 2>/dev/null | grep -q "\"id\""; do sleep 5; done'; then
  echo "SERVER FAILED TO START"; stop_group $SRV_PID; exit 2
fi
echo "server up"

python3 - <<'PYEOF'
import json, os, time, urllib.request
# QML_BENCH_SALT=1: send cache_salt = mix-N, the same id bench_mixed_load.py sends for session N
SALT = lambda i: {"cache_salt": f"mix-{i}"} if os.environ.get("QML_BENCH_SALT", "0") == "1" else {}
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
        data=json.dumps(dict({"model": model, "prompt": history(i), "max_tokens": 1,
                              "temperature": 0}, **SALT(i))).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        r.read()
    print(f"[build] session {i}: {time.time()-t0:.0f}s", flush=True)
PYEOF

timeout 600 bash -c 'prev=0; while true; do cur=$(du -sb '$DN'/qml_store/qml_lmdb 2>/dev/null | cut -f1); if [ "$cur" = "$prev" ] && [ "$cur" != "0" ]; then break; fi; prev=$cur; sleep 10; done'
du -sh $DN/qml_store/qml_lmdb

echo "--- warm mixed load 150s ---"
timeout 600 python3 $REPO/tests/mixed-load/bench_mixed_load.py 8330 stateless-openai $DN 6 32000 150
df -h / | tail -1 | awk '{print "disk "$4" free at end"}'
stop_group $SRV_PID   # the script used to leave this vLLM server running
echo "=== TEST 8 DONE ==="
