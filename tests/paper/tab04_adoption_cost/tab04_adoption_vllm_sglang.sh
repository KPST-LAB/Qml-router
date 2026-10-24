#!/bin/bash
#------------------------------------------------------------
#tab04_adoption_vllm_sglang.sh
#Purpose : vLLM + SGLang adoption cost under the SAME protocol as the llama.cpp run: fresh store AND fresh server for every repetition, so every repetition is a first visit
#------------------------------------------------------------
R=$REPO
DN=$R/tests/_ovh2
OUT=$DN/results_engines_fresh.log
CU13=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib
mkdir -p $DN; : > $OUT
echo "=== vLLM/SGLang adoption cost, fresh store+server per rep (container) $(date) ===" >> $OUT
gmem()
{
    nvidia-smi --query-gpu=memory.used --format=csv,noheader; }
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
# stopall: SIGKILL to the process group of the server this script started (the engine and its
# child processes such as VLLM::EngineCore), as the earlier name-based stop did with -9
stopall()
{
    stop_group $SRV_PID KILL; SRV_PID=""; sleep 8; }
wait_api()
{
    timeout $2 bash -c "until curl -s http://127.0.0.1:$1/v1/models 2>/dev/null | grep -q '\"id\"'; do sleep 5; done"; }

one_vllm()
{
  local tag=$1 rep=$2; shift 2
  stopall
  local ST=$DN/store-fresh-$tag; rm -rf $ST; mkdir -p $ST
  VLLM_PLUGINS="" PYTHONPATH=$R/hooks/vllm setsid nohup python3 -m vllm.entrypoints.openai.api_server \
    --model unsloth/Llama-3.2-3B-Instruct --port 8332 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 "$@" > $DN/vf_$tag$rep.log 2>&1 &
  SRV_PID=$!
  wait_api 8332 900 || { echo "[$tag-r$rep] SERVER FAILED"; tail -6 $DN/vf_$tag$rep.log >> $OUT; return 1; }
  echo "[$tag-r$rep] VRAM=$(gmem) $(grep -m1 -oE 'GPU KV cache size:[^,]*' $DN/vf_$tag$rep.log)" >> $OUT
  python3 $R/tests/adoption-overhead/bench_overhead_client.py 8332 openai $tag-r$rep >> $OUT 2>&1
  stopall
}
one_sgl()
{
  local tag=$1 rep=$2; shift 2
  stopall
  local ST=$DN/store-fresh-$tag; rm -rf $ST; mkdir -p $ST
  LD_LIBRARY_PATH=$CU13:$LD_LIBRARY_PATH PYTHONPATH=$R/hooks/sglang setsid nohup python3 -m sglang.launch_server \
    --model-path unsloth/Llama-3.2-3B-Instruct --port 8333 --context-length 8192 \
    "$@" > $DN/sf_$tag$rep.log 2>&1 &
  SRV_PID=$!
  wait_api 8333 1200 || { echo "[$tag-r$rep] SERVER FAILED"; tail -6 $DN/sf_$tag$rep.log >> $OUT; return 1; }
  echo "[$tag-r$rep] VRAM=$(gmem) $(grep -m1 -oE 'KV Cache is allocated[^.]*' $DN/sf_$tag$rep.log)" >> $OUT
  python3 $R/tests/adoption-overhead/bench_overhead_client.py 8333 openai $tag-r$rep >> $OUT 2>&1
  stopall
}

echo "########## vLLM ##########" >> $OUT
for rep in 1 2 3; do one_vllm vllm-stock $rep; done
for rep in 1 2 3; do
  one_vllm vllm-qml $rep --kv-transfer-config '{"kv_connector":"QmlVllmConnector","kv_connector_module_path":"qml_vllm_connector","kv_role":"kv_both","kv_buffer_device":"cpu","kv_connector_extra_config":{"qml_sqlite_path":"'$DN'/store-fresh-vllm-qml/qml.db","qml_lmdb_path":"'$DN'/store-fresh-vllm-qml/qml_lmdb","qml_self_url":"http://127.0.0.1:8332","qml_read_threads":"8"}}'
done
echo "########## SGLang ##########" >> $OUT
for rep in 1 2 3; do one_sgl sgl-stock $rep; done
for rep in 1 2 3; do one_sgl sgl-hicache $rep --enable-hierarchical-cache; done
for rep in 1 2 3; do
  one_sgl sgl-qml $rep --enable-hierarchical-cache --hicache-storage-backend dynamic \
    --hicache-storage-backend-extra-config '{"module_path":"qml_sglang_storage","class_name":"QmlSglangStorage","backend_name":"qml","qml_sqlite_path":"'$DN'/store-fresh-sgl-qml/qml.db","qml_lmdb_path":"'$DN'/store-fresh-sgl-qml/qml_lmdb"}'
done
echo "ENGINES FRESH PANEL DONE $(date)" >> $OUT
