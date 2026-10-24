#!/bin/bash
# Non-Llama check for the adoption-cost run (Table 4 protocol), size-matched:
# Qwen3-4B F16 GGUF on the RTX 4090 with llama.cpp, in the container
# environment of Table 4 (same as tests/_ovh2 for Llama-3.2-3B). Fresh store
# AND fresh server per repetition, 12 sessions x ~1,500 tokens, stock x3 then
# qml x3 (bench_overhead_client.py; raw-text prompts, so no chat template).
set -u
Q=$REPO
B=$Q/third_party/llama.cpp/build/bin/llama-server
M=models/Qwen3-4B-F16.gguf
DN=$Q/tests/_ovh_qwen4b
P=8404
OUT=$DN/results_qwen3_4b_adoption.log
mkdir -p $DN; : > $OUT
echo "=== Qwen3-4B F16 adoption cost, fresh store+server per rep (RTX 4090 container, llama.cpp) $(date) ===" >> $OUT
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader >> $OUT
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
one(){
  local tag=$1 rep=$2; shift 2
  local ST=$DN/store-fresh; rm -rf $ST; mkdir -p $ST
  stop_group $SRV_PID; sleep 3
  cd $Q
  setsid env LD_LIBRARY_PATH=$Q/deps/lib:${LD_LIBRARY_PATH:-} nohup $B -m $M -ngl 99 -fa on \
    -c 65536 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port $P "$@" \
    > $DN/srv_$tag$rep.log 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 300); do
    python3 -c "import urllib.request,sys;sys.exit(0 if b'ok' in urllib.request.urlopen('http://127.0.0.1:$P/health',timeout=2).read() else 1)" 2>/dev/null && break
    sleep 2
  done
  sleep 2
  local SPID=$SRV_PID
  echo "[$tag-r$rep] gpu_mem_mib=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$SPID" | awk -F', ' '{print $2}' | tr -d ' MiB')" >> $OUT
  python3 $Q/tests/adoption-overhead/bench_overhead_client.py $P llama ${tag}-r$rep 12 1500 >> $OUT 2>&1
  stop_group $SPID; SRV_PID=""; sleep 4
}
for rep in 1 2 3; do one stock $rep; done
for rep in 1 2 3; do one qml $rep --qml-session-store $DN/store-fresh/qml.db --qml-lmdb-store $DN/store-fresh/qml_lmdb; done
echo "ADOPTION PANEL DONE $(date)" >> $OUT
