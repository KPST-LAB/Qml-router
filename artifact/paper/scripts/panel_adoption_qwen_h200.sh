#!/bin/bash
#------------------------------------------------------------
#panel_adoption_qwen_h200.sh
#Purpose : Non-Llama check for the adoption-cost run (Table 4 protocol): Qwen3-32B Q8_0 on the H200 with llama.cpp; fresh store AND fresh server per repetition, 12 sessions x ~1,500 tokens, stock x3 then qml x3 (bench_overhead_client.py)
#------------------------------------------------------------
set -u
DATA=${DATA:-$HOME/.DATA}; R=${REPO:-$DATA/qml-router}
B=$DATA/llama.cpp/build/bin/llama-server
M=$DATA/models/Qwen3-32B/Qwen3-32B-Q8_0.gguf
DN=$DATA/qml-bench/_ovh_qwen
P=58001
OUT=$DN/results_qwen3_32b_adoption.log
mkdir -p $DN; : > $OUT
echo "=== Qwen3-32B Q8_0 adoption cost, fresh store+server per rep (H200, llama.cpp) $(date) ===" >> $OUT
one()
{
  local tag=$1 rep=$2; shift 2
  local ST=$DN/store-fresh; rm -rf $ST; mkdir -p $ST
  for p in $(pgrep -x llama-server); do kill -9 $p; done; sleep 3
  cd $R && nohup $B -m $M -ngl 99 -fa on -c 65536 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port $P "$@" \
    > $DN/srv_$tag$rep.log 2>&1 &
  for i in $(seq 1 300); do curl -s -m 3 http://127.0.0.1:$P/health 2>/dev/null | grep -q ok && break; sleep 2; done
  sleep 2
  local SPID=$(pgrep -x llama-server | tail -1)
  echo "[$tag-r$rep] gpu_mem_mib=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$SPID" | awk -F', ' '{print $2}' | tr -d ' MiB')" >> $OUT
  python3 $R/tests/bench_overhead_client.py $P llama ${tag}-r$rep 12 1500 >> $OUT 2>&1
  kill $SPID 2>/dev/null; sleep 4
}
for rep in 1 2 3; do one stock $rep; done
for rep in 1 2 3; do one qml $rep --qml-session-store $DN/store-fresh/qml.db --qml-lmdb-store $DN/store-fresh/qml_lmdb; done
echo "ADOPTION PANEL DONE $(date)" >> $OUT
