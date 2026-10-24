#!/bin/bash
#------------------------------------------------------------
#tab04_adoption_orin.sh
#Purpose : Orin adoption cost, first-visit condition per repetition: fresh store and a fresh server every repetition
#------------------------------------------------------------
Q=$REPO
B=$Q/third_party/llama.cpp/build/bin/llama-server
DN=$Q/tests/_ovh_orin
P=8405
OUT=$DN/results_orin_overhead.log
mkdir -p $DN; : > $OUT
echo "=== Orin adoption cost (fresh store+server per rep) $(date) ===" >> $OUT
echo "power mode: $(nvpmodel -q 2>/dev/null | grep -i 'power mode')" >> $OUT
df -h $NVME_MOUNT | tail -1 >> $OUT
one()
{
  local tag=$1 rep=$2; shift 2
  local ST=$DN/store-fresh; rm -rf $ST; mkdir -p $ST
  pkill -x llama-server 2>/dev/null; sleep 4
  cd $Q && nohup $B -m models/Llama-3.2-3B-Instruct-F16.gguf -ngl 99 -fa on \
    -c 65536 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port $P "$@" \
    > $DN/srv_$tag$rep.log 2>&1 &
  local ok=0
  for i in $(seq 1 200); do
    if curl -s -m 3 "http://127.0.0.1:$P/health" 2>/dev/null | grep -q ok; then ok=1; break; fi
    sleep 3
  done
  if [ $ok -eq 0 ]; then echo "[$tag-r$rep] SERVER FAILED" >> $OUT; tail -4 $DN/srv_$tag$rep.log >> $OUT; return 1; fi
  sleep 2
  local SPID=$(pgrep -x llama-server | tail -1)
  echo "[$tag-r$rep] gpu_mem_mib=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep "^$SPID" | awk -F', ' '{print $2}' | tr -d ' MiB')" >> $OUT
  python3 $Q/tests/bench_overhead_client.py $P llama ${tag}-r$rep 12 1500 >> $OUT 2>&1
  kill $SPID 2>/dev/null; sleep 4
}
for rep in 1 2 3; do one stock $rep; done
for rep in 1 2 3; do one qml $rep --qml-session-store $DN/store-fresh/qml.db --qml-lmdb-store $DN/store-fresh/qml_lmdb; done
echo "ORIN PANEL DONE $(date)" >> $OUT
