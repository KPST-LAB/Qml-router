#!/bin/bash
#------------------------------------------------------------
#tab04_adoption_llama.sh
#Purpose : Adoption cost, first-visit condition enforced per repetition: every repetition gets a fresh store AND a fresh server, so neither condition can restore anything
#------------------------------------------------------------
Q=$REPO
B=$Q/third_party/llama.cpp/build/bin/llama-server
DN=$Q/tests/_ovh2
P=8404
OUT=$DN/results_llama_container_fresh.log
mkdir -p $DN; : > $OUT
echo "=== llama.cpp adoption cost, fresh store+server per rep (container) $(date) ===" >> $OUT
one()
{
  local tag=$1 rep=$2; shift 2
  local ST=$DN/store-fresh; rm -rf $ST; mkdir -p $ST
  pkill -x llama-server 2>/dev/null; sleep 3
  cd $Q && nohup $B -m models/Llama-3.2-3B-Instruct-F16.gguf -ngl 99 -fa on \
    -c 65536 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port $P "$@" \
    > $DN/llama_fresh_$tag$rep.log 2>&1 &
  for i in $(seq 1 300); do
    python3 -c "import urllib.request,sys;sys.exit(0 if b'ok' in urllib.request.urlopen('http://127.0.0.1:$P/health',timeout=2).read() else 1)" 2>/dev/null && break
    sleep 2
  done
  sleep 2
  local SPID=$(pgrep -x llama-server | tail -1)
  echo "[$tag-r$rep] gpu_mem_mib=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$SPID" | awk -F', ' '{print $2}' | tr -d ' MiB')" >> $OUT
  python3 $Q/tests/bench_overhead_client.py $P llama ${tag}-r$rep 12 1500 >> $OUT 2>&1
  kill $SPID 2>/dev/null; sleep 4
}
for rep in 1 2 3; do one stock $rep; done
for rep in 1 2 3; do one qml $rep --qml-session-store $DN/store-fresh/qml.db --qml-lmdb-store $DN/store-fresh/qml_lmdb; done
echo "FRESH PANEL DONE $(date)" >> $OUT
