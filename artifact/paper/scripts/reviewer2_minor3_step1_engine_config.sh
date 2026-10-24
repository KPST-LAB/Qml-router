#!/bin/bash
# TEST-LEDGER 109 (2026-10-19): R2-3 control. Table 7 procedure (tab07_hw_nvme.sh, 32K only),
# engine configuration A (Table 7: -c 34048 -np 1) versus B (Table 6: -c 131072 --kv-unified -np 8).
R=$REPO
B=$R/third_party/llama.cpp/build/bin
M=$R/models/Llama-3.2-3B-Instruct-F16.gguf
W=$R/r2-3-check
LOG=$W/results_r23.log
PORT=8497
TOK=32000
QML_LIBS=$R/env/lib:/usr/local/cuda/lib64
QF="-ctk q4_0 -ctv q4_0 --cache-ram 0"
exec > $LOG 2>&1
echo "=== R2-3 CONTROL $(date) ==="
hostname; nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
wait_up() { local i=0; while [ $i -lt 200 ]; do curl -s http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 'ok\|status' && return 0; pgrep -f "[l]lama-server.*$PORT" >/dev/null || return 2; sleep 2; i=$((i+1)); done; return 1; }
launch() { local tag=$1; shift; env LD_LIBRARY_PATH=$QML_LIBS $B/llama-server -m $M -ngl 99 -fa on --port $PORT "$@" > $W/srv_$tag.log 2>&1 & wait_up; }
stop() { pkill -9 -f "[l]lama-server.*$PORT"; sleep 3; }
settle() { local prev=0 i=0; while [ $i -lt 120 ]; do cur=$(du -sb $S/qml_lmdb 2>/dev/null | cut -f1); [ "$cur" = "$prev" ] && [ "$cur" != "0" ] && [ -n "$cur" ] && return 0; prev=$cur; sleep 3; i=$((i+1)); done; }
for CFG in A B; do
  if [ $CFG = A ]; then EC="-c $((TOK + 2048)) -np 1"; else EC="-c 131072 --kv-unified -np 8"; fi
  S=$W/store_$CFG; rm -rf $S; mkdir -p $S
  echo ""; echo "########## config $CFG: $EC ##########"
  stop
  launch b$CFG $EC $QF --qml-session-store $S/qml.db --qml-lmdb-store $S/qml_lmdb || { echo "SERVER FAILED build $CFG"; continue; }
  python3 $R/tests/bench_long_context.py $PORT build $S $TOK $W/srv_b$CFG.log
  settle; du -sh $S/qml_lmdb; stop
  launch c$CFG $EC $QF --qml-session-store $S/qml.db --qml-lmdb-store $S/qml_lmdb || { echo "SERVER FAILED cold $CFG"; continue; }
  echo "--- qml restore, page cache dropped (disk cold) ---"
  python3 $R/tests/bench_long_context.py $PORT restore $S $TOK $W/srv_c$CFG.log
  stop
  launch p$CFG $EC $QF || { echo "SERVER FAILED stock $CFG"; continue; }
  echo "--- stock prefill (qml not applied) ---"
  python3 $R/tests/bench_long_context.py $PORT recompute $S $TOK
  stop
done
echo ""; echo "=== R2-3 CONTROL DONE $(date) ==="
