#!/bin/bash
# Non-Llama check for the 500-turn overflow run (Table 5 protocol), size-matched:
# Qwen3-4B F16 GGUF on the RTX 4090 with llama.cpp, stock vs qml, same protocol
# and same host/container/storage as tests/paper/tab05_context_overflow_500turn/tab05_500turn_4090.sh (llama)
# (the Table 5 container, store on NVMe, 32K window, 2 sessions, 500 turns,
# q4_0 KV). Only the model differs from the Llama-3.2-3B row.
# usage: panel_tab5_qwen4b_4090.sh MODE [TURNS] [SESSIONS]     MODE = stock | qml
set -u
R=$REPO
B=$R/third_party/llama.cpp/build/bin/llama-server
M=models/Qwen3-4B-F16.gguf
DN=$BENCH/_x32k_qwen4b
MODE=$1; TURNS=${2:-500}; NS=${3:-2}; PORT=8408
TAG=qwen3-4b-$MODE-t$TURNS
OUT=$DN/result_$TAG.log; ST=$DN/store-$TAG
mkdir -p $DN; : > $OUT; rm -rf $ST; mkdir -p $ST
echo "=== x32k $TAG (sessions $NS) $(date) ===" >> $OUT
df --output=source $DN | tail -1 >> $OUT
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
sleep 3   # (no name-based stop of other llama-server processes before the run)
QF=""; RECALL=""
[ "$MODE" = "qml" ] && QF="--qml-session-store $ST/qml.db --qml-lmdb-store $ST/qml_lmdb" && RECALL="QML_ARCHIVE_RECALL=1"
cd $R
setsid env $RECALL LD_LIBRARY_PATH=$R/deps/lib:${LD_LIBRARY_PATH:-} nohup $B -m $M -ngl 99 -fa on \
  -c 32768 --kv-unified -np $NS -ctk q4_0 -ctv q4_0 --cache-ram 0 --port $PORT $QF \
  > $DN/srv_$TAG.log 2>&1 &
SRV_PID=$!
for i in $(seq 1 200); do curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && break; sleep 3; done
curl -s -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 || { echo "[$TAG] SERVER FAILED" >> $OUT; tail -5 $DN/srv_$TAG.log >> $OUT; exit 1; }
QML_BENCH_TEMPLATE=qwen3 QML_ENGINE_PID=$SRV_PID \
  python3 $R/tests/multi-turn/continuous-session/bench_tab5_crossover_32k.py $PORT llama $MODE $TAG $NS $TURNS 306 $ST >> $OUT 2>&1
echo "[$TAG] store: $(du -sh $ST 2>/dev/null | cut -f1)" >> $OUT
stop_group $SRV_PID; sleep 5
echo "X32K DONE $TAG $(date)" >> $OUT
