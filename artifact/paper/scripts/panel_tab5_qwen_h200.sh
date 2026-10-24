#!/bin/bash
#------------------------------------------------------------
#panel_tab5_qwen_h200.sh
#Purpose : Non-Llama check for the 500-turn overflow run (reviewer item 6): Qwen3-32B Q8_0 on the H200 with llama.cpp, stock vs qml, same protocol as tests/panel_tab5_crossover_32k.sh (32K window, 2 sessions, 500 turns)
#------------------------------------------------------------
set -u
DATA=${DATA:-$HOME/.DATA}; R=${REPO:-$DATA/qml-router}
B=$DATA/llama.cpp/build/bin/llama-server
M=$DATA/models/Qwen3-32B/Qwen3-32B-Q8_0.gguf
DN=$DATA/qml-bench/_x32k_qwen
MODE=$1; TURNS=${2:-500}; NS=${3:-2}; PORT=58001
TAG=qwen3-32b-$MODE-t$TURNS
OUT=$DN/result_$TAG.log; ST=$DN/store-$TAG
mkdir -p $DN; : > $OUT; rm -rf $ST; mkdir -p $ST
echo "=== x32k $TAG (sessions $NS) $(date) ===" >> $OUT
for p in $(pgrep -x llama-server); do kill -9 $p; done; sleep 3
QF=""; RECALL=""
[ "$MODE" = "qml" ] && QF="--qml-session-store $ST/qml.db --qml-lmdb-store $ST/qml_lmdb" && RECALL="QML_ARCHIVE_RECALL=1"
cd $R && env $RECALL nohup $B -m $M -ngl 99 -fa on -c 32768 --kv-unified -np $NS -ctk q4_0 -ctv q4_0 --cache-ram 0 --port $PORT $QF \
  > $DN/srv_$TAG.log 2>&1 &
for i in $(seq 1 200); do curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && break; sleep 3; done
curl -s -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 || { echo "[$TAG] SERVER FAILED" >> $OUT; tail -5 $DN/srv_$TAG.log >> $OUT; exit 1; }
QML_BENCH_TEMPLATE=qwen3 QML_ENGINE_PID=$(pgrep -x llama-server | tail -1) \
  python3 $R/tests/bench_tab5_crossover_32k.py $PORT llama $MODE $TAG $NS $TURNS 306 $ST >> $OUT 2>&1
echo "[$TAG] store: $(du -sh $ST 2>/dev/null | cut -f1)" >> $OUT
for p in $(pgrep -x llama-server); do kill $p; done; sleep 5
echo "X32K DONE $TAG $(date)" >> $OUT
