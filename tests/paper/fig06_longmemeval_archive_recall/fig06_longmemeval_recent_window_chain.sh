#!/bin/bash
#------------------------------------------------------------
#fig06_longmemeval_recent_window_chain.sh
#Purpose : Recent-window baselines for the LongMemEval table: stateless Llama-3.3-70B, windows 594 and 32768 tokens, official paperjudge
#------------------------------------------------------------
set -u
cd $DATA/qml-router
SRV=$DATA/llama.cpp/build/bin/llama-server
MODEL=$DATA/models/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf
PORT=58001
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
LOG=recent_window.log
: > $LOG
rm -f RECENT_DONE
echo "[$(date +%F\ %T)] server start" >> $LOG
setsid nohup $SRV -m $MODEL -ngl 99 -fa on -c 131072 -np 1 --port $PORT > srv_recent.log 2>&1 &
SRV_PID=$!
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break
  sleep 5
done
curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 || { echo "[$(date +%F\ %T)] server start failed" >> $LOG; exit 1; }
for W in 594 32768; do
  echo "[$(date +%F\ %T)] window $W run" >> $LOG
  python3 tests/longmemeval/lme_recent_window.py $PORT longmemeval_s_cleaned.json results_recent$W.json $W >> $LOG 2>&1
  echo "[$(date +%F\ %T)] window $W official judge" >> $LOG
  python3 tests/longmemeval/bench_longmemeval.py paperjudge $PORT /tmp/nostore results_recent$W.json recent$W 0 \
          longmemeval_s_cleaned.json > recent${W}_pj.log 2>&1
  tail -6 recent${W}_pj.log >> $LOG
done
stop_group $SRV_PID
sleep 5
echo "[$(date +%F\ %T)] end" >> $LOG
touch RECENT_DONE
