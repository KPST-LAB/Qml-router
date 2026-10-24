#!/bin/bash
#------------------------------------------------------------
#fig06_longmemeval_full_history_chain.sh
#Purpose : Full-history baseline, resume run: answer the 13 temporal-reasoning questions the 2026-09-04 fullhist run left unprocessed, then re-judge all 266 rows with the official LongMemEval protocol
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
LOG=fullhist13.log
: > $LOG
cp -n results_fullhist.json results_fullhist.pre13.json
echo "[$(date +%F\ %T)] server start" >> $LOG
setsid nohup $SRV -m $MODEL -ngl 99 -fa on -c 131072 -np 1 --port $PORT > srv_fullhist13.log 2>&1 &
SRV_PID=$!
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break
  sleep 5
done
curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 || { echo "[$(date +%F\ %T)] server start failed" >> $LOG; exit 1; }
echo "[$(date +%F\ %T)] server ready, running 13 questions" >> $LOG
python3 tests/longmemeval/lme_fullhist_resume.py $PORT longmemeval_s_cleaned.json results_fullhist.json \
        results_trf2_final_paperjudged.json 131072 >> $LOG 2>&1
echo "[$(date +%F\ %T)] official judge (266 rows)" >> $LOG
python3 tests/longmemeval/bench_longmemeval.py paperjudge $PORT /tmp/nostore results_fullhist.json fullhistj2 0 \
        longmemeval_s_cleaned.json > fullhist13_pj.log 2>&1
tail -6 fullhist13_pj.log >> $LOG
stop_group $SRV_PID
sleep 5
echo "[$(date +%F\ %T)] end" >> $LOG
touch FULLHIST13_DONE
