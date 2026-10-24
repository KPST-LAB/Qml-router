#!/bin/bash
#------------------------------------------------------------
#fig06_longmemeval_full_history_chain.sh
#Purpose : Full-history baseline, resume run: answer the 13 temporal-reasoning questions the 2026-09-04 fullhist run left unprocessed, then re-judge all 266 rows with the official LongMemEval protocol
#------------------------------------------------------------
set -u
DATA=${DATA:-$HOME/.DATA}; cd ${REPO:-$DATA/qml-router}
SRV=$DATA/llama.cpp/build/bin/llama-server
MODEL=$DATA/models/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf
PORT=58001
LOG=fullhist13.log
: > $LOG
cp -n results_fullhist.json results_fullhist.pre13.json
echo "[$(date +%F\ %T)] server start" >> $LOG
nohup $SRV -m $MODEL -ngl 99 -fa on -c 131072 -np 1 --port $PORT > srv_fullhist13.log 2>&1 &
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break
  sleep 5
done
curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 || { echo "[$(date +%F\ %T)] server start failed" >> $LOG; exit 1; }
echo "[$(date +%F\ %T)] server ready, running 13 questions" >> $LOG
python3 fullhist_resume.py $PORT longmemeval_s_cleaned.json results_fullhist.json \
        results_trf2_final_paperjudged.json 131072 >> $LOG 2>&1
echo "[$(date +%F\ %T)] official judge (266 rows)" >> $LOG
python3 tests/bench_longmemeval.py paperjudge $PORT /tmp/nostore results_fullhist.json fullhistj2 0 \
        longmemeval_s_cleaned.json > fullhist13_pj.log 2>&1
tail -6 fullhist13_pj.log >> $LOG
for p in $(pgrep -x llama-server); do kill $p; done
sleep 5
echo "[$(date +%F\ %T)] end" >> $LOG
touch FULLHIST13_DONE
