#!/bin/bash
#------------------------------------------------------------
#fig06_longmemeval_archive_recall.sh
#Purpose : reconstructed 2026-09-07: the k=32 runs (conditions msf and trf) were started as direct command lines on the H200, in the form of the earlier wrapper run_lme70_paper.sh; settings are those recorded in TEST-LEDGER rows 82-85 (facts-row retrieval, top-k request 50 = 32 effective because of the hook's fixed upper bound, date inline, question date, 96-token output cap)
#------------------------------------------------------------
set -u
REPO=${REPO:-$HOME/qml-router}; DATA=${DATA:-$HOME/.DATA}
cd $REPO
MODEL=$DATA/models/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf
SRV=$DATA/llama.cpp/build/bin/llama-server
STORE=$REPO/store-lme-full
DS=$REPO/longmemeval_s_cleaned.json
PORT=58001

QML_ARCHIVE_RECALL=1 QML_ARCHIVE_RECALL_FACTS=1 QML_ARCHIVE_RECALL_TOPK=50 \
nohup $SRV -m $MODEL -ngl 99 -fa on -c 32768 -np 2 --port $PORT \
  --qml-session-store $STORE/qml.db --qml-lmdb-store $STORE/qml_lmdb </dev/null > srv_fullA.log 2>&1 &
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && break; sleep 5; done

run_arm()
{
  export LME_TYPES="$2" LME_TOPK=50 LME_NPRED=96 LME_DATE_INLINE=1 LME_QDATE=1
  python3 tests/bench_longmemeval.py ingest     $PORT $STORE $DS $1 0 > $1_ing.log 2>&1
  python3 tests/bench_longmemeval.py ask        $PORT $STORE $DS $1 0 > $1_ask.log 2>&1
  python3 tests/bench_longmemeval.py paperjudge $PORT $STORE results_$1.json $1 0 $DS > $1_pj.log 2>&1
}
run_arm msf multi-session
run_arm trf temporal-reasoning
for p in $(pgrep -x llama-server); do kill $p; done
