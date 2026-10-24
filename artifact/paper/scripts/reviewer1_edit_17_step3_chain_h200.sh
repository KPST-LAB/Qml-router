#!/bin/bash
#------------------------------------------------------------
#reviewer1_edit_17_step3_chain_h200.sh
#Purpose : H200 chain: reviewer1_edit_17 step3 followed by reviewer1_edit_16 step1
#Record  : TEST-LEDGER 100
#------------------------------------------------------------
set -u
D=$DATA; Q=$D/qml-router; R=$Q/ieee-rev1; L=$R/logs; OUT=$R/data
S=$Q/store-lmes70; DS=$Q/longmemeval_s_cleaned.json
M=$D/models/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf
SRV=$D/llama.cpp/build/bin/llama-server; EMB_MODEL=$D/models/bge-m3-f16.gguf
PORT=58117; EPORT=58172
exec > >(tee -a $L/reviewer1_edit_17_step3_chain_h200.log) 2>&1
echo "=== step3 chain start $(date -Is) ==="
md5sum $SRV $M $EMB_MODEL $R/scripts/reviewer1_edit_17_step3_e2e_embedding.py $R/scripts/reviewer1_edit_16_step1_oracle_context.py
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] || { echo "ABORT: another GPU process is running"; nvidia-smi; exit 2; }
PIDS=""
up()
{
  local tag=$1; shift
  $SRV "$@" </dev/null > $L/${tag}_srv.log 2>&1 &
  local p=$!; PIDS="$PIDS $p"
  local port=$(echo "$@" | grep -o -- "--port [0-9]*" | awk '{print $2}')
  for i in $(seq 1 900); do curl -s -m 2 http://127.0.0.1:$port/health | grep -q ok && break; sleep 2; done
  curl -s -m 2 http://127.0.0.1:$port/health | grep -q ok || { echo "ABORT: server $tag did not come up"; stop_all; exit 3; }
  echo "server up $tag pid=$p $(date -Is)"; echo $p
}
stop_all()
{
    for p in $PIDS; do kill $p 2>/dev/null; wait $p 2>/dev/null; done; echo "servers stopped:$PIDS $(date -Is)"; PIDS=""; }
trap stop_all EXIT

EP=$(up reviewer1_edit_17_step3_emb -m $EMB_MODEL --embedding -ngl 99 -c 8192 -ub 8192 -np 1 --port $EPORT | tail -1)
RP=$(up reviewer1_edit_17_step3_reader -m $M -ngl 99 -fa on -c 32768 -np 2 --port $PORT --qml-session-store $S/qml.db --qml-lmdb-store $S/qml_lmdb | tail -1)
python3 -u $R/scripts/reviewer1_edit_17_step3_e2e_embedding.py $PORT $EPORT $OUT > $L/reviewer1_edit_17_step3_e2e_embedding.log 2>&1
echo "step3 python rc=$? $(date -Is)"; grep -E "reproduction|string-match" $L/reviewer1_edit_17_step3_e2e_embedding.log
kill $EP 2>/dev/null; wait $EP 2>/dev/null; PIDS=" $RP"
for condition in lex emb; do for t in msf trf; do
  typ=multi-session; [ $t = trf ] && typ=temporal-reasoning
  LME_TYPES=$typ timeout 3600 python3 -u $Q/tests/bench_longmemeval.py paperjudge $PORT $S $OUT/results_step3_${condition}_${t}.json s3${condition}${t}j 0 $DS > $L/reviewer1_edit_17_step3_judge_${condition}_${t}.log 2>&1
  echo "judge $condition $t rc=$? $(date -Is)"; grep -A3 OFFICIAL $L/reviewer1_edit_17_step3_judge_${condition}_${t}.log | tail -2
done; done
for condition in lex emb; do for t in msf trf; do mv $OUT/results_step3_${condition}_${t}.json $OUT/reviewer1_edit_17_step3_results_${condition}_${t}.json; mv $OUT/results_step3_${condition}_${t}_paperjudged.json $OUT/reviewer1_edit_17_step3_results_${condition}_${t}_paperjudged.json; done; done
stop_all
echo "=== step3 done $(date -Is) ==="

echo "=== oracle start $(date -Is) ==="
RP=$(up reviewer1_edit_16_step1_reader -m $M -ngl 99 -fa on -c 65536 -np 2 --port $PORT | tail -1)
python3 -u $R/scripts/reviewer1_edit_16_step1_oracle_context.py $PORT $DS \
  $R/data/reviewer1_edit_17_step0_results_msf.json $R/data/reviewer1_edit_17_step0_results_trf.json \
  $OUT/reviewer1_edit_16_step1_oracle_results.json 32768 > $L/reviewer1_edit_16_step1_oracle_context.log 2>&1
echo "oracle python rc=$? $(date -Is)"; tail -1 $L/reviewer1_edit_16_step1_oracle_context.log
timeout 3600 python3 -u $Q/tests/bench_longmemeval.py paperjudge $PORT /tmp/nostore $OUT/reviewer1_edit_16_step1_oracle_results.json oraclej 0 $DS > $L/reviewer1_edit_16_step1_judge.log 2>&1
echo "oracle judge rc=$? $(date -Is)"; grep -A4 OFFICIAL $L/reviewer1_edit_16_step1_judge.log | tail -3
stop_all
echo "=== chain end $(date -Is) ==="
touch $L/reviewer1_edit_17_step3_DONE
