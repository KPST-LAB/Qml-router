#!/bin/bash
#------------------------------------------------------------
#reviewer1_edit_17_step3b_lex_rerun_h200.sh
#Purpose : step3b: re-run the lex (reproduction) condition with the engine's actual query (question text only), after the step3 chain has finished, and run the official judge for both conditions
#------------------------------------------------------------
set -u
D=$DATA; Q=$D/qml-router; R=$Q/ieee-rev1; L=$R/logs; OUT=$R/data
S=$Q/store-lmes70; DS=$Q/longmemeval_s_cleaned.json
M=$D/models/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf
SRV=$D/llama.cpp/build/bin/llama-server; PORT=58117
exec > >(tee -a $L/reviewer1_edit_17_step3b_lex_rerun_h200.log) 2>&1
while [ ! -f $L/reviewer1_edit_17_step3_DONE ]; do sleep 60; done
echo "=== step3b start $(date -Is) ==="
for i in $(seq 1 60); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break; sleep 5; done
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] || { echo "ABORT: another GPU process is still running after 5 min"; nvidia-smi; exit 2; }
mkdir -p $OUT/step3_lex_with_date; mv $OUT/reviewer1_edit_17_step3_results_lex_* $OUT/step3_lex_with_date/ 2>/dev/null
$SRV -m $M -ngl 99 -fa on -c 32768 -np 2 --port $PORT --qml-session-store $S/qml.db --qml-lmdb-store $S/qml_lmdb </dev/null > $L/reviewer1_edit_17_step3b_reader_srv.log 2>&1 &
RP=$!; trap 'kill $RP 2>/dev/null; wait $RP 2>/dev/null; echo "server stopped pid=$RP $(date -Is)"' EXIT
for i in $(seq 1 900); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 2; done
python3 -u $R/scripts/reviewer1_edit_17_step3_e2e_embedding.py $PORT 0 $OUT lex > $L/reviewer1_edit_17_step3b_lex_rerun.log 2>&1
echo "python rc=$? $(date -Is)"; grep -E "reproduction|string-match" $L/reviewer1_edit_17_step3b_lex_rerun.log
for t in msf trf; do
  mv $OUT/results_step3_lex_${t}.json $OUT/reviewer1_edit_17_step3_results_lex_${t}.json
  for condition in lex emb; do
    timeout 3600 python3 -u $Q/tests/bench_longmemeval.py paperjudge $PORT $S $OUT/reviewer1_edit_17_step3_results_${condition}_${t}.json s3b${condition}${t}j 0 $DS > $L/reviewer1_edit_17_step3b_judge_${condition}_${t}.log 2>&1
    echo "judge $condition $t rc=$? $(date -Is)"; grep -A3 OFFICIAL $L/reviewer1_edit_17_step3b_judge_${condition}_${t}.log | tail -2
  done
done
echo "=== step3b done $(date -Is) ==="; touch $L/reviewer1_edit_17_step3b_DONE
