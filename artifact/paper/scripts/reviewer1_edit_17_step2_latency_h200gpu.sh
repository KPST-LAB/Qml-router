#!/bin/bash
#------------------------------------------------------------
#reviewer1_edit_17_step2_latency_h200gpu.sh
#Purpose : reviewer1_edit_17 step2 on the H200: build the portable data set once (CPU), then time lexical and embedding retrieval with bge-m3 on the H200 GPU
#Record  : TEST-LEDGER 99
#------------------------------------------------------------
set -u
D=$DATA; Q=$D/qml-router; R=$Q/ieee-rev1; L=$R/logs; OUT=$R/data
SRV=$D/llama.cpp/build/bin/llama-server; EMB_MODEL=$D/models/bge-m3-f16.gguf; PORT=58172
TAG=reviewer1_edit_17_step2_latency_h200gpu
exec > >(tee -a $L/$TAG.log) 2>&1
echo "=== $TAG start $(date -Is) ==="
md5sum $SRV $EMB_MODEL $R/scripts/reviewer1_edit_17_step2_*.py
[ -f $OUT/reviewer1_edit_17_step2_subset.db ] || python3 -u $R/scripts/reviewer1_edit_17_step2_export_subset.py || { echo "ABORT: export failed"; exit 1; }
md5sum $OUT/reviewer1_edit_17_step2_subset.db $OUT/reviewer1_edit_17_step2_questions.json
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] || { echo "ABORT: another GPU process is running"; nvidia-smi; exit 2; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
$SRV -m $EMB_MODEL --embedding -ngl 99 -c 8192 -ub 8192 -np 1 --port $PORT </dev/null > $L/${TAG}_srv.log 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; echo "embedding server stopped pid=$SPID $(date -Is)"' EXIT
for i in $(seq 1 300); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 2; done
curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok || { echo "ABORT: embedding server did not come up"; exit 3; }
python3 -u $R/scripts/reviewer1_edit_17_step2_retrieval_latency.py $OUT/reviewer1_edit_17_step2_subset.db $OUT/reviewer1_edit_17_step2_questions.json \
  http://127.0.0.1:$PORT $OUT/$TAG all $OUT/reviewer1_edit_17_step2_vectors_h200gpu
echo "python rc=$? $(date -Is)"
echo "=== $TAG done $(date -Is) ==="
