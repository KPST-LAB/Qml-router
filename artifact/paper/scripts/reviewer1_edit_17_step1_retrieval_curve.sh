#!/bin/bash
#------------------------------------------------------------
#reviewer1_edit_17_step1_retrieval_curve.sh
#Purpose : reviewer1_edit_17 step1 - reader-free retrieval comparison for reviewer 1, comment 17
#Record  : TEST-LEDGER 98
#------------------------------------------------------------
set -u
D=$DATA; Q=$D/qml-router; R=$Q/ieee-rev1; L=$R/logs
SRV=$D/llama.cpp/build/bin/llama-server; EMB_MODEL=$D/models/bge-m3-f16.gguf; PORT=58172
TAG=reviewer1_edit_17_step1
mkdir -p $L $R/data
exec > >(tee -a $L/${TAG}_retrieval_curve.log) 2>&1
echo "=== $TAG start $(date -Is) ==="
md5sum $SRV $EMB_MODEL $R/scripts/${TAG}_retrieval_curve.py
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] || { echo "ABORT: another GPU process is running"; nvidia-smi; exit 2; }
$SRV -m $EMB_MODEL --embedding -ngl 99 -c 8192 -ub 8192 -np 1 --port $PORT </dev/null > $L/${TAG}_srv_embedding.log 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; echo "embedding server stopped pid=$SPID $(date -Is)"' EXIT
for i in $(seq 1 300); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 2; done
curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok || { echo "ABORT: embedding server did not come up"; exit 3; }
echo "embedding server up pid=$SPID $(date -Is)"
python3 -u $R/scripts/${TAG}_retrieval_curve.py
echo "python rc=$? $(date -Is)"
echo "=== $TAG done $(date -Is) ==="
