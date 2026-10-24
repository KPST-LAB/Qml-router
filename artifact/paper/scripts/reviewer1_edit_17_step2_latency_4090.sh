#!/bin/bash
#------------------------------------------------------------
#reviewer1_edit_17_step2_latency_4090.sh
#Purpose : reviewer1_edit_17 step2 on the RTX 4090 host
#Record  : TEST-LEDGER 99
#------------------------------------------------------------
set -u
MODE_DEV=${1:?gpu, gpu_nograph or cpu}
B=$NVME_MOUNT/qml-router; R=$B/ieee-rev1; L=$R/logs; OUT=$R/data
SRV=$B/ablation/with-patch/build/bin/llama-server; EMB_MODEL=$B/models/bge-m3-f16.gguf; PORT=58173
TAG=reviewer1_edit_17_step2_latency_4090$MODE_DEV
mkdir -p $L $OUT $R/no-mps
exec > >(tee -a $L/$TAG.log) 2>&1
echo "=== $TAG start $(date -Is) ==="
md5sum $SRV $EMB_MODEL $R/scripts/reviewer1_edit_17_step2_retrieval_latency.py $OUT/reviewer1_edit_17_step2_subset.db $OUT/reviewer1_edit_17_step2_questions.json
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] || { echo "ABORT: another GPU process is running"; nvidia-smi; exit 2; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; lscpu | grep -E "^Model name|^CPU\(s\)"
GRAPHS=""
case $MODE_DEV in gpu) NGL=99; WMODE=all;; gpu_nograph) NGL=99; WMODE=all; GRAPHS=GGML_CUDA_DISABLE_GRAPHS=1;; cpu) NGL=0; WMODE=sample:10;; *) echo "unknown mode"; exit 1;; esac
env $GRAPHS CUDA_MPS_PIPE_DIRECTORY=$R/no-mps CUDA_MPS_LOG_DIRECTORY=$R/no-mps LD_LIBRARY_PATH=$B/env/lib:/usr/local/cuda/lib64 \
  $SRV -m $EMB_MODEL --embedding -ngl $NGL -c 8192 -ub 8192 -np 1 --port $PORT </dev/null > $L/${TAG}_srv.log 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; echo "embedding server stopped pid=$SPID $(date -Is)"' EXIT
for i in $(seq 1 300); do python3 -c "import urllib.request,sys;sys.exit(0 if b'ok' in urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=2).read() else 1)" 2>/dev/null && break; sleep 2; done
python3 -c "import urllib.request,sys;sys.exit(0 if b'ok' in urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=2).read() else 1)" 2>/dev/null || { echo "ABORT: embedding server did not come up"; tail -20 $L/${TAG}_srv.log; exit 3; }
python3 -u $R/scripts/reviewer1_edit_17_step2_retrieval_latency.py $OUT/reviewer1_edit_17_step2_subset.db $OUT/reviewer1_edit_17_step2_questions.json \
  http://127.0.0.1:$PORT $OUT/$TAG $WMODE $OUT/reviewer1_edit_17_step2_vectors_4090gpu
echo "python rc=$? $(date -Is)"
echo "=== $TAG done $(date -Is) ==="
