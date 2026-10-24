#!/bin/bash
#------------------------------------------------------------
#reviewer1_edit_10_step1_shared_prefix_storage.sh
#Purpose : reviewer1_edit_10 step1 on the RTX 4090 host: storage of a shared prefix under identity keying
#Record  : TEST-LEDGER 103
#------------------------------------------------------------
set -u
B=$NVME_MOUNT/qml-router; R=$B/ieee-rev1; L=$R/logs; OUT=$R/data
SRV=$B/ablation/with-patch/build/bin/llama-server; M=$B/models/Llama-3.2-3B-Instruct-F16.gguf
S=$R/store-shared-prefix; PORT=58175; TAG=reviewer1_edit_10_step1_shared_prefix_storage
mkdir -p $L $OUT $R/no-mps
exec > >(tee -a $L/$TAG.log) 2>&1
echo "=== $TAG start $(date -Is) ==="
md5sum $SRV $M $R/scripts/$TAG.py
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] || { echo "ABORT: another GPU process is running"; nvidia-smi; exit 2; }
rm -rf $S; mkdir -p $S
env CUDA_MPS_PIPE_DIRECTORY=$R/no-mps CUDA_MPS_LOG_DIRECTORY=$R/no-mps LD_LIBRARY_PATH=$B/env/lib:/usr/local/cuda/lib64 \
  $SRV -m $M -ngl 99 -fa on -c 34816 -np 1 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port $PORT \
  --qml-session-store $S/qml.db --qml-lmdb-store $S/qml_lmdb </dev/null > $L/${TAG}_srv.log 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; echo "server stopped pid=$SPID $(date -Is)"' EXIT
for i in $(seq 1 300); do python3 -c "import urllib.request,sys;sys.exit(0 if b'ok' in urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=2).read() else 1)" 2>/dev/null && break; sleep 2; done
python3 -c "import urllib.request,sys;sys.exit(0 if b'ok' in urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=2).read() else 1)" 2>/dev/null || { echo "ABORT: server did not come up"; tail -20 $L/${TAG}_srv.log; exit 3; }
python3 -u $R/scripts/$TAG.py $PORT $S $OUT/$TAG.json 10
echo "python rc=$? $(date -Is)"
sleep 3; du -sb $S/qml_lmdb $S/qml.db
echo "server capture lines: $(grep -c '\[qml\]\[timing\] capture slot' $L/${TAG}_srv.log)"
echo "=== $TAG done $(date -Is) ==="
