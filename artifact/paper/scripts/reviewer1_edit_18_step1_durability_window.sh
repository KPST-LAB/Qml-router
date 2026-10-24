#!/bin/bash
#------------------------------------------------------------
#reviewer1_edit_18_step1_durability_window.sh
#Purpose : reviewer1_edit_18 step1 on the RTX 4090 host: durability window of the asynchronous save
#Record  : TEST-LEDGER 102
#------------------------------------------------------------
set -u
B=$NVME_MOUNT/qml-router; R=$B/ieee-rev1; L=$R/logs; OUT=$R/data
SRV=$B/ablation/with-patch/build/bin/llama-server; M=$B/models/Llama-3.2-3B-Instruct-F16.gguf
S=$R/store-durability; PORT=58174; TAG=reviewer1_edit_18_step1_durability_window
mkdir -p $L $OUT $R/no-mps
exec > >(tee -a $L/$TAG.log) 2>&1
echo "=== $TAG start $(date -Is) ==="
md5sum $SRV $M $R/scripts/$TAG.py
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; cat /sys/block/nvme0n1/device/model 2>/dev/null
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] || { echo "ABORT: another GPU process is running"; nvidia-smi; exit 2; }
rm -rf $S; mkdir -p $S
env CUDA_MPS_PIPE_DIRECTORY=$R/no-mps CUDA_MPS_LOG_DIRECTORY=$R/no-mps LD_LIBRARY_PATH=$B/env/lib:/usr/local/cuda/lib64 \
  $SRV -m $M -ngl 99 -fa on -c 34816 -np 1 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port $PORT \
  --qml-session-store $S/qml.db --qml-lmdb-store $S/qml_lmdb </dev/null > $L/${TAG}_srv.log 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; echo "server stopped pid=$SPID $(date -Is)"' EXIT
for i in $(seq 1 300); do python3 -c "import urllib.request,sys;sys.exit(0 if b'ok' in urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=2).read() else 1)" 2>/dev/null && break; sleep 2; done
python3 -c "import urllib.request,sys;sys.exit(0 if b'ok' in urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=2).read() else 1)" 2>/dev/null || { echo "ABORT: server did not come up"; tail -20 $L/${TAG}_srv.log; exit 3; }
grep -oE "kv_unified = .[a-z]+.|QML_CAPTURE_MIN_DELTA[^)]*" $L/${TAG}_srv.log | head -2
python3 -u $R/scripts/$TAG.py $PORT $S $OUT/$TAG.json 100 nvme0n1
echo "python rc=$? $(date -Is)"
sleep 5; du -sb $S/qml_lmdb $S/qml.db
echo "server capture lines: $(grep -c '\[qml\]\[timing\] capture slot' $L/${TAG}_srv.log), skip lines: $(grep -c 'capture SKIP' $L/${TAG}_srv.log)"
echo "=== $TAG done $(date -Is) ==="
