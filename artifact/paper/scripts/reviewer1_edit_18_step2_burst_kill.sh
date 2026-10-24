#!/bin/bash
#------------------------------------------------------------
#reviewer1_edit_18_step2_burst_kill.sh
#Purpose : reviewer1_edit_18 step2 on the RTX 4090 host: burst of turns then SIGKILL
#Record  : TEST-LEDGER 104
#------------------------------------------------------------
set -u
B=$NVME_MOUNT/qml-router; R=$B/ieee-rev1; L=$R/logs; OUT=$R/data
SRV=$B/ablation/with-patch/build/bin/llama-server; M=$B/models/Llama-3.2-3B-Instruct-F16.gguf
S=$R/store-burst-kill; PORT=58176; TAG=reviewer1_edit_18_step2_burst_kill
mkdir -p $L $OUT $R/no-mps
exec > >(tee -a $L/$TAG.log) 2>&1
echo "=== $TAG start $(date -Is) ==="
md5sum $SRV $M $R/scripts/$TAG.py
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] || { echo "ABORT: another GPU process is running"; nvidia-smi; exit 2; }
CMDF=$R/scripts/${TAG}_server_cmd.txt
cat > $CMDF <<CMD
$SRV -m $M -ngl 99 -fa on -c 34816 -np 1 -ctk q4_0 -ctv q4_0 --cache-ram 0
CUDA_MPS_PIPE_DIRECTORY=$R/no-mps
CUDA_MPS_LOG_DIRECTORY=$R/no-mps
LD_LIBRARY_PATH=$B/env/lib:/usr/local/cuda/lib64
CMD
SRV_LOG=$L/${TAG}_srv.log python3 -u $R/scripts/$TAG.py $PORT $S $OUT/$TAG.json $CMDF 5 2
echo "python rc=$? $(date -Is)"
echo "leftover llama-server on port $PORT: $(ss -ltnp 2>/dev/null | grep -c ":$PORT ")"
echo "=== $TAG done $(date -Is) ==="
